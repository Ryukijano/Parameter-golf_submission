#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <glob.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include "kernels.cuh"

// ----------------------------------------------------------------------------
// Error checking macros
// ----------------------------------------------------------------------------
#define cudaCheck(err) do {                                           \
    cudaError_t err_ = (err);                                         \
    if (err_ != cudaSuccess) {                                        \
        fprintf(stderr, "CUDA error %d at %s:%d: %s\n",               \
                err_, __FILE__, __LINE__, cudaGetErrorString(err_));  \
        exit(1);                                                      \
    }                                                                 \
} while (0)

#define cublasCheck(err) do {                                         \
    cublasStatus_t err_ = (err);                                      \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                              \
        fprintf(stderr, "CUBLAS error %d at %s:%d\n",                 \
                err_, __FILE__, __LINE__);                            \
        exit(1);                                                      \
    }                                                                 \
} while (0)

// ----------------------------------------------------------------------------
// Hyperparameters
// ----------------------------------------------------------------------------
typedef struct {
    int vocab_size;
    int num_layers;
    int model_dim;
    int num_heads;
    int num_kv_heads;
    int mlp_mult;
    int seq_len;
    int batch_size;
    float rope_base;
    float logit_softcap;
    float qk_gain_init;
} Hyperparameters;

// ----------------------------------------------------------------------------
// Data Loader (mmap)
// ----------------------------------------------------------------------------
typedef struct {
    int fd;
    size_t file_size;
    uint16_t* tokens;
    int num_tokens;
    int pos;
} TokenShard;

typedef struct {
    glob_t glob_result;
    int current_shard_idx;
    TokenShard current_shard;
} DataLoader;

void open_shard(DataLoader* loader, int shard_idx) {
    if (loader->current_shard.fd > 0) {
        munmap(loader->current_shard.tokens - 128, loader->current_shard.file_size); // -128 since we advanced past header (256 ints)
        close(loader->current_shard.fd);
    }
    const char* filename = loader->glob_result.gl_pathv[shard_idx];
    int fd = open(filename, O_RDONLY);
    if (fd < 0) { perror("Failed to open shard"); exit(1); }
    struct stat sb; fstat(fd, &sb);
    size_t file_size = sb.st_size;
    void* map = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
    if (map == MAP_FAILED) { perror("mmap failed"); exit(1); }
    int32_t* header = (int32_t*)map;
    if (header[0] != 20240520 || header[1] != 1) { fprintf(stderr, "Invalid shard header\n"); exit(1); }
    loader->current_shard.fd = fd;
    loader->current_shard.file_size = file_size;
    loader->current_shard.tokens = (uint16_t*)(header + 256);
    loader->current_shard.num_tokens = header[2];
    loader->current_shard.pos = 0;
    loader->current_shard_idx = shard_idx;
}

void init_dataloader(DataLoader* loader, const char* pattern) {
    if (glob(pattern, 0, NULL, &loader->glob_result) != 0 || loader->glob_result.gl_pathc == 0) {
        fprintf(stderr, "No files found for pattern %s\n", pattern); exit(1);
    }
    loader->current_shard.fd = -1;
    open_shard(loader, 0);
}

void next_batch(DataLoader* loader, uint16_t* batch_buf, int batch_tokens) {
    int tokens_read = 0;
    while (tokens_read < batch_tokens) {
        int avail = loader->current_shard.num_tokens - loader->current_shard.pos;
        int to_read = (batch_tokens - tokens_read < avail) ? (batch_tokens - tokens_read) : avail;
        memcpy(batch_buf + tokens_read, loader->current_shard.tokens + loader->current_shard.pos, to_read * sizeof(uint16_t));
        loader->current_shard.pos += to_read;
        tokens_read += to_read;
        if (loader->current_shard.pos >= loader->current_shard.num_tokens) {
            open_shard(loader, (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc);
        }
    }
}

// ----------------------------------------------------------------------------
// Parameters structure
// ----------------------------------------------------------------------------
typedef struct {
    float* tok_emb;        // (vocab_size, model_dim)
    // Layer params
    float** attn_norm_w;   // [num_layers] (model_dim) -- wait, RMSNorm in train_gpt.py has no weights unless specified?
    float** c_q_w;         // [num_layers] (model_dim, model_dim)
    float** c_k_w;         // [num_layers] (kv_dim, model_dim)
    float** c_v_w;         // [num_layers] (kv_dim, model_dim)
    float** proj_w;        // [num_layers] (model_dim, model_dim)
    float** q_gain;        // [num_layers] (num_heads)
    float** mlp_norm_w;    // [num_layers] (model_dim)
    float** fc_w;          // [num_layers] (hidden_dim, model_dim)
    float** mlp_proj_w;    // [num_layers] (model_dim, hidden_dim)
    float** attn_scale;    // [num_layers] (model_dim)
    float** mlp_scale;     // [num_layers] (model_dim)
    float** resid_mix;     // [num_layers] (2, model_dim)
    float* final_norm_w;   // (model_dim)
} GPTParams;

void malloc_params(GPTParams* p, const Hyperparameters* args) {
    // We will allocate one big chunk for all parameters to optimize transfers and memory
    int model_dim = args->model_dim;
    int kv_dim = args->num_kv_heads * (model_dim / args->num_heads);
    int hidden_dim = args->mlp_mult * model_dim;
    
    size_t total_floats = args->vocab_size * model_dim; // tok_emb
    
    // Per layer
    size_t layer_floats = 0;
    layer_floats += model_dim * model_dim; // c_q
    layer_floats += kv_dim * model_dim;    // c_k
    layer_floats += kv_dim * model_dim;    // c_v
    layer_floats += model_dim * model_dim; // proj
    layer_floats += args->num_heads;       // q_gain
    layer_floats += hidden_dim * model_dim; // fc
    layer_floats += model_dim * hidden_dim; // mlp_proj
    layer_floats += model_dim;             // attn_scale
    layer_floats += model_dim;             // mlp_scale
    layer_floats += 2 * model_dim;         // resid_mix
    
    total_floats += args->num_layers * layer_floats;
    
    float* d_params;
    cudaCheck(cudaMalloc(&d_params, total_floats * sizeof(float)));
    cudaCheck(cudaMemset(d_params, 0, total_floats * sizeof(float)));
    
    // Set pointers
    p->tok_emb = d_params; d_params += args->vocab_size * model_dim;
    
    p->c_q_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->c_k_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->c_v_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->proj_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->q_gain = (float**)malloc(args->num_layers * sizeof(float*));
    p->fc_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->mlp_proj_w = (float**)malloc(args->num_layers * sizeof(float*));
    p->attn_scale = (float**)malloc(args->num_layers * sizeof(float*));
    p->mlp_scale = (float**)malloc(args->num_layers * sizeof(float*));
    p->resid_mix = (float**)malloc(args->num_layers * sizeof(float*));
    
    for (int i = 0; i < args->num_layers; i++) {
        p->c_q_w[i] = d_params; d_params += model_dim * model_dim;
        p->c_k_w[i] = d_params; d_params += kv_dim * model_dim;
        p->c_v_w[i] = d_params; d_params += kv_dim * model_dim;
        p->proj_w[i] = d_params; d_params += model_dim * model_dim;
        p->q_gain[i] = d_params; d_params += args->num_heads;
        p->fc_w[i] = d_params; d_params += hidden_dim * model_dim;
        p->mlp_proj_w[i] = d_params; d_params += model_dim * hidden_dim;
        p->attn_scale[i] = d_params; d_params += model_dim;
        p->mlp_scale[i] = d_params; d_params += model_dim;
        p->resid_mix[i] = d_params; d_params += 2 * model_dim;
    }
}

// ----------------------------------------------------------------------------
// Main Training Loop
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
    printf("Starting C/CUDA Parameter Golf Trainer\n");

    Hyperparameters args;
    args.vocab_size = 1024;
    args.num_layers = 11;
    args.model_dim = 512;
    args.num_heads = 8;
    args.num_kv_heads = 4;
    args.mlp_mult = 3;
    args.seq_len = 1024;
    args.batch_size = 8;
    args.rope_base = 10000.0f;
    args.logit_softcap = 30.0f;
    args.qk_gain_init = 1.5f;

    DataLoader loader;
    init_dataloader(&loader, "../data/datasets/fineweb10B_sp1024/fineweb_train_*.bin");
    printf("Found %zu training shards\n", loader.glob_result.gl_pathc);

    GPTParams params;
    malloc_params(&params, &args);
    printf("Allocated parameters on device.\n");

    // Initialize cuBLAS
    cublasHandle_t cublas_handle;
    cublasCheck(cublasCreate(&cublas_handle));

    int batch_tokens = args.batch_size * args.seq_len + 1;
    uint16_t* h_batch_buf = (uint16_t*)malloc(batch_tokens * sizeof(uint16_t));

    for (int step = 0; step < 10; step++) {
        next_batch(&loader, h_batch_buf, batch_tokens);
        printf("Step %d: First token %d\n", step, h_batch_buf[0]);
    }

    free(h_batch_buf);
    cublasCheck(cublasDestroy(cublas_handle));
    return 0;
}
