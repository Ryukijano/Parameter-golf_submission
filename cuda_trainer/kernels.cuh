#pragma once
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

// Helper macros
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ----------------------------------------------------------------------------
// RMSNorm
// ----------------------------------------------------------------------------
__global__ void rmsnorm_forward_kernel(float* out, const float* inp, const float* weight, float eps, int N, int C) {
    int idx = blockIdx.x; // token index
    if (idx >= N) return;
    
    int tid = threadIdx.x;
    const float* x = inp + idx * C;
    float* y = out + idx * C;
    
    // Sum of squares
    extern __shared__ float shared_sum[];
    float sum_sq = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        sum_sq += x[i] * x[i];
    }
    
    // Warp-level reduction
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum_sq += warp.shfl_down(sum_sq, offset);
    }
    
    if (warp.thread_rank() == 0) {
        shared_sum[warp.meta_group_rank()] = sum_sq;
    }
    block.sync();
    
    sum_sq = 0.0f;
    if (warp.meta_group_rank() == 0 && warp.thread_rank() < warp.meta_group_size()) {
        sum_sq = shared_sum[warp.thread_rank()];
    }
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        sum_sq += warp.shfl_down(sum_sq, offset);
    }
    
    __shared__ float rms;
    if (tid == 0) {
        rms = rsqrtf(sum_sq / C + eps);
    }
    block.sync();
    
    for (int i = tid; i < C; i += blockDim.x) {
        float w = weight ? weight[i] : 1.0f;
        y[i] = x[i] * rms * w;
    }
}

// ----------------------------------------------------------------------------
// ReLU^2
// ----------------------------------------------------------------------------
__global__ void relu_sq_forward_kernel(float* out, const float* inp, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = inp[i];
        float val = fmaxf(0.0f, x);
        out[i] = val * val;
    }
}

// ----------------------------------------------------------------------------
// RoPE (Rotary Positional Embedding)
// ----------------------------------------------------------------------------
__global__ void rope_forward_kernel(float* q, float* k, int B, int T, int num_heads, int num_kv_heads, int head_dim, float base) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_q = B * T * num_heads * (head_dim / 2);
    int total_k = B * T * num_kv_heads * (head_dim / 2);
    
    if (idx < total_q) {
        // compute q RoPE
        int d = idx % (head_dim / 2);
        int h = (idx / (head_dim / 2)) % num_heads;
        int t = (idx / (head_dim / 2) / num_heads) % T;
        int b = (idx / (head_dim / 2) / num_heads / T);
        
        float inv_freq = 1.0f / powf(base, (float)(d * 2) / head_dim);
        float freq = t * inv_freq;
        float cos_val = cosf(freq);
        float sin_val = sinf(freq);
        
        int q_idx1 = b * (T * num_heads * head_dim) + t * (num_heads * head_dim) + h * head_dim + d;
        int q_idx2 = q_idx1 + (head_dim / 2);
        
        float q1 = q[q_idx1];
        float q2 = q[q_idx2];
        
        q[q_idx1] = q1 * cos_val - q2 * sin_val;
        q[q_idx2] = q1 * sin_val + q2 * cos_val;
    }
    
    if (idx < total_k) {
        // compute k RoPE
        int d = idx % (head_dim / 2);
        int h = (idx / (head_dim / 2)) % num_kv_heads;
        int t = (idx / (head_dim / 2) / num_kv_heads) % T;
        int b = (idx / (head_dim / 2) / num_kv_heads / T);
        
        float inv_freq = 1.0f / powf(base, (float)(d * 2) / head_dim);
        float freq = t * inv_freq;
        float cos_val = cosf(freq);
        float sin_val = sinf(freq);
        
        int k_idx1 = b * (T * num_kv_heads * head_dim) + t * (num_kv_heads * head_dim) + h * head_dim + d;
        int k_idx2 = k_idx1 + (head_dim / 2);
        
        float k1 = k[k_idx1];
        float k2 = k[k_idx2];
        
        k[k_idx1] = k1 * cos_val - k2 * sin_val;
        k[k_idx2] = k1 * sin_val + k2 * cos_val;
    }
}
