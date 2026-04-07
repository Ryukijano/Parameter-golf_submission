NVCC = nvcc
ARCH = -arch=sm_90
CFLAGS = -O3 --use_fast_math $(ARCH) -std=c++17 -diag-suppress 177,550,1650
BF16_CFLAGS = -O3 --use_fast_math $(ARCH) -std=c++17 -diag-suppress 177,550,1650 -DENABLE_BF16
LDFLAGS = -lcublas -lcublasLt -lm

all: train_gpt_cuda

train_gpt_cuda: train_gpt_cuda.cu
	$(NVCC) $(CFLAGS) $< -o $@ $(LDFLAGS)

train_gpt_cuda_bf16: train_gpt_cuda.cu
	$(NVCC) $(BF16_CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f train_gpt_cuda train_gpt_cuda_bf16
