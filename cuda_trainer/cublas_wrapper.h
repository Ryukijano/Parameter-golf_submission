#pragma once
#include <cublas_v2.h>
#include <cublasLt.h>

void matmul_forward_cublas(cublasHandle_t cublas_handle, float* out, const float* inp, const float* weight, int B, int T, int C, int OC) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // weight is (OC, C), inp is (B*T, C), out is (B*T, OC)
    // cublas assumes column-major, so we compute out^T = weight * inp^T
    // out^T is (OC, B*T)
    cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B * T, C, &alpha, weight, C, inp, C, &beta, out, OC));
}
