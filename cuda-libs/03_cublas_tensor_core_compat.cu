// cuBLAS with Tensor Cores: Mixed precision (FP16) - Colab Compatible Version
// Compatible with CUDA 10.x and above

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void initMatrixFP32(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

void convertFP32toFP16(float* fp32, __half* fp16, int size) {
    for (int i = 0; i < size; i++) {
        fp16[i] = __float2half(fp32[i]);
    }
}

int main() {
    // Check for Tensor Core capability
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);

    if (prop.major < 7) {
        printf("Warning: Tensor Cores require compute capability >= 7.0\n");
        printf("This GPU will run in compatibility mode (no Tensor Cores)\n\n");
    }

    // Matrix dimensions optimized for Tensor Cores (multiples of 8)
    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    size_t bytes_A_fp16 = M * K * sizeof(__half);
    size_t bytes_B_fp16 = K * N * sizeof(__half);

    // Host memory
    float *h_A_fp32 = (float*)malloc(bytes_A);
    float *h_B_fp32 = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);
    __half *h_A_fp16 = (__half*)malloc(bytes_A_fp16);
    __half *h_B_fp16 = (__half*)malloc(bytes_B_fp16);

    // Initialize and convert to FP16
    initMatrixFP32(h_A_fp32, M, K);
    initMatrixFP32(h_B_fp32, K, N);
    convertFP32toFP16(h_A_fp32, h_A_fp16, M * K);
    convertFP32toFP16(h_B_fp32, h_B_fp16, K * N);

    // Device memory
    __half *d_A, *d_B;
    float *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A_fp16));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B_fp16));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_fp16, bytes_A_fp16, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_fp16, bytes_B_fp16, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Try to enable Tensor Core operations (will gracefully degrade if not available)
#if CUDART_VERSION >= 9000
    // CUDA 9.0+ supports mixed precision
    cublasStatus_t math_status = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    if (math_status != CUBLAS_STATUS_SUCCESS) {
        printf("Note: Tensor Core math mode not available, using default\n");
        CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
    }
#else
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
#endif

    // Warm-up
    cublasGemmEx(handle,
                 CUBLAS_OP_N, CUBLAS_OP_N,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16F, N,
                 d_A, CUDA_R_16F, K,
                 &beta,
                 d_C, CUDA_R_32F, N,
                 CUDA_R_32F,
                 CUBLAS_GEMM_DEFAULT);

    // Timing for Tensor Core GEMM
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // FP16 input, FP32 output
    CHECK_CUBLAS(cublasGemmEx(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             d_B, CUDA_R_16F, N,
                             d_A, CUDA_R_16F, K,
                             &beta,
                             d_C, CUDA_R_32F, N,
                             CUDA_R_32F,
                             CUBLAS_GEMM_DEFAULT));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tensor = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tensor, start, stop));

    // Compare with standard FP32 computation
    float *d_A_fp32, *d_B_fp32, *d_C_fp32;
    CHECK_CUDA(cudaMalloc(&d_A_fp32, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B_fp32, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C_fp32, bytes_C));

    CHECK_CUDA(cudaMemcpy(d_A_fp32, h_A_fp32, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_fp32, h_B_fp32, bytes_B, cudaMemcpyHostToDevice));

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B_fp32, N,
                            d_A_fp32, K,
                            &beta,
                            d_C_fp32, N));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_fp32 = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_fp32, start, stop));

    // Copy results
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Performance metrics
    double gflops_tensor = (2.0 * M * N * K * 1e-9) / (ms_tensor / 1000.0);
    double gflops_fp32 = (2.0 * M * N * K * 1e-9) / (ms_fp32 / 1000.0);

    printf("\n=== Performance Comparison ===\n");
    printf("Matrix: %dx%dx%d\n\n", M, K, N);

    printf("Mixed Precision (FP16 input):\n");
    printf("  Time: %.3f ms\n", ms_tensor);
    printf("  Performance: %.2f TFLOPS\n\n", gflops_tensor / 1000.0);

    printf("Standard (FP32):\n");
    printf("  Time: %.3f ms\n", ms_fp32);
    printf("  Performance: %.2f TFLOPS\n\n", gflops_fp32 / 1000.0);

    printf("Speedup: %.2fx\n\n", ms_fp32 / ms_tensor);

    // Sample output
    printf("Result sample (4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_fp32));
    CHECK_CUDA(cudaFree(d_B_fp32));
    CHECK_CUDA(cudaFree(d_C_fp32));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_A_fp32);
    free(h_B_fp32);
    free(h_C);
    free(h_A_fp16);
    free(h_B_fp16);

    printf("\nTensor Core operations completed successfully!\n");
    return 0;
}
