// cuBLAS GEMM: General Matrix Multiplication C = alpha*A*B + beta*C
// Essential for neural network dense layers and transformer models

#include <cuda_runtime.h>
#include <cublas_v2.h>
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

void initMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)(rand() % 100) / 10.0f;
    }
}

void printMatrix(const char* name, float* mat, int rows, int cols, int maxDisplay = 4) {
    printf("%s:\n", name);
    int displayRows = (rows < maxDisplay) ? rows : maxDisplay;
    int displayCols = (cols < maxDisplay) ? cols : maxDisplay;

    for (int i = 0; i < displayRows; i++) {
        for (int j = 0; j < displayCols; j++) {
            printf("%.2f ", mat[i * cols + j]);
        }
        if (cols > maxDisplay) printf("...");
        printf("\n");
    }
    if (rows > maxDisplay) printf("...\n");
    printf("\n");
}

int main() {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    const int M = 512;  // Rows of A and C
    const int K = 512;  // Cols of A, Rows of B
    const int N = 512;  // Cols of B and C

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // Host memory allocation
    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    // Initialize matrices
    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    // Device memory allocation
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_A));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_B));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_C));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Warm-up
    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Perform GEMM: Note column-major ordering (cuBLAS uses Fortran ordering)
    CHECK_CUBLAS(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            N, M, K,
                            &alpha,
                            d_B, N,
                            d_A, K,
                            &beta,
                            d_C, N));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost));

    // Calculate GFLOPS
    double gflops = (2.0 * M * N * K * 1e-9) / (milliseconds / 1000.0);

    printf("Matrix Multiplication: C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n\n", gflops);

    printMatrix("Matrix A (sample)", h_A, M, K);
    printMatrix("Matrix B (sample)", h_B, K, N);
    printMatrix("Matrix C (sample result)", h_C, M, N);

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_A);
    free(h_B);
    free(h_C);

    printf("cuBLAS GEMM completed successfully!\n");
    return 0;
}
