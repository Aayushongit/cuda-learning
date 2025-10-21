// cuBLAS Batched Matrix Operations: Process multiple matrices simultaneously
// Critical for transformer attention mechanisms and batch processing in AI

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

int main() {
    const int M = 128;
    const int K = 128;
    const int N = 128;
    const int batchCount = 64;  // Number of matrices in batch

    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_per_matrix_A = M * K * sizeof(float);
    size_t bytes_per_matrix_B = K * N * sizeof(float);
    size_t bytes_per_matrix_C = M * N * sizeof(float);

    // Allocate host memory for batched matrices
    float *h_A = (float*)malloc(bytes_per_matrix_A * batchCount);
    float *h_B = (float*)malloc(bytes_per_matrix_B * batchCount);
    float *h_C = (float*)malloc(bytes_per_matrix_C * batchCount);

    // Initialize with different values for each batch
    for (int batch = 0; batch < batchCount; batch++) {
        for (int i = 0; i < M * K; i++) {
            h_A[batch * M * K + i] = (float)(rand() % 100) / 10.0f;
        }
        for (int i = 0; i < K * N; i++) {
            h_B[batch * K * N + i] = (float)(rand() % 100) / 10.0f;
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes_per_matrix_A * batchCount));
    CHECK_CUDA(cudaMalloc(&d_B, bytes_per_matrix_B * batchCount));
    CHECK_CUDA(cudaMalloc(&d_C, bytes_per_matrix_C * batchCount));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes_per_matrix_A * batchCount, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes_per_matrix_B * batchCount, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Create array of pointers for each matrix in batch
    float **d_A_array, **d_B_array, **d_C_array;
    float **h_A_array = (float**)malloc(batchCount * sizeof(float*));
    float **h_B_array = (float**)malloc(batchCount * sizeof(float*));
    float **h_C_array = (float**)malloc(batchCount * sizeof(float*));

    // Set up pointer arrays
    for (int i = 0; i < batchCount; i++) {
        h_A_array[i] = d_A + i * M * K;
        h_B_array[i] = d_B + i * K * N;
        h_C_array[i] = d_C + i * M * N;
    }

    CHECK_CUDA(cudaMalloc(&d_A_array, batchCount * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_B_array, batchCount * sizeof(float*)));
    CHECK_CUDA(cudaMalloc(&d_C_array, batchCount * sizeof(float*)));

    CHECK_CUDA(cudaMemcpy(d_A_array, h_A_array, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_array, h_B_array, batchCount * sizeof(float*), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_C_array, h_C_array, batchCount * sizeof(float*), cudaMemcpyHostToDevice));

    // Warm-up
    CHECK_CUBLAS(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    (const float**)d_B_array, N,
                                    (const float**)d_A_array, K,
                                    &beta,
                                    d_C_array, N,
                                    batchCount));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Batched GEMM operation
    CHECK_CUBLAS(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    N, M, K,
                                    &alpha,
                                    (const float**)d_B_array, N,
                                    (const float**)d_A_array, K,
                                    &beta,
                                    d_C_array, N,
                                    batchCount));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results back
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes_per_matrix_C * batchCount, cudaMemcpyDeviceToHost));

    // Calculate performance
    double total_ops = 2.0 * M * N * K * batchCount;
    double gflops = (total_ops * 1e-9) / (milliseconds / 1000.0);

    printf("Batched Matrix Multiplication:\n");
    printf("Batch size: %d\n", batchCount);
    printf("Matrix dimensions: C(%dx%d) = A(%dx%d) * B(%dx%d)\n", M, N, M, K, K, N);
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Average time per matrix: %.3f ms\n\n", milliseconds / batchCount);

    // Display sample results from first and last batch
    printf("First batch result (sample 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    printf("\nLast batch result (sample 4x4):\n");
    int lastBatch = (batchCount - 1) * M * N;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_C[lastBatch + i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(d_A_array));
    CHECK_CUDA(cudaFree(d_B_array));
    CHECK_CUDA(cudaFree(d_C_array));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_A_array);
    free(h_B_array);
    free(h_C_array);

    printf("\ncuBLAS Batched operations completed successfully!\n");
    return 0;
}
