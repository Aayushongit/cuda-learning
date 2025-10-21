// cuSPARSE SpMM: Sparse Matrix-Matrix multiplication
// Used in graph convolutions, attention mechanisms, and large-scale ML models

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSPARSE(call) { \
    cusparseStatus_t status = call; \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    // Matrix dimensions: C(M x N) = A(M x K) * B(K x N)
    const int M = 4096;
    const int K = 4096;
    const int N = 128;  // Multiple right-hand sides (as in batch operations)
    const float density = 0.05f;  // 5% sparse

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Generate sparse matrix A in CSR format
    int nnz = (int)(M * K * density);
    int *h_csrRowPtr = (int*)malloc((M + 1) * sizeof(int));
    int *h_csrColIdx = (int*)malloc(nnz * sizeof(int));
    float *h_csrValues = (float*)malloc(nnz * sizeof(float));

    // Simple sparse pattern generation
    int nnz_per_row = nnz / M;
    h_csrRowPtr[0] = 0;
    for (int i = 0; i < M; i++) {
        h_csrRowPtr[i + 1] = h_csrRowPtr[i] + nnz_per_row;
    }
    h_csrRowPtr[M] = nnz;  // Adjust last element

    for (int i = 0; i < nnz; i++) {
        h_csrColIdx[i] = rand() % K;
        h_csrValues[i] = (float)(rand() % 100) / 10.0f;
    }

    // Generate dense matrix B
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < K * N; i++) {
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device memory
    int *d_csrRowPtr, *d_csrColIdx;
    float *d_csrValues, *d_B, *d_C;

    CHECK_CUDA(cudaMalloc(&d_csrRowPtr, (M + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrColIdx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrValues, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_C, M * N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColIdx, h_csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrValues, h_csrValues, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix A
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, M, K, nnz,
                                     d_csrRowPtr, d_csrColIdx, d_csrValues,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense matrices B and C
    cusparseDnMatDescr_t matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    // Allocate buffer
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(handle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, matB, &beta, matC,
                                           CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                                           &bufferSize));

    void* buffer = NULL;
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    // Warm-up
    CHECK_CUSPARSE(cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                               buffer));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // SpMM: C = alpha * A * B + beta * C
    CHECK_CUSPARSE(cusparseSpMM(handle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, matB, &beta, matC,
                               CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT,
                               buffer));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result
    CHECK_CUDA(cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

    // Performance metrics
    double gflops = (2.0 * nnz * N * 1e-9) / (milliseconds / 1000.0);

    printf("=== Sparse Matrix-Matrix Multiplication ===\n");
    printf("C(%dx%d) = A(%dx%d, %.1f%% sparse) * B(%dx%d)\n",
           M, N, M, K, density * 100, K, N);
    printf("Non-zeros in A: %d\n", nnz);
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);

    // Display sample result
    printf("\nResult C (4x4 corner):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.2f ", h_C[i * N + j]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matB));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matC));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColIdx));
    CHECK_CUDA(cudaFree(d_csrValues));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_csrRowPtr);
    free(h_csrColIdx);
    free(h_csrValues);
    free(h_B);
    free(h_C);

    printf("\ncuSPARSE SpMM completed successfully!\n");
    return 0;
}
