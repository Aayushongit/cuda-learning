// cuSPARSE SpMV: Sparse Matrix-Vector multiplication
// Essential for graph neural networks, recommendation systems, and scientific computing

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

// Create sparse matrix in CSR format with random sparsity
void generateSparseMatrix(int rows, int cols, float density,
                         int** rowPtr, int** colIdx, float** values, int* nnz) {
    // Count non-zeros
    *nnz = 0;
    int* temp_row = (int*)malloc((rows + 1) * sizeof(int));
    temp_row[0] = 0;

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < density) {
                (*nnz)++;
            }
        }
        temp_row[i + 1] = *nnz;
    }

    // Allocate CSR arrays
    *rowPtr = (int*)malloc((rows + 1) * sizeof(int));
    *colIdx = (int*)malloc(*nnz * sizeof(int));
    *values = (float*)malloc(*nnz * sizeof(float));

    memcpy(*rowPtr, temp_row, (rows + 1) * sizeof(int));

    // Fill values and column indices
    int idx = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if ((float)rand() / RAND_MAX < density) {
                (*colIdx)[idx] = j;
                (*values)[idx] = (float)(rand() % 100) / 10.0f;
                idx++;
            }
        }
    }

    free(temp_row);
}

int main() {
    const int rows = 10000;
    const int cols = 10000;
    const float density = 0.01f;  // 1% sparse (99% zeros)

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Generate sparse matrix in CSR format
    int *h_csrRowPtr, *h_csrColIdx;
    float *h_csrValues;
    int nnz;

    printf("Generating sparse matrix (%dx%d, %.1f%% density)...\n", rows, cols, density * 100);
    generateSparseMatrix(rows, cols, density, &h_csrRowPtr, &h_csrColIdx, &h_csrValues, &nnz);

    printf("Non-zero elements: %d (%.2f%% of total)\n", nnz, (float)nnz / (rows * cols) * 100);

    // Generate dense vector
    float *h_x = (float*)malloc(cols * sizeof(float));
    float *h_y = (float*)malloc(rows * sizeof(float));

    for (int i = 0; i < cols; i++) {
        h_x[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device memory
    int *d_csrRowPtr, *d_csrColIdx;
    float *d_csrValues, *d_x, *d_y;

    CHECK_CUDA(cudaMalloc(&d_csrRowPtr, (rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrColIdx, nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_csrValues, nnz * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, rows * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_csrRowPtr, h_csrRowPtr, (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrColIdx, h_csrColIdx, nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_csrValues, h_csrValues, nnz * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, h_x, cols * sizeof(float), cudaMemcpyHostToDevice));

    // Create cuSPARSE handle
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, rows, cols, nnz,
                                     d_csrRowPtr, d_csrColIdx, d_csrValues,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // Create dense vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, cols, d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, rows, d_y, CUDA_R_32F));

    // Allocate buffer for SpMV
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matA, vecX, &beta, vecY,
                                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                           &bufferSize));

    void* buffer = NULL;
    CHECK_CUDA(cudaMalloc(&buffer, bufferSize));

    // Warm-up
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Perform SpMV: y = alpha * A * x + beta * y
    CHECK_CUSPARSE(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                               &alpha, matA, vecX, &beta, vecY,
                               CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result
    CHECK_CUDA(cudaMemcpy(h_y, d_y, rows * sizeof(float), cudaMemcpyDeviceToHost));

    // Performance metrics
    double gflops = (2.0 * nnz * 1e-9) / (milliseconds / 1000.0);
    double bandwidth = (nnz * (sizeof(float) + sizeof(int)) + rows * sizeof(int) +
                       (rows + cols) * sizeof(float)) * 1e-9 / (milliseconds / 1000.0);

    printf("\n=== SpMV Performance ===\n");
    printf("Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    printf("Bandwidth: %.2f GB/s\n", bandwidth);
    printf("Throughput: %.2f M nonzeros/sec\n", nnz / (milliseconds * 1000.0));

    // Display sample results
    printf("\nInput vector x (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_x[i]);
    }

    printf("\n\nOutput vector y (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_y[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUSPARSE(cusparseDestroySpMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecX));
    CHECK_CUSPARSE(cusparseDestroyDnVec(vecY));
    CHECK_CUSPARSE(cusparseDestroy(handle));

    CHECK_CUDA(cudaFree(d_csrRowPtr));
    CHECK_CUDA(cudaFree(d_csrColIdx));
    CHECK_CUDA(cudaFree(d_csrValues));
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));
    CHECK_CUDA(cudaFree(buffer));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_csrRowPtr);
    free(h_csrColIdx);
    free(h_csrValues);
    free(h_x);
    free(h_y);

    printf("\ncuSPARSE SpMV completed successfully!\n");
    return 0;
}
