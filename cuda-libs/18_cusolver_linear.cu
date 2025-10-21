// cuSOLVER: Linear system solvers and matrix decompositions
// Essential for optimization, least squares, and scientific computing

#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSOLVER Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void generateSymmetricPositiveDefinite(float* A, int n) {
    // Generate random matrix and make it symmetric positive definite
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (float)(rand() % 100) / 100.0f;
        }
        A[i * n + i] += n;  // Make diagonally dominant
    }

    // Symmetrize
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            A[j * n + i] = A[i * n + j];
        }
    }
}

int main() {
    const int n = 1024;  // Matrix size (n x n)

    printf("=== cuSOLVER Linear System Solver ===\n");
    printf("Matrix size: %dx%d\n\n", n, n);

    size_t matrix_bytes = n * n * sizeof(float);
    size_t vector_bytes = n * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(matrix_bytes);
    float *h_b = (float*)malloc(vector_bytes);
    float *h_x = (float*)malloc(vector_bytes);

    // Generate symmetric positive definite matrix
    generateSymmetricPositiveDefinite(h_A, n);

    // Generate right-hand side vector
    for (int i = 0; i < n; i++) {
        h_b[i] = (float)(rand() % 100) / 10.0f;
    }

    // Device memory
    float *d_A, *d_b;
    int *d_info;
    CHECK_CUDA(cudaMalloc(&d_A, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_b, vector_bytes));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, matrix_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, vector_bytes, cudaMemcpyHostToDevice));

    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    CHECK_CUSOLVER(cusolverDnCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. LU DECOMPOSITION AND SOLVE
    int *d_pivot;
    CHECK_CUDA(cudaMalloc(&d_pivot, n * sizeof(int)));

    int lwork_lu = 0;
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(handle, n, n, d_A, n, &lwork_lu));

    float *d_work_lu;
    CHECK_CUDA(cudaMalloc(&d_work_lu, lwork_lu * sizeof(float)));

    // Copy A for LU (will be overwritten)
    float *d_A_lu;
    CHECK_CUDA(cudaMalloc(&d_A_lu, matrix_bytes));
    CHECK_CUDA(cudaMemcpy(d_A_lu, d_A, matrix_bytes, cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaEventRecord(start));

    // LU factorization
    CHECK_CUSOLVER(cusolverDnSgetrf(handle, n, n, d_A_lu, n, d_work_lu, d_pivot, d_info));

    // Solve using LU
    CHECK_CUSOLVER(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, 1, d_A_lu, n, d_pivot, d_b, n, d_info));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_lu = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_lu, start, stop));

    // Copy solution
    CHECK_CUDA(cudaMemcpy(h_x, d_b, vector_bytes, cudaMemcpyDeviceToHost));

    // 2. CHOLESKY DECOMPOSITION (for symmetric positive definite)
    float *d_A_chol;
    CHECK_CUDA(cudaMalloc(&d_A_chol, matrix_bytes));
    CHECK_CUDA(cudaMemcpy(d_A_chol, d_A, matrix_bytes, cudaMemcpyDeviceToDevice));

    // Reload b (was overwritten)
    CHECK_CUDA(cudaMemcpy(d_b, h_b, vector_bytes, cudaMemcpyHostToDevice));

    int lwork_chol = 0;
    CHECK_CUSOLVER(cusolverDnSpotrf_bufferSize(handle, CUBLAS_FILL_MODE_LOWER, n, d_A_chol, n, &lwork_chol));

    float *d_work_chol;
    CHECK_CUDA(cudaMalloc(&d_work_chol, lwork_chol * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));

    // Cholesky factorization
    CHECK_CUSOLVER(cusolverDnSpotrf(handle, CUBLAS_FILL_MODE_LOWER, n, d_A_chol, n,
                                    d_work_chol, lwork_chol, d_info));

    // Solve using Cholesky
    CHECK_CUSOLVER(cusolverDnSpotrs(handle, CUBLAS_FILL_MODE_LOWER, n, 1, d_A_chol, n, d_b, n, d_info));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_chol = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_chol, start, stop));

    // 3. QR DECOMPOSITION
    float *d_A_qr, *d_tau;
    CHECK_CUDA(cudaMalloc(&d_A_qr, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_tau, vector_bytes));
    CHECK_CUDA(cudaMemcpy(d_A_qr, d_A, matrix_bytes, cudaMemcpyDeviceToDevice));

    int lwork_qr = 0;
    CHECK_CUSOLVER(cusolverDnSgeqrf_bufferSize(handle, n, n, d_A_qr, n, &lwork_qr));

    float *d_work_qr;
    CHECK_CUDA(cudaMalloc(&d_work_qr, lwork_qr * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));

    // QR factorization
    CHECK_CUSOLVER(cusolverDnSgeqrf(handle, n, n, d_A_qr, n, d_tau, d_work_qr, lwork_qr, d_info));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_qr = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_qr, start, stop));

    // 4. SINGULAR VALUE DECOMPOSITION (SVD)
    float *d_A_svd, *d_S, *d_U, *d_VT;
    CHECK_CUDA(cudaMalloc(&d_A_svd, matrix_bytes));
    CHECK_CUDA(cudaMalloc(&d_S, vector_bytes));      // Singular values
    CHECK_CUDA(cudaMalloc(&d_U, matrix_bytes));      // Left singular vectors
    CHECK_CUDA(cudaMalloc(&d_VT, matrix_bytes));     // Right singular vectors (transposed)

    CHECK_CUDA(cudaMemcpy(d_A_svd, d_A, matrix_bytes, cudaMemcpyDeviceToDevice));

    int lwork_svd = 0;
    CHECK_CUSOLVER(cusolverDnSgesvd_bufferSize(handle, n, n, &lwork_svd));

    float *d_work_svd;
    int *d_info_svd;
    CHECK_CUDA(cudaMalloc(&d_work_svd, lwork_svd * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_info_svd, sizeof(int)));

    CHECK_CUDA(cudaEventRecord(start));

    // SVD computation
    CHECK_CUSOLVER(cusolverDnSgesvd(handle, 'A', 'A', n, n, d_A_svd, n, d_S,
                                    d_U, n, d_VT, n, d_work_svd, lwork_svd,
                                    NULL, d_info_svd));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_svd = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_svd, start, stop));

    // Copy singular values
    float *h_S = (float*)malloc(vector_bytes);
    CHECK_CUDA(cudaMemcpy(h_S, d_S, vector_bytes, cudaMemcpyDeviceToHost));

    // Performance metrics
    double gflops_lu = (2.0 * n * n * n / 3.0 * 1e-9) / (ms_lu / 1000.0);
    double gflops_chol = (n * n * n / 3.0 * 1e-9) / (ms_chol / 1000.0);
    double gflops_qr = (2.0 * n * n * n / 3.0 * 1e-9) / (ms_qr / 1000.0);
    double gflops_svd = (11.0 * n * n * n * 1e-9) / (ms_svd / 1000.0);

    printf("=== Performance ===\n");
    printf("LU Decomposition:       %.3f ms (%.2f GFLOPS)\n", ms_lu, gflops_lu);
    printf("Cholesky Decomposition: %.3f ms (%.2f GFLOPS)\n", ms_chol, gflops_chol);
    printf("QR Decomposition:       %.3f ms (%.2f GFLOPS)\n", ms_qr, gflops_qr);
    printf("SVD:                    %.3f ms (%.2f GFLOPS)\n", ms_svd, gflops_svd);

    printf("\n=== Solution Sample (first 10 elements) ===\n");
    for (int i = 0; i < 10; i++) {
        printf("x[%d] = %.4f\n", i, h_x[i]);
    }

    printf("\n=== Singular Values (first 10) ===\n");
    for (int i = 0; i < 10; i++) {
        printf("S[%d] = %.4f\n", i, h_S[i]);
    }

    printf("\nCondition number estimate: %.2e\n", h_S[0] / h_S[n-1]);

    // Cleanup
    CHECK_CUSOLVER(cusolverDnDestroy(handle));

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_info));
    CHECK_CUDA(cudaFree(d_pivot));
    CHECK_CUDA(cudaFree(d_A_lu));
    CHECK_CUDA(cudaFree(d_work_lu));
    CHECK_CUDA(cudaFree(d_A_chol));
    CHECK_CUDA(cudaFree(d_work_chol));
    CHECK_CUDA(cudaFree(d_A_qr));
    CHECK_CUDA(cudaFree(d_tau));
    CHECK_CUDA(cudaFree(d_work_qr));
    CHECK_CUDA(cudaFree(d_A_svd));
    CHECK_CUDA(cudaFree(d_S));
    CHECK_CUDA(cudaFree(d_U));
    CHECK_CUDA(cudaFree(d_VT));
    CHECK_CUDA(cudaFree(d_work_svd));
    CHECK_CUDA(cudaFree(d_info_svd));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_A);
    free(h_b);
    free(h_x);
    free(h_S);

    printf("\ncuSOLVER operations completed successfully!\n");
    return 0;
}
