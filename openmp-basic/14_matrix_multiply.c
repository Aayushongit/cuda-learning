// Matrix multiplication - performance comparison
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 512

void matrix_multiply_serial(double *A, double *B, double *C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void matrix_multiply_parallel(double *A, double *B, double *C, int n) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    printf("Matrix size: %dx%d\n", N, N);
    printf("Threads available: %d\n\n", omp_get_max_threads());

    // Serial version
    double start = omp_get_wtime();
    matrix_multiply_serial(A, B, C, N);
    double end = omp_get_wtime();
    printf("Serial time: %.3f seconds\n", end - start);

    // Parallel version
    start = omp_get_wtime();
    matrix_multiply_parallel(A, B, C, N);
    end = omp_get_wtime();
    printf("Parallel time: %.3f seconds\n", end - start);

    free(A);
    free(B);
    free(C);

    return 0;
}
