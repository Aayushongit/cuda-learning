// SIMD vectorization directives
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10000000

void demo_basic_simd() {
    float *a = malloc(N * sizeof(float));
    float *b = malloc(N * sizeof(float));
    float *c = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        a[i] = i * 0.5f;
        b[i] = i * 0.3f;
    }

    printf("Basic SIMD vectorization:\n");

    // Without SIMD
    double start = omp_get_wtime();
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    double time_no_simd = omp_get_wtime() - start;
    printf("  Without SIMD: %.4f seconds\n", time_no_simd);

    // With SIMD
    start = omp_get_wtime();
    #pragma omp simd
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }
    double time_simd = omp_get_wtime() - start;
    printf("  With SIMD:    %.4f seconds\n", time_simd);
    printf("  Speedup:      %.2fx\n\n", time_no_simd / time_simd);

    free(a); free(b); free(c);
}

void demo_parallel_simd() {
    double *x = malloc(N * sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *z = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        x[i] = i * 1.5;
        y[i] = i * 2.3;
    }

    printf("Combined parallel for simd:\n");

    // Just parallel
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        z[i] = x[i] * y[i] + x[i];
    }
    double time_parallel = omp_get_wtime() - start;
    printf("  Parallel only:        %.4f seconds\n", time_parallel);

    // Parallel + SIMD
    start = omp_get_wtime();
    #pragma omp parallel for simd
    for (int i = 0; i < N; i++) {
        z[i] = x[i] * y[i] + x[i];
    }
    double time_both = omp_get_wtime() - start;
    printf("  Parallel + SIMD:      %.4f seconds\n", time_both);
    printf("  Additional speedup:   %.2fx\n\n", time_parallel / time_both);

    free(x); free(y); free(z);
}

void demo_simd_reduction() {
    double *data = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        data[i] = i * 0.1;
    }

    printf("SIMD with reduction:\n");

    double sum1 = 0.0;
    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum1)
    for (int i = 0; i < N; i++) {
        sum1 += data[i];
    }
    double time1 = omp_get_wtime() - start;
    printf("  Parallel reduction:      %.4f seconds, sum=%.2f\n", time1, sum1);

    double sum2 = 0.0;
    start = omp_get_wtime();
    #pragma omp parallel for simd reduction(+:sum2)
    for (int i = 0; i < N; i++) {
        sum2 += data[i];
    }
    double time2 = omp_get_wtime() - start;
    printf("  Parallel+SIMD reduction: %.4f seconds, sum=%.2f\n", time2, sum2);
    printf("  Speedup: %.2fx\n\n", time1 / time2);

    free(data);
}

void demo_simd_safelen() {
    int a[100];
    for (int i = 0; i < 100; i++) a[i] = i;

    printf("SIMD safelen (safe vector length):\n");

    // Without safelen - compiler decides
    #pragma omp simd
    for (int i = 4; i < 100; i++) {
        a[i] = a[i-4] + 1;
    }

    // With safelen=4 (dependency every 4 elements)
    #pragma omp simd safelen(4)
    for (int i = 4; i < 100; i++) {
        a[i] = a[i-4] + 2;
    }

    printf("  SIMD safelen clause allows specifying dependency distance\n");
    printf("  First few results: %d %d %d %d\n\n", a[4], a[5], a[6], a[7]);
}

int main() {
    printf("SIMD Vectorization Examples\n");
    printf("===========================\n\n");

    demo_basic_simd();
    demo_parallel_simd();
    demo_simd_reduction();
    demo_simd_safelen();

    printf("Note: SIMD effectiveness depends on:\n");
    printf("  - Compiler support and optimization flags\n");
    printf("  - CPU SIMD capabilities (SSE, AVX, etc.)\n");
    printf("  - Memory alignment\n");
    printf("  - Data dependencies\n");

    return 0;
}
