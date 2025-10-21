// Nested loops - parallelize outer or inner loop
#include <stdio.h>
#include <omp.h>

int main() {
    const int N = 4;
    const int M = 3;
    int matrix[N][M];

    printf("Parallelizing outer loop:\n");
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        int tid = omp_get_thread_num();
        for (int j = 0; j < M; j++) {
            matrix[i][j] = i * M + j;
            printf("matrix[%d][%d] = %d (thread %d)\n", i, j, matrix[i][j], tid);
        }
    }

    printf("\n");

    // Collapse directive combines multiple loops
    printf("Collapsed loops (both levels parallelized):\n");
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            int tid = omp_get_thread_num();
            printf("Processing [%d][%d] on thread %d\n", i, j, tid);
        }
    }

    return 0;
}
