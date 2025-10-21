// Firstprivate and lastprivate clauses
#include <stdio.h>
#include <omp.h>

int main() {
    int x = 100;
    int y = 200;

    printf("Initial: x = %d, y = %d\n\n", x, y);

    // Firstprivate: each thread gets initialized copy
    printf("FIRSTPRIVATE example:\n");
    #pragma omp parallel for firstprivate(x) num_threads(4)
    for (int i = 0; i < 4; i++) {
        printf("Thread %d: x starts at %d, ", omp_get_thread_num(), x);
        x += i;
        printf("becomes %d\n", x);
    }
    printf("After parallel region: x = %d (unchanged)\n\n", x);

    // Lastprivate: last iteration value copied back
    printf("LASTPRIVATE example:\n");
    #pragma omp parallel for lastprivate(y) num_threads(4)
    for (int i = 0; i < 8; i++) {
        y = i * 10;
        printf("Iteration %d: y = %d (thread %d)\n", i, y, omp_get_thread_num());
    }
    printf("After parallel region: y = %d (from last iteration)\n", y);

    return 0;
}
