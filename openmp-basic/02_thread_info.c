// Get thread ID, total threads, and other runtime info
#include <stdio.h>
#include <omp.h>

int main() {
    int max_threads = omp_get_max_threads();
    int num_procs = omp_get_num_procs();

    printf("Available processors: %d\n", num_procs);
    printf("Max threads: %d\n\n", max_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        printf("Thread %d of %d threads\n", tid, nthreads);
    }

    return 0;
}
