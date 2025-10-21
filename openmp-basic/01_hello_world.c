// Basic parallel region - multiple threads execute the same block
#include <stdio.h>
#include <omp.h>

int main() {
    printf("Starting parallel region...\n");

    #pragma omp parallel
    {
        printf("Hello from thread!\n");
    }

    printf("\nWith 4 threads:\n");
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        printf("Hello from thread %d\n", omp_get_thread_num());
    }

    return 0;
}
