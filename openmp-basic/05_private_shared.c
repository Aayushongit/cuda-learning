// Data sharing: private vs shared variables
#include <stdio.h>
#include <omp.h>

int main() {
    int shared_var = 0;
    int private_var = 100;

    printf("Before parallel region:\n");
    printf("shared_var = %d, private_var = %d\n\n", shared_var, private_var);

    #pragma omp parallel private(private_var) shared(shared_var) num_threads(4)
    {
        int tid = omp_get_thread_num();

        // Each thread has its own copy of private_var (uninitialized!)
        printf("Thread %d: private_var (uninitialized) = %d\n", tid, private_var);

        private_var = tid * 10;
        shared_var += tid;  // All threads access the same shared_var (race condition!)

        printf("Thread %d: private_var = %d, shared_var = %d\n",
               tid, private_var, shared_var);
    }

    printf("\nAfter parallel region:\n");
    printf("shared_var = %d (unpredictable due to race!)\n", shared_var);
    printf("private_var = %d (unchanged)\n", private_var);

    return 0;
}
