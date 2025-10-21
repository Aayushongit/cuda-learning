// Barrier synchronization - wait for all threads
#include <stdio.h>
#include <omp.h>
#include <unistd.h>

int main() {
    omp_set_num_threads(4);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        printf("Thread %d: Phase 1 starting\n", tid);
        sleep(tid);  // Different threads take different times
        printf("Thread %d: Phase 1 completed\n", tid);

        // All threads wait here until everyone reaches this point
        #pragma omp barrier

        printf("Thread %d: Phase 2 starting (after barrier)\n", tid);
    }

    printf("\nAll threads synchronized!\n");

    return 0;
}
