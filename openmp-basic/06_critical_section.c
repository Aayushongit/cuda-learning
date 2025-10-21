// Critical sections - mutual exclusion for shared resources
#include <stdio.h>
#include <omp.h>

int main() {
    int counter = 0;
    int safe_counter = 0;

    #pragma omp parallel num_threads(4)
    {
        // Unsafe increment - race condition
        for (int i = 0; i < 1000; i++) {
            counter++;
        }

        // Safe increment - critical section ensures only one thread at a time
        for (int i = 0; i < 1000; i++) {
            #pragma omp critical
            {
                safe_counter++;
            }
        }
    }

    printf("Unsafe counter (race condition): %d (expected 4000)\n", counter);
    printf("Safe counter (critical section): %d (expected 4000)\n", safe_counter);

    return 0;
}
