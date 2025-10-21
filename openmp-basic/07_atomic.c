// Atomic operations - lightweight alternative to critical for simple operations
#include <stdio.h>
#include <omp.h>

int main() {
    int counter = 0;
    double sum = 0.0;

    #pragma omp parallel num_threads(4)
    {
        for (int i = 0; i < 1000; i++) {
            #pragma omp atomic
            counter++;

            #pragma omp atomic
            sum += 1.5;
        }
    }

    printf("Atomic counter: %d (expected 4000)\n", counter);
    printf("Atomic sum: %.1f (expected 6000.0)\n", sum);

    return 0;
}
