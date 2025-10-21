// Advanced atomic operations: read, write, update, capture
#include <stdio.h>
#include <omp.h>

void demo_atomic_types() {
    int counter = 0;
    int snapshot;
    int old_value;

    printf("Different ATOMIC operation types:\n\n");

    #pragma omp parallel num_threads(4)
    {
        // ATOMIC UPDATE (default)
        #pragma omp atomic update
        counter++;

        // ATOMIC WRITE
        int tid = omp_get_thread_num();
        #pragma omp atomic write
        snapshot = tid;

        // ATOMIC READ
        int local_snap;
        #pragma omp atomic read
        local_snap = snapshot;

        // ATOMIC CAPTURE (get old value, then update)
        #pragma omp atomic capture
        {
            old_value = counter;
            counter++;
        }

        #pragma omp single
        {
            printf("ATOMIC UPDATE: counter incremented to %d\n", counter);
            printf("ATOMIC WRITE:  snapshot written by some thread\n");
            printf("ATOMIC READ:   value read safely\n");
            printf("ATOMIC CAPTURE: old_value=%d, new counter=%d\n\n",
                   old_value, counter);
        }
    }
}

void demo_atomic_operators() {
    int x = 10, y = 5;

    printf("ATOMIC with different operators:\n");

    // Addition
    #pragma omp parallel num_threads(4)
    {
        #pragma omp atomic
        x += 2;  // x = x + 2
    }
    printf("  After +=:  x = %d (expected: 18)\n", x);

    // Subtraction
    x = 20;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp atomic
        x -= 1;  // x = x - 1
    }
    printf("  After -=:  x = %d (expected: 16)\n", x);

    // Multiplication
    x = 2;
    #pragma omp parallel num_threads(4)
    {
        #pragma omp atomic
        x *= 2;  // x = x * 2
    }
    printf("  After *=:  x = %d (expected: 32)\n", x);

    // Bitwise operations
    x = 15;  // 0b1111
    #pragma omp parallel num_threads(2)
    {
        #pragma omp atomic
        x &= 7;  // x = x & 7 (0b0111)
    }
    printf("  After &=:  x = %d (bitwise AND)\n", x);

    x = 8;   // 0b1000
    #pragma omp parallel num_threads(2)
    {
        #pragma omp atomic
        x |= 4;  // x = x | 4 (0b0100)
    }
    printf("  After |=:  x = %d (bitwise OR)\n\n", x);
}

void demo_atomic_capture_variants() {
    int counter = 100;
    int captured;

    printf("ATOMIC CAPTURE variants:\n");

    // Variant 1: capture then update
    #pragma omp parallel num_threads(1)
    {
        #pragma omp atomic capture
        {
            captured = counter;
            counter++;
        }
    }
    printf("  Capture-then-update: captured=%d, counter=%d\n", captured, counter);

    // Variant 2: update then capture
    #pragma omp parallel num_threads(1)
    {
        #pragma omp atomic capture
        {
            counter++;
            captured = counter;
        }
    }
    printf("  Update-then-capture: captured=%d, counter=%d\n", captured, counter);

    // Variant 3: expression form (pre-increment)
    #pragma omp atomic capture
    captured = ++counter;
    printf("  Pre-increment:       captured=%d, counter=%d\n", captured, counter);

    // Variant 4: expression form (post-increment)
    #pragma omp atomic capture
    captured = counter++;
    printf("  Post-increment:      captured=%d, counter=%d\n\n", captured, counter);
}

void demo_atomic_vs_critical() {
    int atomic_counter = 0;
    int critical_counter = 0;
    const int ITERATIONS = 1000000;

    printf("Performance: ATOMIC vs CRITICAL\n");

    // Using atomic
    double start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < ITERATIONS; i++) {
        #pragma omp atomic
        atomic_counter++;
    }
    double time_atomic = omp_get_wtime() - start;

    // Using critical
    start = omp_get_wtime();
    #pragma omp parallel for
    for (int i = 0; i < ITERATIONS; i++) {
        #pragma omp critical
        {
            critical_counter++;
        }
    }
    double time_critical = omp_get_wtime() - start;

    printf("  ATOMIC:   %.4f seconds (result: %d)\n", time_atomic, atomic_counter);
    printf("  CRITICAL: %.4f seconds (result: %d)\n", time_critical, critical_counter);
    printf("  Speedup:  %.2fx faster with atomic\n\n", time_critical / time_atomic);
}

void demo_named_critical() {
    int resource_A = 0;
    int resource_B = 0;

    printf("Named CRITICAL sections (independent locks):\n");

    #pragma omp parallel num_threads(4)
    {
        // Two different critical sections can execute in parallel
        #pragma omp critical(lock_A)
        {
            resource_A++;
            printf("Thread %d in critical(lock_A)\n", omp_get_thread_num());
        }

        #pragma omp critical(lock_B)
        {
            resource_B++;
            printf("Thread %d in critical(lock_B)\n", omp_get_thread_num());
        }
    }

    printf("  resource_A: %d, resource_B: %d\n\n", resource_A, resource_B);
}

int main() {
    printf("Advanced Atomic Operations\n");
    printf("===========================\n\n");

    demo_atomic_types();
    demo_atomic_operators();
    demo_atomic_capture_variants();
    demo_atomic_vs_critical();
    demo_named_critical();

    printf("Key points:\n");
    printf("  - ATOMIC is much faster than CRITICAL for simple operations\n");
    printf("  - ATOMIC supports read/write/update/capture variants\n");
    printf("  - ATOMIC works with many operators: +,-,*,&,|,^,etc\n");
    printf("  - Use named CRITICAL sections for independent resources\n");

    return 0;
}
