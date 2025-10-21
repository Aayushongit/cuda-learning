// Advanced loop directives: ordered, if, nowait
#include <stdio.h>
#include <omp.h>

void demo_ordered() {
    printf("ORDERED directive (maintains sequential order for part of loop):\n");

    #pragma omp parallel for ordered
    for (int i = 0; i < 10; i++) {
        int tid = omp_get_thread_num();
        printf("Thread %d processing iteration %d\n", tid, i);

        #pragma omp ordered
        {
            // This part executes in sequential order despite parallel execution
            printf("  -> Ordered output: iteration %d\n", i);
        }
    }
    printf("\n");
}

void demo_if_clause() {
    printf("IF clause (conditional parallelization):\n");

    int small_n = 10;
    int large_n = 1000000;

    // Small loop - runs serially due to if(false)
    printf("Small loop (n=%d):\n", small_n);
    #pragma omp parallel for if(small_n > 100)
    for (int i = 0; i < small_n; i++) {
        if (i == 0) {
            printf("  Threads used: %d\n", omp_get_num_threads());
        }
    }

    // Large loop - runs in parallel due to if(true)
    printf("Large loop (n=%d):\n", large_n);
    #pragma omp parallel for if(large_n > 100)
    for (int i = 0; i < large_n; i++) {
        if (i == 0) {
            printf("  Threads used: %d\n", omp_get_num_threads());
        }
    }
    printf("\n");
}

void demo_nowait() {
    printf("NOWAIT clause (skip implicit barrier):\n");

    #pragma omp parallel num_threads(4)
    {
        // First loop with nowait - threads don't wait for others
        #pragma omp for nowait
        for (int i = 0; i < 4; i++) {
            printf("Thread %d: first loop iteration %d\n",
                   omp_get_thread_num(), i);
        }
        // Threads can immediately proceed here without waiting

        // Each thread does independent work
        printf("Thread %d: doing independent work (no wait!)\n",
               omp_get_thread_num());

        // Second loop - has implicit barrier
        #pragma omp for
        for (int i = 0; i < 4; i++) {
            printf("Thread %d: second loop iteration %d\n",
                   omp_get_thread_num(), i);
        }
        // All threads wait here before proceeding
    }
    printf("\n");
}

void demo_schedule_comparison() {
    printf("SCHEDULE clause comparison:\n");
    const int N = 16;

    printf("STATIC (default):\n");
    #pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < N; i++) {
        printf("  Iter %2d -> Thread %d\n", i, omp_get_thread_num());
    }

    printf("\nSTATIC with chunk=4:\n");
    #pragma omp parallel for schedule(static, 4) num_threads(4)
    for (int i = 0; i < N; i++) {
        printf("  Iter %2d -> Thread %d\n", i, omp_get_thread_num());
    }

    printf("\nDYNAMIC with chunk=2:\n");
    #pragma omp parallel for schedule(dynamic, 2) num_threads(4)
    for (int i = 0; i < N; i++) {
        printf("  Iter %2d -> Thread %d\n", i, omp_get_thread_num());
    }

    printf("\nGUIDED (adaptive chunk sizes):\n");
    #pragma omp parallel for schedule(guided) num_threads(4)
    for (int i = 0; i < N; i++) {
        printf("  Iter %2d -> Thread %d\n", i, omp_get_thread_num());
    }
    printf("\n");
}

int main() {
    printf("Advanced Loop Directives Demo\n");
    printf("==============================\n\n");

    demo_ordered();
    demo_if_clause();
    demo_nowait();
    demo_schedule_comparison();

    return 0;
}
