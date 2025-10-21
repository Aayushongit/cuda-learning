// Comprehensive example with combined clauses
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void demo_kitchen_sink() {
    printf("Example 1: Everything combined\n");

    const int N = 10000000;
    double *data = malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        data[i] = i * 0.5;
    }

    double sum = 0.0;
    double max_val = 0.0;
    int max_idx = 0;
    int local_temp;

    double start = omp_get_wtime();

    #pragma omp parallel for \
        num_threads(8) \
        schedule(guided, 1000) \
        private(local_temp) \
        shared(data) \
        reduction(+:sum) \
        reduction(max:max_val) \
        if(N > 1000)
    for (int i = 0; i < N; i++) {
        local_temp = i % 100;
        sum += data[i];
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    double elapsed = omp_get_wtime() - start;

    printf("  Processed %d elements in %.4f seconds\n", N, elapsed);
    printf("  Sum: %.2f, Max: %.2f\n\n", sum, max_val);

    free(data);
}

void demo_nested_parallelism() {
    printf("Example 2: Nested parallelism with collapse\n");

    const int M = 1000, N = 1000;
    double **matrix = malloc(M * sizeof(double*));
    for (int i = 0; i < M; i++) {
        matrix[i] = malloc(N * sizeof(double));
    }

    double start = omp_get_wtime();

    #pragma omp parallel for \
        collapse(2) \
        schedule(dynamic, 50) \
        num_threads(4)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matrix[i][j] = (i + j) * 0.5;
        }
    }

    double elapsed = omp_get_wtime() - start;
    printf("  Initialized %dx%d matrix in %.4f seconds\n\n", M, N, elapsed);

    for (int i = 0; i < M; i++) free(matrix[i]);
    free(matrix);
}

void demo_reduction_combinations() {
    printf("Example 3: Multiple reduction operations\n");

    const int N = 1000000;
    int *data = malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        data[i] = i % 100;
    }

    int sum = 0;
    int min_val = 999999;
    int max_val = -999999;
    int count_zeros = 0;

    #pragma omp parallel for \
        reduction(+:sum) \
        reduction(min:min_val) \
        reduction(max:max_val) \
        reduction(+:count_zeros) \
        schedule(static, 10000)
    for (int i = 0; i < N; i++) {
        sum += data[i];
        if (data[i] < min_val) min_val = data[i];
        if (data[i] > max_val) max_val = data[i];
        if (data[i] == 0) count_zeros++;
    }

    printf("  Sum: %d\n", sum);
    printf("  Min: %d, Max: %d\n", min_val, max_val);
    printf("  Zeros count: %d\n\n", count_zeros);

    free(data);
}

void demo_complex_data_sharing() {
    printf("Example 4: Complex data sharing\n");

    int shared_counter = 0;
    int init_value = 42;

    #pragma omp parallel \
        num_threads(4) \
        shared(shared_counter) \
        firstprivate(init_value)
    {
        int tid = omp_get_thread_num();
        int private_sum = 0;

        // Each thread has init_value = 42
        private_sum = init_value + tid;

        #pragma omp for schedule(static) nowait
        for (int i = 0; i < 8; i++) {
            private_sum += i;
        }

        #pragma omp critical
        {
            shared_counter += private_sum;
            printf("  Thread %d: private_sum=%d, shared_counter=%d\n",
                   tid, private_sum, shared_counter);
        }

        #pragma omp barrier

        #pragma omp single
        {
            printf("  Final shared_counter: %d\n\n", shared_counter);
        }
    }
}

void demo_mixed_worksharing() {
    printf("Example 5: Mixed work-sharing constructs\n");

    #pragma omp parallel num_threads(4)
    {
        // Sections for different tasks
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                printf("  Section 1 on thread %d\n", omp_get_thread_num());
            }

            #pragma omp section
            {
                printf("  Section 2 on thread %d\n", omp_get_thread_num());
            }
        }

        // Parallel for after sections
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < 8; i++) {
            printf("  Loop iteration %d on thread %d\n",
                   i, omp_get_thread_num());
        }

        // Single thread work
        #pragma omp single
        {
            printf("  Single-thread work on thread %d\n",
                   omp_get_thread_num());
        }

        // Master-only work
        #pragma omp master
        {
            printf("  Master-only work on thread %d\n",
                   omp_get_thread_num());
        }
    }
    printf("\n");
}

void demo_conditional_parallel() {
    printf("Example 6: Conditional parallelization\n");

    for (int size = 10; size <= 10000000; size *= 100) {
        double *arr = malloc(size * sizeof(double));
        for (int i = 0; i < size; i++) arr[i] = i;

        double sum = 0.0;
        int actual_threads = 0;

        double start = omp_get_wtime();

        // Only parallelize if array is large enough
        #pragma omp parallel for \
            if(size > 1000) \
            reduction(+:sum) \
            shared(actual_threads)
        for (int i = 0; i < size; i++) {
            sum += arr[i];
            if (i == 0) {
                actual_threads = omp_get_num_threads();
            }
        }

        double elapsed = omp_get_wtime() - start;

        printf("  Size %8d: threads=%d, time=%.6f, sum=%.0f\n",
               size, actual_threads, elapsed, sum);

        free(arr);
    }
    printf("\n");
}

int main() {
    printf("Combined Clauses - Comprehensive Examples\n");
    printf("==========================================\n\n");

    printf("Available threads: %d\n\n", omp_get_max_threads());

    demo_kitchen_sink();
    demo_nested_parallelism();
    demo_reduction_combinations();
    demo_complex_data_sharing();
    demo_mixed_worksharing();
    demo_conditional_parallel();

    printf("Key combinations shown:\n");
    printf("  - Multiple reductions together\n");
    printf("  - collapse(N) with schedule clauses\n");
    printf("  - if clause for conditional parallelization\n");
    printf("  - private/shared/firstprivate together\n");
    printf("  - nowait to skip barriers\n");
    printf("  - Mixed sections/for/single/master\n");

    return 0;
}
