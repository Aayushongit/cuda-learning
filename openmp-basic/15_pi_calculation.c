// Monte Carlo Pi estimation - reduction in action
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double estimate_pi_serial(long num_samples) {
    long count = 0;

    for (long i = 0; i < num_samples; i++) {
        double x = (double)rand() / RAND_MAX;
        double y = (double)rand() / RAND_MAX;
        if (x * x + y * y <= 1.0) {
            count++;
        }
    }

    return 4.0 * count / num_samples;
}

double estimate_pi_parallel(long num_samples) {
    long count = 0;

    #pragma omp parallel
    {
        unsigned int seed = omp_get_thread_num();
        long local_count = 0;

        #pragma omp for
        for (long i = 0; i < num_samples; i++) {
            double x = (double)rand_r(&seed) / RAND_MAX;
            double y = (double)rand_r(&seed) / RAND_MAX;
            if (x * x + y * y <= 1.0) {
                local_count++;
            }
        }

        #pragma omp atomic
        count += local_count;
    }

    return 4.0 * count / num_samples;
}

int main() {
    long num_samples = 100000000;

    printf("Estimating Pi with %ld samples\n", num_samples);
    printf("Threads: %d\n\n", omp_get_max_threads());

    double start = omp_get_wtime();
    double pi_serial = estimate_pi_serial(num_samples);
    double time_serial = omp_get_wtime() - start;

    start = omp_get_wtime();
    double pi_parallel = estimate_pi_parallel(num_samples);
    double time_parallel = omp_get_wtime() - start;

    printf("Serial:   Pi ≈ %.6f (%.3f seconds)\n", pi_serial, time_serial);
    printf("Parallel: Pi ≈ %.6f (%.3f seconds)\n", pi_parallel, time_parallel);
    printf("Speedup: %.2fx\n", time_serial / time_parallel);

    return 0;
}
