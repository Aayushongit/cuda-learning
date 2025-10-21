// cuRAND: High-performance random number generation
// Critical for dropout, weight initialization, and Monte Carlo simulations

#include <cuda_runtime.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CURAND(call) { \
    curandStatus_t status = call; \
    if (status != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "cuRAND Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void computeStatistics(float* data, int n, float* mean, float* stddev) {
    double sum = 0.0, sum_sq = 0.0;

    for (int i = 0; i < n; i++) {
        sum += data[i];
        sum_sq += data[i] * data[i];
    }

    *mean = sum / n;
    *stddev = sqrt(sum_sq / n - (*mean) * (*mean));
}

void computeHistogram(float* data, int n, int* hist, int bins, float min_val, float max_val) {
    for (int i = 0; i < bins; i++) hist[i] = 0;

    float range = max_val - min_val;
    for (int i = 0; i < n; i++) {
        if (data[i] >= min_val && data[i] < max_val) {
            int bin = (int)((data[i] - min_val) / range * bins);
            if (bin >= 0 && bin < bins) hist[bin]++;
        }
    }
}

void printHistogram(int* hist, int bins, int max_height = 20) {
    int max_count = 0;
    for (int i = 0; i < bins; i++) {
        if (hist[i] > max_count) max_count = hist[i];
    }

    printf("\nHistogram:\n");
    for (int i = 0; i < bins; i++) {
        int bar_length = (hist[i] * max_height) / max_count;
        printf("%2d: ", i);
        for (int j = 0; j < bar_length; j++) printf("█");
        printf(" %d\n", hist[i]);
    }
}

int main() {
    const int N = 10000000;  // 10M random numbers
    const unsigned long long seed = 1234ULL;

    size_t bytes = N * sizeof(float);

    // Host memory
    float *h_uniform = (float*)malloc(bytes);
    float *h_normal = (float*)malloc(bytes);
    float *h_lognormal = (float*)malloc(bytes);

    // Device memory
    float *d_random;
    CHECK_CUDA(cudaMalloc(&d_random, bytes));

    // Create cuRAND generator
    curandGenerator_t gen;
    CHECK_CURAND(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(gen, seed));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Generate uniform distribution [0, 1]
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CURAND(curandGenerateUniform(gen, d_random, N));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_uniform = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_uniform, start, stop));
    CHECK_CUDA(cudaMemcpy(h_uniform, d_random, bytes, cudaMemcpyDeviceToHost));

    // Generate normal distribution (mean=0, stddev=1)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CURAND(curandGenerateNormal(gen, d_random, N, 0.0f, 1.0f));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_normal = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_normal, start, stop));
    CHECK_CUDA(cudaMemcpy(h_normal, d_random, bytes, cudaMemcpyDeviceToHost));

    // Generate log-normal distribution
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CURAND(curandGenerateLogNormal(gen, d_random, N, 0.0f, 1.0f));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_lognormal = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_lognormal, start, stop));
    CHECK_CUDA(cudaMemcpy(h_lognormal, d_random, bytes, cudaMemcpyDeviceToHost));

    // Compute statistics
    float mean, stddev;

    printf("=== cuRAND Random Number Generation ===\n");
    printf("Generated %d random numbers\n\n", N);

    // Uniform distribution stats
    computeStatistics(h_uniform, N, &mean, &stddev);
    printf("Uniform Distribution [0, 1]:\n");
    printf("  Time: %.3f ms\n", ms_uniform);
    printf("  Throughput: %.2f billion numbers/sec\n", N / (ms_uniform * 1e6));
    printf("  Mean: %.6f (expected: 0.5)\n", mean);
    printf("  StdDev: %.6f (expected: 0.289)\n", stddev);

    int hist_uniform[10];
    computeHistogram(h_uniform, N, hist_uniform, 10, 0.0f, 1.0f);
    printHistogram(hist_uniform, 10);

    // Normal distribution stats
    computeStatistics(h_normal, N, &mean, &stddev);
    printf("\n\nNormal Distribution (mean=0, stddev=1):\n");
    printf("  Time: %.3f ms\n", ms_normal);
    printf("  Throughput: %.2f billion numbers/sec\n", N / (ms_normal * 1e6));
    printf("  Mean: %.6f (expected: 0.0)\n", mean);
    printf("  StdDev: %.6f (expected: 1.0)\n", stddev);

    int hist_normal[20];
    computeHistogram(h_normal, N, hist_normal, 20, -4.0f, 4.0f);
    printHistogram(hist_normal, 20);

    // Log-normal distribution stats
    computeStatistics(h_lognormal, N, &mean, &stddev);
    printf("\n\nLog-Normal Distribution:\n");
    printf("  Time: %.3f ms\n", ms_lognormal);
    printf("  Throughput: %.2f billion numbers/sec\n", N / (ms_lognormal * 1e6));
    printf("  Mean: %.6f\n", mean);
    printf("  StdDev: %.6f\n", stddev);

    // Sample values
    printf("\nSample uniform values (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_uniform[i]);
    }
    printf("\n");

    printf("\nSample normal values (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.6f ", h_normal[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CURAND(curandDestroyGenerator(gen));
    CHECK_CUDA(cudaFree(d_random));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_uniform);
    free(h_normal);
    free(h_lognormal);

    printf("\ncuRAND operations completed successfully!\n");
    return 0;
}
