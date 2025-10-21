// Thrust: High-level parallel algorithms for sorting and reductions
// Simplifies data processing in ML pipelines and preprocessing

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/extrema.h>
#include <thrust/unique.h>
#include <thrust/inner_product.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int N = 10000000;  // 10M elements

    printf("=== Thrust Parallel Algorithms ===\n");
    printf("Data size: %d elements\n\n", N);

    // Initialize host data
    thrust::host_vector<float> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 1000);
    }

    // Transfer to device
    thrust::device_vector<float> d_data = h_data;
    thrust::device_vector<float> d_sorted(N);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. SORT
    d_sorted = d_data;
    CHECK_CUDA(cudaEventRecord(start));
    thrust::sort(d_sorted.begin(), d_sorted.end());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_sort = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sort, start, stop));

    // 2. REDUCTION (sum)
    CHECK_CUDA(cudaEventRecord(start));
    float sum = thrust::reduce(d_data.begin(), d_data.end(), 0.0f, thrust::plus<float>());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_reduce = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_reduce, start, stop));

    // 3. MIN/MAX
    CHECK_CUDA(cudaEventRecord(start));
    auto minmax = thrust::minmax_element(d_data.begin(), d_data.end());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_minmax = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_minmax, start, stop));

    float min_val = *minmax.first;
    float max_val = *minmax.second;

    // 4. MEAN (using reduce)
    float mean = sum / N;

    // 5. DOT PRODUCT
    thrust::device_vector<float> d_data2 = d_data;
    CHECK_CUDA(cudaEventRecord(start));
    float dot_product = thrust::inner_product(d_data.begin(), d_data.end(),
                                              d_data2.begin(), 0.0f);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_dot = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_dot, start, stop));

    // 6. COUNT (elements greater than threshold)
    float threshold = 500.0f;
    CHECK_CUDA(cudaEventRecord(start));
    int count = thrust::count_if(d_data.begin(), d_data.end(),
                                 [=] __device__ (float x) { return x > threshold; });
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_count = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_count, start, stop));

    // 7. UNIQUE (count unique elements)
    d_sorted = d_data;
    thrust::sort(d_sorted.begin(), d_sorted.end());
    CHECK_CUDA(cudaEventRecord(start));
    auto new_end = thrust::unique(d_sorted.begin(), d_sorted.end());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_unique = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_unique, start, stop));

    int unique_count = new_end - d_sorted.begin();

    // Display results
    printf("=== Statistics ===\n");
    printf("Sum:              %.2f\n", sum);
    printf("Mean:             %.2f\n", mean);
    printf("Min:              %.2f\n", min_val);
    printf("Max:              %.2f\n", max_val);
    printf("Count > %.0f:      %d (%.1f%%)\n", threshold, count, (float)count / N * 100);
    printf("Unique elements:  %d (%.1f%%)\n", unique_count, (float)unique_count / N * 100);
    printf("Dot product:      %.2e\n", dot_product);

    printf("\n=== Performance (Throughput) ===\n");
    printf("Sort:             %.2f M elements/sec (%.3f ms)\n",
           N / (ms_sort * 1000.0), ms_sort);
    printf("Reduce (sum):     %.2f B elements/sec (%.3f ms)\n",
           N / (ms_reduce * 1e6), ms_reduce);
    printf("Min/Max:          %.2f B elements/sec (%.3f ms)\n",
           N / (ms_minmax * 1e6), ms_minmax);
    printf("Dot product:      %.2f B elements/sec (%.3f ms)\n",
           N / (ms_dot * 1e6), ms_dot);
    printf("Count if:         %.2f B elements/sec (%.3f ms)\n",
           N / (ms_count * 1e6), ms_count);
    printf("Unique:           %.2f M elements/sec (%.3f ms)\n",
           N / (ms_unique * 1000.0), ms_unique);

    // Show sample sorted data
    thrust::host_vector<float> h_sorted = d_sorted;
    printf("\nSorted data (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_sorted[i]);
    }
    printf("\n\nSorted data (last 10):\n");
    for (int i = N - 10; i < N; i++) {
        printf("%.1f ", h_sorted[i]);
    }
    printf("\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nThrust operations completed successfully!\n");
    return 0;
}
