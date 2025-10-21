// CUB Device-Wide Operations: High-performance parallel primitives
// Optimized single-kernel operations for entire device

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int N = 100000000;  // 100M elements

    printf("=== CUB Device-Wide Operations ===\n");
    printf("Data size: %d elements\n\n", N);

    // Allocate and initialize
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100);
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. DEVICE REDUCE - Sum
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_sum = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sum, start, stop));

    float sum_result;
    CHECK_CUDA(cudaMemcpy(&sum_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // 2. DEVICE REDUCE - Min/Max
    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceReduce::Min(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_min = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_min, start, stop));

    float min_result;
    CHECK_CUDA(cudaMemcpy(&min_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_out, N);
    float max_result;
    CHECK_CUDA(cudaMemcpy(&max_result, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // 3. DEVICE SCAN - Inclusive
    float *d_scan_out;
    CHECK_CUDA(cudaMalloc(&d_scan_out, N * sizeof(float)));

    CHECK_CUDA(cudaFree(d_temp_storage));
    temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_scan_out, N);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_scan_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_scan = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_scan, start, stop));

    // 4. DEVICE SELECT - Filter elements
    float threshold = 50.0f;
    int *d_num_selected;
    float *d_selected;
    CHECK_CUDA(cudaMalloc(&d_selected, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_num_selected, sizeof(int)));

    CHECK_CUDA(cudaFree(d_temp_storage));
    temp_storage_bytes = 0;

    auto select_op = [=] __device__ (float x) { return x > threshold; };
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_selected, d_num_selected, N, select_op);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_in, d_selected, d_num_selected, N, select_op);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_select = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_select, start, stop));

    int num_selected;
    CHECK_CUDA(cudaMemcpy(&num_selected, d_num_selected, sizeof(int), cudaMemcpyDeviceToHost));

    // 5. DEVICE RADIX SORT
    float *d_sorted_keys;
    CHECK_CUDA(cudaMalloc(&d_sorted_keys, N * sizeof(float)));

    CHECK_CUDA(cudaFree(d_temp_storage));
    temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_sorted_keys, N);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_in, d_sorted_keys, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_sort = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sort, start, stop));

    // 6. DEVICE HISTOGRAM
    const int NUM_BINS = 100;
    int *d_histogram;
    CHECK_CUDA(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));

    CHECK_CUDA(cudaFree(d_temp_storage));
    temp_storage_bytes = 0;

    int num_levels = NUM_BINS + 1;
    float lower_level = 0.0f;
    float upper_level = 100.0f;

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_in, d_histogram, num_levels,
                                        lower_level, upper_level, N);
    CHECK_CUDA(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    CHECK_CUDA(cudaEventRecord(start));
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_in, d_histogram, num_levels,
                                        lower_level, upper_level, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_histogram = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_histogram, start, stop));

    // Results
    printf("=== Reduction Results ===\n");
    printf("Sum:     %.2f\n", sum_result);
    printf("Min:     %.2f\n", min_result);
    printf("Max:     %.2f\n", max_result);
    printf("Mean:    %.2f\n", sum_result / N);

    printf("\n=== Selection Results ===\n");
    printf("Elements > %.0f: %d (%.2f%%)\n", threshold, num_selected, (float)num_selected / N * 100);

    // Sample results
    float *h_scan = (float*)malloc(20 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_scan, d_scan_out, 20 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nInclusive Scan (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_scan[i]);
    }
    printf("\n");

    float *h_sorted = (float*)malloc(20 * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_sorted, d_sorted_keys, 20 * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nSorted (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_sorted[i]);
    }
    printf("\n");

    printf("\n=== Performance ===\n");
    printf("Reduce (Sum):    %.3f ms (%.2f B elem/sec)\n", ms_sum, N / (ms_sum * 1e6));
    printf("Reduce (Min):    %.3f ms (%.2f B elem/sec)\n", ms_min, N / (ms_min * 1e6));
    printf("Scan:            %.3f ms (%.2f B elem/sec)\n", ms_scan, N / (ms_scan * 1e6));
    printf("Select:          %.3f ms (%.2f B elem/sec)\n", ms_select, N / (ms_select * 1e6));
    printf("Radix Sort:      %.3f ms (%.2f M elem/sec)\n", ms_sort, N / (ms_sort * 1000.0));
    printf("Histogram:       %.3f ms (%.2f B elem/sec)\n", ms_histogram, N / (ms_histogram * 1e6));

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_scan_out));
    CHECK_CUDA(cudaFree(d_selected));
    CHECK_CUDA(cudaFree(d_num_selected));
    CHECK_CUDA(cudaFree(d_sorted_keys));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaFree(d_temp_storage));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_data);
    free(h_scan);
    free(h_sorted);

    printf("\nCUB device-wide operations completed successfully!\n");
    return 0;
}
