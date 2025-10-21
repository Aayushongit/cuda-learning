// CUB Block-Level Primitives: Optimized cooperative thread operations
// Provides building blocks for custom high-performance kernels

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

// Block-level reduction kernel
template <int BLOCK_SIZE>
__global__ void blockReduceKernel(float* d_in, float* d_out, int n) {
    typedef cub::BlockReduce<float, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (tid < n) ? d_in[tid] : 0.0f;

    // Block-wide reduction
    float aggregate = BlockReduce(temp_storage).Sum(value);

    // First thread writes block result
    if (threadIdx.x == 0) {
        d_out[blockIdx.x] = aggregate;
    }
}

// Block-level scan (prefix sum) kernel
template <int BLOCK_SIZE>
__global__ void blockScanKernel(float* d_in, float* d_out, int n) {
    typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (tid < n) ? d_in[tid] : 0.0f;

    // Block-wide inclusive scan
    BlockScan(temp_storage).InclusiveSum(value, value);

    if (tid < n) {
        d_out[tid] = value;
    }
}

// Block-level sort kernel
template <int BLOCK_SIZE, int ITEMS_PER_THREAD>
__global__ void blockSortKernel(float* d_in, float* d_out, int n) {
    typedef cub::BlockLoad<float, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<float, BLOCK_SIZE, ITEMS_PER_THREAD, cub::BLOCK_STORE_TRANSPOSE> BlockStore;
    typedef cub::BlockRadixSort<float, BLOCK_SIZE, ITEMS_PER_THREAD> BlockRadixSort;

    __shared__ union {
        typename BlockLoad::TempStorage load;
        typename BlockStore::TempStorage store;
        typename BlockRadixSort::TempStorage sort;
    } temp_storage;

    int block_offset = blockIdx.x * (BLOCK_SIZE * ITEMS_PER_THREAD);
    float items[ITEMS_PER_THREAD];

    // Load items
    BlockLoad(temp_storage.load).Load(d_in + block_offset, items, n - block_offset, 0.0f);
    __syncthreads();

    // Sort items
    BlockRadixSort(temp_storage.sort).Sort(items);
    __syncthreads();

    // Store sorted items
    BlockStore(temp_storage.store).Store(d_out + block_offset, items, n - block_offset);
}

// Block-level histogram kernel
template <int BLOCK_SIZE, int NUM_BINS>
__global__ void blockHistogramKernel(float* d_in, int* d_out, int n, float min_val, float max_val) {
    typedef cub::BlockHistogram<float, BLOCK_SIZE, NUM_BINS> BlockHistogram;
    __shared__ typename BlockHistogram::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (tid < n) ? d_in[tid] : min_val;

    // Scale to bin range
    float range = max_val - min_val;
    int bin = (int)((value - min_val) / range * NUM_BINS);
    if (bin >= NUM_BINS) bin = NUM_BINS - 1;
    if (bin < 0) bin = 0;

    int histogram[NUM_BINS];
    BlockHistogram(temp_storage).Histogram(value, histogram);

    // Aggregate block histograms
    if (threadIdx.x < NUM_BINS) {
        atomicAdd(&d_out[threadIdx.x], histogram[threadIdx.x]);
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int BLOCK_SIZE = 256;
    const int ITEMS_PER_THREAD = 4;

    printf("=== CUB Block-Level Primitives ===\n");
    printf("Data size: %d elements\n", N);
    printf("Block size: %d threads\n\n", BLOCK_SIZE);

    // Allocate and initialize data
    float *h_data = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100);
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. BLOCK REDUCE
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float *d_block_results;
    CHECK_CUDA(cudaMalloc(&d_block_results, numBlocks * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));
    blockReduceKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(d_in, d_block_results, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_reduce = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_reduce, start, stop));

    float *h_block_results = (float*)malloc(numBlocks * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_block_results, d_block_results, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float total_sum = 0.0f;
    for (int i = 0; i < numBlocks; i++) {
        total_sum += h_block_results[i];
    }

    // 2. BLOCK SCAN
    CHECK_CUDA(cudaEventRecord(start));
    blockScanKernel<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_scan = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_scan, start, stop));

    // 3. BLOCK SORT
    int sortBlocks = (N + BLOCK_SIZE * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);

    CHECK_CUDA(cudaEventRecord(start));
    blockSortKernel<BLOCK_SIZE, ITEMS_PER_THREAD><<<sortBlocks, BLOCK_SIZE>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_sort = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sort, start, stop));

    // 4. BLOCK HISTOGRAM
    const int NUM_BINS = 32;
    int *d_histogram;
    CHECK_CUDA(cudaMalloc(&d_histogram, NUM_BINS * sizeof(int)));
    CHECK_CUDA(cudaMemset(d_histogram, 0, NUM_BINS * sizeof(int)));

    CHECK_CUDA(cudaEventRecord(start));
    blockHistogramKernel<BLOCK_SIZE, NUM_BINS><<<numBlocks, BLOCK_SIZE>>>(d_in, d_histogram, N, 0.0f, 100.0f);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_histogram = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_histogram, start, stop));

    // Results
    printf("=== Results ===\n");
    printf("Block Reduction Sum: %.2f\n", total_sum);

    float *h_scanned = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_scanned, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nBlock Scan (first 10 prefix sums):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_scanned[i]);
    }
    printf("\n");

    float *h_sorted = (float*)malloc(N * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_sorted, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));
    printf("\nBlock Sort (first 10 sorted):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_sorted[i]);
    }
    printf("\n");

    int *h_histogram = (int*)malloc(NUM_BINS * sizeof(int));
    CHECK_CUDA(cudaMemcpy(h_histogram, d_histogram, NUM_BINS * sizeof(int), cudaMemcpyDeviceToHost));
    printf("\nHistogram (%d bins):\n", NUM_BINS);
    for (int i = 0; i < NUM_BINS; i++) {
        printf("Bin %2d: %d\n", i, h_histogram[i]);
    }

    printf("\n=== Performance ===\n");
    printf("Block Reduce:    %.3f ms (%.2f B elem/sec)\n", ms_reduce, N / (ms_reduce * 1e6));
    printf("Block Scan:      %.3f ms (%.2f B elem/sec)\n", ms_scan, N / (ms_scan * 1e6));
    printf("Block Sort:      %.3f ms (%.2f M elem/sec)\n", ms_sort, N / (ms_sort * 1000.0));
    printf("Block Histogram: %.3f ms (%.2f B elem/sec)\n", ms_histogram, N / (ms_histogram * 1e6));

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_block_results));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_data);
    free(h_block_results);
    free(h_scanned);
    free(h_sorted);
    free(h_histogram);

    printf("\nCUB block-level operations completed successfully!\n");
    return 0;
}
