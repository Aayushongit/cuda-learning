// Cooperative Groups: Advanced thread synchronization and communication
// Enables flexible parallel patterns beyond traditional block/grid organization

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Thread block reduction using cooperative groups
__global__ void cooperativeBlockReduce(float* d_in, float* d_out, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float shared[256];

    // Load data
    float value = (tid < n) ? d_in[tid] : 0.0f;

    // Warp-level reduction using tiles
    for (int offset = tile32.size() / 2; offset > 0; offset /= 2) {
        value += tile32.shfl_down(value, offset);
    }

    // First thread in each warp writes to shared memory
    if (tile32.thread_rank() == 0) {
        shared[threadIdx.x / 32] = value;
    }

    block.sync();

    // Final reduction in first warp
    if (threadIdx.x < 32) {
        value = (threadIdx.x < blockDim.x / 32) ? shared[threadIdx.x] : 0.0f;

        for (int offset = 16; offset > 0; offset /= 2) {
            value += tile32.shfl_down(value, offset);
        }

        if (threadIdx.x == 0) {
            d_out[blockIdx.x] = value;
        }
    }
}

// Grid-wide synchronization using cooperative groups
__global__ void cooperativeGridSync(float* d_data, int n, int iterations) {
    cg::grid_group grid = cg::this_grid();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int iter = 0; iter < iterations; iter++) {
        // Each thread updates its element
        if (tid < n) {
            d_data[tid] = d_data[tid] * 0.99f + 1.0f;
        }

        // Synchronize entire grid
        grid.sync();

        // All threads can now safely read updated values
        if (tid < n && tid > 0) {
            float neighbor_avg = (d_data[tid - 1] + d_data[tid]) / 2.0f;
            d_data[tid] = neighbor_avg;
        }

        grid.sync();
    }
}

// Warp-level primitives demonstration
__global__ void warpLevelOperations(float* d_in, float* d_out, int n) {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float value = (tid < n) ? d_in[tid] : 0.0f;

    // Warp shuffle for prefix sum
    for (int offset = 1; offset < warp.size(); offset *= 2) {
        float temp = warp.shfl_up(value, offset);
        if (warp.thread_rank() >= offset) {
            value += temp;
        }
    }

    if (tid < n) {
        d_out[tid] = value;
    }
}

// Coalesced groups for divergent execution
__global__ void coalescedGroups(int* d_data, int* d_result, int n, int threshold) {
    cg::thread_block block = cg::this_thread_block();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int value = (tid < n) ? d_data[tid] : 0;

    // Create coalesced group of threads that meet condition
    cg::coalesced_group active = cg::coalesced_threads();

    if (value > threshold) {
        // Only threads meeting condition participate
        int group_sum = value;

        // Reduce within active threads
        for (int offset = active.size() / 2; offset > 0; offset /= 2) {
            group_sum += active.shfl_down(group_sum, offset);
        }

        if (active.thread_rank() == 0 && tid < n) {
            atomicAdd(d_result, group_sum);
        }
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    printf("=== Cooperative Groups Demo ===\n");
    printf("Array size: %d elements\n", N);
    printf("Block size: %d\n", blockSize);
    printf("Grid size:  %d\n\n", gridSize);

    // Allocate memory
    float *h_data = (float*)malloc(N * sizeof(float));
    float *h_result = (float*)malloc(gridSize * sizeof(float));

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(i % 100);
    }

    float *d_in, *d_out;
    CHECK_CUDA(cudaMalloc(&d_in, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_out, N * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_in, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. BLOCK-LEVEL REDUCTION
    float *d_block_results;
    CHECK_CUDA(cudaMalloc(&d_block_results, gridSize * sizeof(float)));

    CHECK_CUDA(cudaEventRecord(start));
    cooperativeBlockReduce<<<gridSize, blockSize>>>(d_in, d_block_results, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_block = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_block, start, stop));

    CHECK_CUDA(cudaMemcpy(h_result, d_block_results, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    float total_sum = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_sum += h_result[i];
    }

    printf("Block Reduction Result: %.2f\n", total_sum);
    printf("Time: %.3f ms\n\n", ms_block);

    // 2. WARP-LEVEL PREFIX SUM
    CHECK_CUDA(cudaEventRecord(start));
    warpLevelOperations<<<gridSize, blockSize>>>(d_in, d_out, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_warp = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_warp, start, stop));

    CHECK_CUDA(cudaMemcpy(h_result, d_out, 32 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Warp-level Prefix Sum (first warp):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.1f ", h_result[i]);
    }
    printf("\nTime: %.3f ms\n\n", ms_warp);

    // 3. COALESCED GROUPS
    int *h_int_data = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        h_int_data[i] = rand() % 100;
    }

    int *d_int_data, *d_int_result;
    CHECK_CUDA(cudaMalloc(&d_int_data, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_int_result, sizeof(int)));
    CHECK_CUDA(cudaMemset(d_int_result, 0, sizeof(int)));

    CHECK_CUDA(cudaMemcpy(d_int_data, h_int_data, N * sizeof(int), cudaMemcpyHostToDevice));

    int threshold = 50;

    CHECK_CUDA(cudaEventRecord(start));
    coalescedGroups<<<gridSize, blockSize>>>(d_int_data, d_int_result, N, threshold);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms_coalesced = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_coalesced, start, stop));

    int coalesced_result;
    CHECK_CUDA(cudaMemcpy(&coalesced_result, d_int_result, sizeof(int), cudaMemcpyDeviceToHost));

    printf("Coalesced Groups (sum of values > %d): %d\n", threshold, coalesced_result);
    printf("Time: %.3f ms\n\n", ms_coalesced);

    // 4. GRID-WIDE SYNC (requires cooperative launch)
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    if (prop.cooperativeLaunch) {
        printf("Grid-wide synchronization supported!\n");

        void* kernel_args[] = {&d_in, &N, (void*)3};  // 3 iterations

        CHECK_CUDA(cudaEventRecord(start));

        CHECK_CUDA(cudaLaunchCooperativeKernel(
            (void*)cooperativeGridSync,
            gridSize, blockSize,
            kernel_args, 0, 0));

        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaDeviceSynchronize());

        float ms_grid = 0;
        CHECK_CUDA(cudaEventElapsedTime(&ms_grid, start, stop));

        printf("Grid sync time: %.3f ms\n", ms_grid);
    } else {
        printf("Grid-wide synchronization not supported on this device.\n");
    }

    printf("\n=== Performance Summary ===\n");
    printf("Block reduction:    %.3f ms (%.2f B elem/sec)\n", ms_block, N / (ms_block * 1e6));
    printf("Warp operations:    %.3f ms (%.2f B elem/sec)\n", ms_warp, N / (ms_warp * 1e6));
    printf("Coalesced groups:   %.3f ms (%.2f B elem/sec)\n", ms_coalesced, N / (ms_coalesced * 1e6));

    // Cleanup
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_block_results));
    CHECK_CUDA(cudaFree(d_int_data));
    CHECK_CUDA(cudaFree(d_int_result));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_data);
    free(h_result);
    free(h_int_data);

    printf("\nCooperative Groups operations completed successfully!\n");
    return 0;
}
