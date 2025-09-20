#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

using namespace cooperative_groups;

#define N 1024
#define BLOCK_SIZE 256

// Basic cooperative groups usage
__global__ void coop_groups_basic() {
    // Get the thread block group
    thread_block block = this_thread_block();
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    __shared__ int shared_data[BLOCK_SIZE];
    shared_data[tid] = tid + bid * BLOCK_SIZE;
    
    // Synchronize the entire thread block
    block.sync();
    
    if (tid == 0) {
        printf("Block %d: All threads synchronized\n", bid);
    }
    
    // Get warp-level group
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    // Warp-level operations
    int warp_sum = warp.shfl_down(shared_data[tid], 16);
    warp_sum += warp.shfl_down(warp_sum, 8);
    warp_sum += warp.shfl_down(warp_sum, 4);
    warp_sum += warp.shfl_down(warp_sum, 2);
    warp_sum += warp.shfl_down(warp_sum, 1);
    
    if (warp.thread_rank() == 0) {
        printf("Block %d, Warp %d: Sum = %d\n", bid, tid / 32, warp_sum);
    }
}

// Warp-level reduction using cooperative groups
__global__ void warp_reduction_coop(float *input, float *output, int n) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    float value = (gid < n) ? input[gid] : 0.0f;
    
    // Warp-level reduction using shuffle
    for (int offset = warp.size() / 2; offset > 0; offset /= 2) {
        value += warp.shfl_down(value, offset);
    }
    
    // Store warp results in shared memory
    __shared__ float warp_results[32];
    if (warp.thread_rank() == 0) {
        warp_results[tid / 32] = value;
    }
    
    block.sync();
    
    // Final reduction of warp results
    if (tid < 32) {
        thread_block_tile<32> final_warp = tiled_partition<32>(block);
        float final_value = (tid < blockDim.x / 32) ? warp_results[tid] : 0.0f;
        
        for (int offset = final_warp.size() / 2; offset > 0; offset /= 2) {
            final_value += final_warp.shfl_down(final_value, offset);
        }
        
        if (final_warp.thread_rank() == 0) {
            output[blockIdx.x] = final_value;
        }
    }
}

// Multi-warp cooperative sorting
__global__ void bitonic_sort_coop(int *data, int n) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid >= n) return;
    
    extern __shared__ int shared_data[];
    shared_data[tid] = data[gid];
    
    block.sync();
    
    // Bitonic sort phases
    for (int phase = 2; phase <= blockDim.x; phase *= 2) {
        for (int step = phase / 2; step > 0; step /= 2) {
            int partner = tid ^ step;
            bool up = ((tid & phase) == 0);
            
            if (partner < blockDim.x) {
                int a = shared_data[tid];
                int b = shared_data[partner];
                
                if ((a > b) == up) {
                    shared_data[tid] = b;
                    shared_data[partner] = a;
                }
            }
            
            block.sync();
        }
    }
    
    data[gid] = shared_data[tid];
}

// Subgroup operations (variable warp size)
__global__ void subgroup_operations() {
    thread_block block = this_thread_block();
    
    int tid = threadIdx.x;
    
    // Create different sized subgroups
    thread_block_tile<16> tile16 = tiled_partition<16>(block);
    thread_block_tile<8> tile8 = tiled_partition<8>(block);
    thread_block_tile<4> tile4 = tiled_partition<4>(block);
    
    int value = tid + 1;
    
    // Operations within 16-thread tile
    if (tid < 64) { // First 4 tiles of 16
        int sum16 = tile16.shfl_down(value, 8);
        sum16 += tile16.shfl_down(sum16, 4);
        sum16 += tile16.shfl_down(sum16, 2);
        sum16 += tile16.shfl_down(sum16, 1);
        
        if (tile16.thread_rank() == 0) {
            printf("Tile16 %d: Sum = %d\n", tid / 16, sum16);
        }
    }
    
    // Operations within 8-thread tile
    if (tid < 32) { // First 4 tiles of 8
        int sum8 = tile8.shfl_down(value, 4);
        sum8 += tile8.shfl_down(sum8, 2);
        sum8 += tile8.shfl_down(sum8, 1);
        
        if (tile8.thread_rank() == 0) {
            printf("Tile8 %d: Sum = %d\n", tid / 8, sum8);
        }
    }
}

// Cooperative matrix multiplication
__global__ void matrix_mult_coop(float *A, float *B, float *C, int n) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    __shared__ float shared_A[16][16];
    __shared__ float shared_B[16][16];
    
    int tx = threadIdx.x % 16;
    int ty = threadIdx.x / 16;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    for (int tile = 0; tile < (n + 15) / 16; tile++) {
        // Cooperative loading into shared memory
        if (row < n && (tile * 16 + tx) < n) {
            shared_A[ty][tx] = A[row * n + tile * 16 + tx];
        } else {
            shared_A[ty][tx] = 0.0f;
        }
        
        if ((tile * 16 + ty) < n && col < n) {
            shared_B[ty][tx] = B[(tile * 16 + ty) * n + col];
        } else {
            shared_B[ty][tx] = 0.0f;
        }
        
        block.sync();
        
        // Compute partial result
        for (int k = 0; k < 16; k++) {
            sum += shared_A[ty][k] * shared_B[k][tx];
        }
        
        block.sync();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Histogram using cooperative groups
__global__ void histogram_coop(int *data, int *hist, int n, int num_bins) {
    thread_block block = this_thread_block();
    thread_block_tile<32> warp = tiled_partition<32>(block);
    
    extern __shared__ int shared_hist[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        shared_hist[i] = 0;
    }
    
    block.sync();
    
    // Process data
    for (int i = gid; i < n; i += blockDim.x * gridDim.x) {
        int bin = data[i] % num_bins;
        atomicAdd(&shared_hist[bin], 1);
    }
    
    block.sync();
    
    // Merge to global histogram
    for (int i = tid; i < num_bins; i += blockDim.x) {
        atomicAdd(&hist[i], shared_hist[i]);
    }
}

int main() {
    printf("=== CUDA Cooperative Groups Demo ===\n");
    printf("This example demonstrates advanced cooperative groups features.\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 6) {
        printf("Cooperative Groups requires compute capability 6.0 or higher\n");
        printf("Some features may not be available on this device\n");
    } else {
        printf("Cooperative Groups: Fully supported\n");
    }
    printf("\n");
    
    // 1. Basic cooperative groups
    printf("=== Basic Cooperative Groups ===\n");
    coop_groups_basic<<<2, 64>>>();
    cudaDeviceSynchronize();
    
    // 2. Warp reduction with cooperative groups
    printf("\n=== Warp Reduction with Cooperative Groups ===\n");
    
    size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *d_input, *d_output;
    
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, 32 * sizeof(float)); // Max 32 blocks
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    int grid_size = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 32);
    warp_reduction_coop<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, N);
    
    float *h_output = (float*)malloc(grid_size * sizeof(float));
    cudaMemcpy(h_output, d_output, grid_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total = 0.0f;
    for (int i = 0; i < grid_size; i++) {
        total += h_output[i];
    }
    
    printf("Reduction result: %.0f (expected: %d)\n", total, N);
    
    // 3. Bitonic sort with cooperative groups
    printf("\n=== Bitonic Sort with Cooperative Groups ===\n");
    
    int sort_size = 512;
    int *h_sort_data = (int*)malloc(sort_size * sizeof(int));
    int *d_sort_data;
    
    for (int i = 0; i < sort_size; i++) {
        h_sort_data[i] = rand() % 100;
    }
    
    cudaMalloc(&d_sort_data, sort_size * sizeof(int));
    cudaMemcpy(d_sort_data, h_sort_data, sort_size * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Original: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_sort_data[i]);
    }
    printf("...\n");
    
    int sort_blocks = (sort_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bitonic_sort_coop<<<sort_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_sort_data, sort_size);
    
    int *h_sorted = (int*)malloc(sort_size * sizeof(int));
    cudaMemcpy(h_sorted, d_sort_data, sort_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sorted:   ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_sorted[i]);
    }
    printf("...\n");
    
    // 4. Subgroup operations
    printf("\n=== Subgroup Operations ===\n");
    subgroup_operations<<<1, 128>>>();
    cudaDeviceSynchronize();
    
    // 5. Histogram with cooperative groups
    printf("\n=== Histogram with Cooperative Groups ===\n");
    
    int hist_size = 1024;
    int num_bins = 16;
    int *h_hist_data = (int*)malloc(hist_size * sizeof(int));
    int *h_histogram = (int*)malloc(num_bins * sizeof(int));
    int *d_hist_data, *d_histogram;
    
    for (int i = 0; i < hist_size; i++) {
        h_hist_data[i] = rand() % num_bins;
    }
    
    cudaMalloc(&d_hist_data, hist_size * sizeof(int));
    cudaMalloc(&d_histogram, num_bins * sizeof(int));
    
    cudaMemcpy(d_hist_data, h_hist_data, hist_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_histogram, 0, num_bins * sizeof(int));
    
    int hist_blocks = (hist_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    histogram_coop<<<hist_blocks, BLOCK_SIZE, num_bins * sizeof(int)>>>(
        d_hist_data, d_histogram, hist_size, num_bins);
    
    cudaMemcpy(h_histogram, d_histogram, num_bins * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Histogram results:\n");
    for (int i = 0; i < num_bins; i++) {
        printf("Bin %d: %d\n", i, h_histogram[i]);
    }
    
    printf("\nKey Learnings:\n");
    printf("- Cooperative groups provide flexible thread synchronization\n");
    printf("- Enable sub-warp and cross-warp coordination\n");
    printf("- Warp-level primitives are more efficient than block-level\n");
    printf("- Tiled partitions allow variable group sizes\n");
    printf("- Shuffle operations work within cooperative groups\n");
    printf("- Useful for irregular algorithms and load balancing\n");
    printf("- Can improve performance of reduction and scan operations\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_sort_data);
    cudaFree(d_hist_data);
    cudaFree(d_histogram);
    free(h_input);
    free(h_output);
    free(h_sort_data);
    free(h_sorted);
    free(h_hist_data);
    free(h_histogram);
    
    return 0;
}