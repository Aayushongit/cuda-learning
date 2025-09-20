#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define N 1024*1024
#define BLOCK_SIZE 256

__global__ void reduction_naive(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Naive reduction with divergent warps
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_optimized(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Optimized reduction - no divergent warps
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_warp_optimized(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Reduce using shared memory
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no synchronization needed within warp)
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

__global__ void reduction_shuffle(float *input, float *output, int n) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data
    float val = (idx < n) ? input[idx] : 0.0f;
    
    // Warp-level reduction using shuffle instructions
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    
    // Store result from first thread of each warp
    __shared__ float warp_results[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction of warp results
    if (tid < 32) {
        val = warp_results[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
    }
    
    if (tid == 0) {
        output[blockIdx.x] = val;
    }
}

float perform_reduction(void (*kernel)(float*, float*, int), 
                       float *d_input, float *d_temp, int n, const char *name) {
    int num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // First level reduction
    if (kernel == reduction_shuffle) {
        kernel<<<num_blocks, BLOCK_SIZE>>>(d_input, d_temp, n);
    } else {
        kernel<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_temp, n);
    }
    
    // Continue until we have single result
    int remaining = num_blocks;
    float *current_input = d_temp;
    while (remaining > 1) {
        int next_blocks = (remaining + BLOCK_SIZE - 1) / BLOCK_SIZE;
        if (kernel == reduction_shuffle) {
            kernel<<<next_blocks, BLOCK_SIZE>>>(current_input, d_temp, remaining);
        } else {
            kernel<<<next_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(current_input, d_temp, remaining);
        }
        remaining = next_blocks;
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%s: %.3f ms\n", name, time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time;
}

int main() {
    printf("=== Parallel Reduction Optimization Demo ===\n");
    printf("This example shows different reduction strategies and their performance impact.\n\n");
    
    size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    
    // Initialize with ones for easy verification
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_temp;
    int max_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_temp, max_blocks * sizeof(float));
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Test different reduction strategies
    float naive_time = perform_reduction(reduction_naive, d_input, d_temp, N, 
                                       "Naive Reduction (divergent warps)");
    
    float optimized_time = perform_reduction(reduction_optimized, d_input, d_temp, N,
                                           "Optimized Reduction (no divergent warps)");
    
    float warp_time = perform_reduction(reduction_warp_optimized, d_input, d_temp, N,
                                       "Warp-Optimized Reduction");
    
    float shuffle_time = perform_reduction(reduction_shuffle, d_input, d_temp, N,
                                         "Shuffle-based Reduction");
    
    // Verify result
    float result;
    cudaMemcpy(&result, d_temp, sizeof(float), cudaMemcpyDeviceToHost);
    printf("\nResult: %.0f (Expected: %d)\n", result, N);
    
    printf("\nPerformance Analysis:\n");
    printf("Optimized vs Naive: %.2fx speedup\n", naive_time / optimized_time);
    printf("Warp-optimized vs Optimized: %.2fx speedup\n", optimized_time / warp_time);
    printf("Shuffle vs Warp-optimized: %.2fx speedup\n", warp_time / shuffle_time);
    
    printf("\nKey Learnings:\n");
    printf("- Avoid warp divergence in reduction loops\n");
    printf("- Use warp-level primitives for final reduction steps\n");
    printf("- Shuffle instructions eliminate shared memory for warp operations\n");
    printf("- Multiple reduction levels may be needed for large arrays\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_temp);
    free(h_input);
    
    return 0;
}