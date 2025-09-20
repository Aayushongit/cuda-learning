#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <stdio.h>

#define N 1024*1024

// Kernel with high register usage (poor occupancy)
__global__ void high_register_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Use many registers to reduce occupancy
    float reg1 = data[idx];
    float reg2 = reg1 * 2.0f;
    float reg3 = reg2 * 3.0f;
    float reg4 = reg3 * 4.0f;
    float reg5 = reg4 * 5.0f;
    float reg6 = reg5 * 6.0f;
    float reg7 = reg6 * 7.0f;
    float reg8 = reg7 * 8.0f;
    float reg9 = reg8 * 9.0f;
    float reg10 = reg9 * 10.0f;
    float reg11 = reg10 * 11.0f;
    float reg12 = reg11 * 12.0f;
    float reg13 = reg12 * 13.0f;
    float reg14 = reg13 * 14.0f;
    float reg15 = reg14 * 15.0f;
    float reg16 = reg15 * 16.0f;
    
    // Complex computation using all registers
    float result = (reg1 + reg2 + reg3 + reg4 + reg5 + reg6 + reg7 + reg8 +
                   reg9 + reg10 + reg11 + reg12 + reg13 + reg14 + reg15 + reg16) / 16.0f;
    
    data[idx] = result;
}

// Optimized kernel with fewer registers (better occupancy)
__global__ void low_register_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Use fewer registers by reusing variables
    float temp = data[idx];
    for (int i = 2; i <= 16; i++) {
        temp = temp * i;
    }
    temp = temp / 16.0f;
    
    data[idx] = temp;
}

// Kernel with large shared memory usage
__global__ void large_shared_memory_kernel(float *data, int n) {
    extern __shared__ float sdata[];
    // Large shared memory allocation reduces occupancy
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Simple computation
    float result = sdata[tid] * 2.0f;
    if (tid > 0) {
        result += sdata[tid - 1];
    }
    if (tid < blockDim.x - 1) {
        result += sdata[tid + 1];
    }
    
    __syncthreads();
    
    if (idx < n) {
        data[idx] = result;
    }
}

// Optimized kernel with minimal shared memory
__global__ void small_shared_memory_kernel(float *data, int n) {
    __shared__ float sdata[256]; // Fixed small size
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Same computation as large version
    float result = sdata[tid] * 2.0f;
    if (tid > 0) {
        result += sdata[tid - 1];
    }
    if (tid < blockDim.x - 1) {
        result += sdata[tid + 1];
    }
    
    __syncthreads();
    
    if (idx < n) {
        data[idx] = result;
    }
}

// Kernel optimized for specific block size
__global__ void __launch_bounds__(256, 4) // 256 threads, min 4 blocks per SM
occupancy_optimized_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float temp = data[idx];
    
    // Optimized computation that balances registers and performance
    temp = temp * 1.5f + 0.5f;
    temp = sqrtf(temp);
    temp = temp * temp + 1.0f;
    
    data[idx] = temp;
}

void analyze_occupancy() {
    printf("=== Occupancy Analysis ===\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Number of SMs: %d\n", prop.multiProcessorCount);
    printf("Shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Registers per block: %d\n", prop.regsPerBlock);
    printf("\n");
    
    // Calculate theoretical occupancy for different kernels
    int min_grid_size, block_size;
    size_t dynamic_shared_mem = 0;
    
    // High register kernel
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                      high_register_kernel, dynamic_shared_mem, 0);
    printf("High register kernel - Optimal block size: %d\n", block_size);
    
    float occupancy;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&min_grid_size, 
                                                 high_register_kernel, block_size, dynamic_shared_mem);
    occupancy = (float)min_grid_size * block_size / prop.maxThreadsPerMultiProcessor;
    printf("High register kernel - Theoretical occupancy: %.1f%%\n", occupancy * 100);
    
    // Low register kernel
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                      low_register_kernel, dynamic_shared_mem, 0);
    printf("Low register kernel - Optimal block size: %d\n", block_size);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&min_grid_size, 
                                                 low_register_kernel, block_size, dynamic_shared_mem);
    occupancy = (float)min_grid_size * block_size / prop.maxThreadsPerMultiProcessor;
    printf("Low register kernel - Theoretical occupancy: %.1f%%\n", occupancy * 100);
    
    // Large shared memory kernel (with 4KB shared memory)
    dynamic_shared_mem = 4096;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                      large_shared_memory_kernel, dynamic_shared_mem, 0);
    printf("Large shared memory kernel - Optimal block size: %d\n", block_size);
    
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&min_grid_size, 
                                                 large_shared_memory_kernel, block_size, dynamic_shared_mem);
    occupancy = (float)min_grid_size * block_size / prop.maxThreadsPerMultiProcessor;
    printf("Large shared memory kernel - Theoretical occupancy: %.1f%%\n", occupancy * 100);
    
    printf("\n");
}

float benchmark_kernel(const char *name, void (*kernel_launcher)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel_launcher();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        kernel_launcher();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    time /= 10.0f; // Average time
    
    printf("%s: %.3f ms\n", name, time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time;
}

int main() {
    printf("=== CUDA Occupancy Optimization Demo ===\n");
    printf("This example demonstrates how occupancy affects kernel performance.\n\n");
    
    analyze_occupancy();
    
    size_t bytes = N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i / N;
    }
    
    float *d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    int grid_size = (N + 255) / 256;
    
    printf("=== Performance Comparison ===\n");
    
    // High register usage kernel
    auto high_reg_test = [&]() {
        high_register_kernel<<<grid_size, 256>>>(d_data, N);
    };
    float high_reg_time = benchmark_kernel("High Register Kernel", high_reg_test);
    
    // Low register usage kernel
    auto low_reg_test = [&]() {
        low_register_kernel<<<grid_size, 256>>>(d_data, N);
    };
    float low_reg_time = benchmark_kernel("Low Register Kernel", low_reg_test);
    
    // Large shared memory kernel
    auto large_shmem_test = [&]() {
        large_shared_memory_kernel<<<grid_size, 256, 4096>>>(d_data, N);
    };
    float large_shmem_time = benchmark_kernel("Large Shared Memory Kernel", large_shmem_test);
    
    // Small shared memory kernel
    auto small_shmem_test = [&]() {
        small_shared_memory_kernel<<<grid_size, 256>>>(d_data, N);
    };
    float small_shmem_time = benchmark_kernel("Small Shared Memory Kernel", small_shmem_test);
    
    // Occupancy optimized kernel
    auto optimized_test = [&]() {
        occupancy_optimized_kernel<<<grid_size, 256>>>(d_data, N);
    };
    float optimized_time = benchmark_kernel("Occupancy Optimized Kernel", optimized_test);
    
    printf("\nSpeedup Analysis:\n");
    printf("Low register vs High register: %.2fx\n", high_reg_time / low_reg_time);
    printf("Small shmem vs Large shmem: %.2fx\n", large_shmem_time / small_shmem_time);
    printf("Optimized vs High register: %.2fx\n", high_reg_time / optimized_time);
    
    printf("\nKey Learnings:\n");
    printf("- Higher occupancy often leads to better performance\n");
    printf("- Register usage is a major factor limiting occupancy\n");
    printf("- Shared memory usage can also limit occupancy\n");
    printf("- Use __launch_bounds__ to optimize for specific configurations\n");
    printf("- CUDA occupancy calculator helps find optimal parameters\n");
    printf("- Balance between occupancy and per-thread work is crucial\n");
    printf("- Sometimes lower occupancy with more work per thread is better\n");
    
    // Cleanup
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}