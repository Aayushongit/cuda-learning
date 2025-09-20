#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>

#define N 1024*1024
#define BLOCK_SIZE 256

// Intentionally inefficient kernel for profiling
__global__ void inefficient_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Bad: uncoalesced memory access
        int stride = 32;
        if (idx * stride < n) {
            data[idx * stride] = data[idx * stride] * 2.0f + 1.0f;
        }
        
        // Bad: divergent branching
        if (idx % 2 == 0) {
            for (int i = 0; i < 100; i++) {
                data[idx] = sqrtf(data[idx]);
            }
        } else {
            for (int i = 0; i < 50; i++) {
                data[idx] = data[idx] * data[idx];
            }
        }
        
        // Bad: expensive operations in loop
        for (int i = 0; i < 10; i++) {
            data[idx] = sinf(data[idx]) + cosf(data[idx]);
        }
    }
}

// Optimized version of the kernel
__global__ void optimized_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Good: coalesced memory access
        float value = data[idx];
        
        // Reduced branching and computation
        value = value * 2.0f + 1.0f;
        
        // Simplified computation
        value = value * value; // Square operation
        value = value + 0.5f;  // Simple addition
        
        data[idx] = value;
    }
}

// Kernel with debug prints (use sparingly)
__global__ void debug_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Only print from a few threads to avoid overwhelming output
    if (idx < 10) {
        printf("Thread %d: processing data[%d] = %.2f\n", idx, idx, data[idx]);
    }
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
    
    // Debug assertion (only in debug builds)
    #ifdef DEBUG
    assert(data[idx] >= 0.0f);
    #endif
}

// Kernel that demonstrates common bugs
__global__ void buggy_kernel(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bug 1: No bounds checking (can cause segmentation fault)
    // if (idx < n) {  // This line is commented out intentionally
        output[idx] = input[idx] * 2.0f;
    // }
    
    // Bug 2: Race condition in shared memory
    __shared__ float shared_data[256];
    shared_data[threadIdx.x] = input[idx];
    // Missing __syncthreads() here
    
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; i++) {
            sum += shared_data[i]; // May read uninitialized data
        }
        output[blockIdx.x] = sum;
    }
}

// Function to demonstrate CUDA error checking
void checkCudaError(cudaError_t error, const char *operation) {
    if (error != cudaSuccess) {
        printf("CUDA Error during %s: %s\n", operation, cudaGetErrorString(error));
        exit(1);
    }
}

void demonstrate_timing_and_profiling() {
    printf("=== Timing and Profiling Demo ===\n");
    
    size_t bytes = N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    float *d_data;
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i / N;
    }
    
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Method 1: Basic timing with cudaEventRecord
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    printf("\n--- Method 1: CUDA Events ---\n");
    cudaEventRecord(start);
    inefficient_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float inefficient_time;
    cudaEventElapsedTime(&inefficient_time, start, stop);
    printf("Inefficient kernel time: %.3f ms\n", inefficient_time);
    
    // Reset data
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    cudaEventRecord(start);
    optimized_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float optimized_time;
    cudaEventElapsedTime(&optimized_time, start, stop);
    printf("Optimized kernel time: %.3f ms\n", optimized_time);
    printf("Speedup: %.2fx\n", inefficient_time / optimized_time);
    
    // Method 2: Using CUDA Profiler API
    printf("\n--- Method 2: CUDA Profiler API ---\n");
    printf("Starting profiler...\n");
    
    cudaProfilerStart();\n    
    inefficient_kernel<<<grid_size, BLOCK_SIZE>>>(d_data, N);
    cudaDeviceSynchronize();
    cudaProfilerStop();
    
    printf("Profiler stopped. Check nvprof or Nsight output for detailed metrics.\n");
    
    // Method 3: Memory bandwidth analysis
    printf("\n--- Method 3: Memory Bandwidth Analysis ---\n");
    
    size_t bytes_transferred = bytes * 2; // Read + Write
    float bandwidth_gb_s = (bytes_transferred / (1024.0*1024.0*1024.0)) / (optimized_time / 1000.0);
    
    printf("Data size: %.1f MB\n", bytes / (1024.0*1024.0));
    printf("Bytes transferred: %.1f MB\n", bytes_transferred / (1024.0*1024.0));
    printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gb_s);
    
    // Get device peak bandwidth for comparison
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    float peak_bandwidth = (prop.memoryClockRate * 2.0 * prop.memoryBusWidth / 8) / 1e6;
    printf("Peak memory bandwidth: %.2f GB/s\n", peak_bandwidth);
    printf("Bandwidth efficiency: %.1f%%\n", (bandwidth_gb_s / peak_bandwidth) * 100);
    
    cudaFree(d_data);
    free(h_data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_error_checking() {
    printf("\n=== Error Checking and Debugging ===\n");
    
    size_t bytes = 1024 * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    float *d_input, *d_output;
    
    // Initialize data
    for (int i = 0; i < 1024; i++) {
        h_data[i] = (float)i;
    }
    
    // Proper error checking for memory allocation
    cudaError_t error = cudaMalloc(&d_input, bytes);
    checkCudaError(error, "cudaMalloc d_input");
    
    error = cudaMalloc(&d_output, bytes);
    checkCudaError(error, "cudaMalloc d_output");
    
    error = cudaMemcpy(d_input, h_data, bytes, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy to device");
    
    printf("Memory allocation and copy: SUCCESS\n");
    
    // Launch debug kernel
    printf("\n--- Debug Kernel Output ---\n");
    debug_kernel<<<4, 256>>>(d_input, 1024);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    checkCudaError(error, "debug_kernel launch");
    
    // Synchronize and check for execution errors
    error = cudaDeviceSynchronize();
    checkCudaError(error, "debug_kernel execution");
    
    printf("\n--- Demonstrating Common Bugs ---\n");
    printf("Intentionally buggy kernel (may cause errors):\n");
    
    // This might cause runtime errors
    buggy_kernel<<<4, 256>>>(d_input, d_output, 1024);
    
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Expected error from buggy kernel: %s\n", cudaGetErrorString(error));
        cudaGetLastError(); // Clear the error
    }
    
    // Demonstrate memory debugging
    printf("\n--- Memory Debugging ---\n");
    
    // Try to access invalid memory (this should be caught by CUDA-memcheck)
    float *invalid_ptr = NULL;
    error = cudaMemcpy(invalid_ptr, h_data, bytes, cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
        printf("Caught invalid memory access: %s\n", cudaGetErrorString(error));
        cudaGetLastError(); // Clear the error
    }
    
    // Memory leak detection (intentional)
    float *leaked_memory;
    cudaMalloc(&leaked_memory, bytes);
    printf("Allocated memory that won't be freed (leak detection demo)\n");
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_data);
    // Note: leaked_memory is intentionally not freed
}

void demonstrate_device_queries() {
    printf("\n=== Device Information for Debugging ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Number of CUDA devices: %d\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaSetDevice(i);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("\nDevice %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Clock Rate: %.1f MHz\n", prop.clockRate / 1000.0);
        printf("  Memory Clock Rate: %.1f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  L2 Cache Size: %.1f KB\n", prop.l2CacheSize / 1024.0);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Shared Memory per Block: %.1f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        
        // Memory information
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("  Memory: %.1f MB free / %.1f MB total\n", 
               free_mem / (1024.0*1024.0), total_mem / (1024.0*1024.0));
        
        // Current GPU usage (approximate)
        printf("  Memory Usage: %.1f%%\n", 
               (1.0 - (float)free_mem / total_mem) * 100);
    }
}

void demonstrate_assertion_and_debugging() {
    printf("\n=== Assertions and Debug Techniques ===\n");
    
    printf("Compile with -DDEBUG flag to enable assertions\n");
    printf("Use cuda-gdb for interactive debugging\n");
    printf("Use cuda-memcheck for memory error detection\n");
    printf("Use nvprof or Nsight for performance profiling\n");
    
    printf("\nUseful debugging commands:\n");
    printf("  nvcc -g -G -DDEBUG program.cu    # Debug build\n");
    printf("  cuda-memcheck ./program          # Memory checking\n");
    printf("  nvprof ./program                 # Basic profiling\n");
    printf("  nvprof --metrics all ./program   # Detailed metrics\n");
    printf("  cuda-gdb ./program               # Interactive debugging\n");
    
    printf("\nCommon debugging strategies:\n");
    printf("  1. Add printf statements in kernels (use sparingly)\n");
    printf("  2. Check CUDA errors after every CUDA API call\n");
    printf("  3. Use cudaDeviceSynchronize() to catch asynchronous errors\n");
    printf("  4. Reduce problem size to isolate issues\n");
    printf("  5. Use debug builds with assertions\n");
    printf("  6. Profile before and after optimizations\n");
}

int main() {
    printf("=== CUDA Profiling and Debugging Demo ===\n");
    printf("This example demonstrates profiling techniques and debugging strategies.\n\n");
    
    // Check compute capability for debugging features
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (Compute %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("Debug features available: %s\n", prop.major >= 2 ? "Yes" : "Limited");
    printf("\n");
    
    // Demonstrate timing and profiling
    demonstrate_timing_and_profiling();
    
    // Demonstrate error checking
    demonstrate_error_checking();
    
    // Device information for debugging
    demonstrate_device_queries();
    
    // Debugging techniques
    demonstrate_assertion_and_debugging();
    
    printf("\nKey Learnings:\n");
    printf("- Always check for CUDA errors after API calls\n");
    printf("- Use CUDA events for accurate GPU timing\n");
    printf("- Profile regularly to identify bottlenecks\n");
    printf("- Use debug builds during development\n");
    printf("- cuda-memcheck catches memory errors\n");
    printf("- nvprof provides detailed performance metrics\n");
    printf("- Print statements in kernels should be used sparingly\n");
    printf("- Reduce problem size to isolate bugs\n");
    printf("- Synchronize before checking for kernel errors\n");
    
    return 0;
}