#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024*1024
#define BLOCK_SIZE 256

// Simple vector addition kernel
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Kernel that accesses data in a specific pattern
__global__ void strided_access(float *data, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx * stride < n) {
        data[idx * stride] *= 2.0f;
    }
}

// CPU function for comparison
void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void demonstrate_traditional_memory() {
    printf("=== Traditional Memory Management ===\n");
    
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i + 1);
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Copy to device
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Execute kernel
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid_size, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    
    // Copy back to host
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Traditional memory time: %.3f ms\n", time);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_unified_memory() {
    printf("\n=== Unified Memory Management ===\n");
    
    size_t bytes = N * sizeof(float);
    float *a, *b, *c;
    
    // Allocate unified memory
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    
    // Initialize data (can be done from host)
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i + 1);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Execute kernel directly on unified memory
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid_size, BLOCK_SIZE>>>(a, b, c, N);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Unified memory time: %.3f ms\n", time);
    
    // Verify results (can access directly from host)
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (abs(c[i] - (a[i] + b[i])) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_memory_prefetching() {
    printf("\n=== Memory Prefetching Optimization ===\n");
    
    size_t bytes = N * sizeof(float);
    float *data;
    cudaMallocManaged(&data, bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }
    
    int device = 0;
    cudaGetDevice(&device);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Test without prefetching
    cudaEventRecord(start);
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid_size, BLOCK_SIZE>>>(data, data, data, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_no_prefetch;
    cudaEventElapsedTime(&time_no_prefetch, start, stop);
    printf("Without prefetching: %.3f ms\n", time_no_prefetch);
    
    // Reset data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }
    
    // Test with prefetching
    cudaEventRecord(start);
    cudaMemPrefetchAsync(data, bytes, device);
    vector_add<<<grid_size, BLOCK_SIZE>>>(data, data, data, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time_with_prefetch;
    cudaEventElapsedTime(&time_with_prefetch, start, stop);
    printf("With prefetching: %.3f ms\n", time_with_prefetch);
    printf("Prefetch speedup: %.2fx\n", time_no_prefetch / time_with_prefetch);
    
    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_memory_advise() {
    printf("\n=== Memory Advise Optimization ===\n");
    
    size_t bytes = N * sizeof(float);
    float *data;
    cudaMallocManaged(&data, bytes);
    
    int device = 0;
    cudaGetDevice(&device);
    
    // Set memory advice
    cudaMemAdvise(data, bytes, cudaMemAdviseSetReadMostly, device);
    cudaMemAdvise(data, bytes, cudaMemAdviseSetPreferredLocation, device);
    cudaMemAdvise(data, bytes, cudaMemAdviseSetAccessedBy, cudaCpuDeviceId);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    vector_add<<<grid_size, BLOCK_SIZE>>>(data, data, data, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("With memory advise: %.3f ms\n", time);
    
    printf("Memory advise hints help the runtime optimize data placement\n");
    
    cudaFree(data);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_concurrent_access() {
    printf("\n=== Concurrent Host-Device Access ===\n");
    
    size_t bytes = N * sizeof(float);
    float *data;
    cudaMallocManaged(&data, bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Launch kernel asynchronously
    strided_access<<<grid_size, BLOCK_SIZE, 0, stream>>>(data, N, 2);
    
    // CPU can work on different parts of the data concurrently
    // Note: This requires careful coordination to avoid race conditions
    for (int i = 1; i < N; i += 2) { // Work on odd indices
        data[i] += 1.0f;
    }
    
    // Wait for GPU to finish
    cudaStreamSynchronize(stream);
    
    printf("Concurrent CPU-GPU access completed\n");
    printf("First few results: %.1f, %.1f, %.1f, %.1f\n", 
           data[0], data[1], data[2], data[3]);
    
    cudaStreamDestroy(stream);
    cudaFree(data);
}

void demonstrate_oversubscription() {
    printf("\n=== Memory Oversubscription ===\n");
    
    // Allocate more memory than available on GPU
    size_t gpu_memory;
    cudaMemGetInfo(&gpu_memory, &gpu_memory);
    size_t large_size = gpu_memory * 2; // 2x GPU memory
    
    printf("Attempting to allocate %zu MB (2x GPU memory)\n", large_size / (1024*1024));
    
    float *large_data;
    cudaError_t err = cudaMallocManaged(&large_data, large_size);
    
    if (err == cudaSuccess) {
        printf("Large allocation successful - unified memory handles oversubscription\n");
        
        // Touch some of the data
        for (size_t i = 0; i < large_size / sizeof(float); i += 1024*1024) {
            large_data[i] = 1.0f;
        }
        
        printf("Data access successful - pages migrated as needed\n");
        cudaFree(large_data);
    } else {
        printf("Large allocation failed: %s\n", cudaGetErrorString(err));
    }
}

void query_unified_memory_support() {
    printf("=== Unified Memory Support Query ===\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Unified Memory: %s\n", prop.managedMemory ? "Supported" : "Not supported");
    printf("Concurrent Managed Access: %s\n", 
           prop.concurrentManagedAccess ? "Supported" : "Not supported");
    printf("Page Migration: %s\n", 
           prop.pageableMemoryAccess ? "Supported" : "Not supported");
    
    if (prop.managedMemory) {
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        printf("GPU Memory: %zu MB free / %zu MB total\n", 
               free_mem / (1024*1024), total_mem / (1024*1024));
    }
    printf("\n");
}

int main() {
    printf("=== CUDA Unified Memory Demo ===\n");
    printf("This example demonstrates unified memory features and optimizations.\n\n");
    
    // Check unified memory support
    query_unified_memory_support();
    
    // Compare traditional vs unified memory
    demonstrate_traditional_memory();
    demonstrate_unified_memory();
    
    // Advanced unified memory features
    demonstrate_memory_prefetching();
    demonstrate_memory_advise();
    demonstrate_concurrent_access();
    demonstrate_oversubscription();
    
    printf("\nKey Learnings:\n");
    printf("- Unified memory simplifies memory management\n");
    printf("- Automatic data migration between host and device\n");
    printf("- Prefetching can improve performance significantly\n");
    printf("- Memory advise hints help optimize data placement\n");
    printf("- Enables memory oversubscription beyond GPU capacity\n");
    printf("- Concurrent access requires careful synchronization\n");
    printf("- Check device capabilities before using advanced features\n");
    printf("- May have performance overhead compared to explicit management\n");
    
    return 0;
}