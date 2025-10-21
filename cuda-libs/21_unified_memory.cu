// Unified Memory: Automatic memory management between CPU and GPU
// Simplifies development with page migration and prefetching

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

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void matrixInit(float* matrix, int rows, int cols) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < rows * cols) {
        matrix[tid] = (float)(tid % 100);
    }
}

int main() {
    const int N = 1 << 24;  // 16M elements
    const size_t bytes = N * sizeof(float);

    printf("=== Unified Memory Management ===\n");
    printf("Array size: %d elements (%.2f MB)\n\n", N, bytes / 1024.0 / 1024.0);

    // Check Unified Memory support
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));

    printf("GPU: %s\n", prop.name);
    if (prop.managedMemory) {
        printf("✓ Unified Memory supported\n");
    } else {
        printf("✗ Unified Memory NOT supported\n");
    }

    if (prop.concurrentManagedAccess) {
        printf("✓ Concurrent CPU/GPU access supported\n");
    }
    printf("\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ===== METHOD 1: Traditional malloc/cudaMalloc =====
    printf("=== Traditional Memory Management ===\n");

    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    float *d_a, *d_b, *d_c;

    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Initialize
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    CHECK_CUDA(cudaEventRecord(start));

    // Manual memory transfers
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);

    CHECK_CUDA(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_traditional = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_traditional, start, stop));

    printf("Time: %.3f ms\n\n", ms_traditional);

    free(h_a); free(h_b); free(h_c);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    // ===== METHOD 2: Unified Memory =====
    printf("=== Unified Memory (cudaMallocManaged) ===\n");

    float *um_a, *um_b, *um_c;
    CHECK_CUDA(cudaMallocManaged(&um_a, bytes));
    CHECK_CUDA(cudaMallocManaged(&um_b, bytes));
    CHECK_CUDA(cudaMallocManaged(&um_c, bytes));

    // Initialize directly from CPU
    for (int i = 0; i < N; i++) {
        um_a[i] = i;
        um_b[i] = i * 2;
    }

    CHECK_CUDA(cudaEventRecord(start));

    // No explicit memory transfers needed!
    vectorAdd<<<gridSize, blockSize>>>(um_a, um_b, um_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Access results directly from CPU
    float sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += um_c[i];
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_unified = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_unified, start, stop));

    printf("Time: %.3f ms\n", ms_unified);
    printf("Verification sum (first 100): %.2f\n\n", sum);

    // ===== METHOD 3: Unified Memory with Prefetching =====
    printf("=== Unified Memory with Prefetching ===\n");

    CHECK_CUDA(cudaEventRecord(start));

    // Prefetch data to GPU before kernel launch
    CHECK_CUDA(cudaMemPrefetchAsync(um_a, bytes, 0));  // 0 = device 0
    CHECK_CUDA(cudaMemPrefetchAsync(um_b, bytes, 0));

    vectorAdd<<<gridSize, blockSize>>>(um_a, um_b, um_c, N);

    // Prefetch results back to CPU
    CHECK_CUDA(cudaMemPrefetchAsync(um_c, bytes, cudaCpuDeviceId));
    CHECK_CUDA(cudaDeviceSynchronize());

    sum = 0.0f;
    for (int i = 0; i < 100; i++) {
        sum += um_c[i];
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_prefetch = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_prefetch, start, stop));

    printf("Time: %.3f ms\n", ms_prefetch);
    printf("Verification sum (first 100): %.2f\n\n", sum);

    // ===== METHOD 4: Memory Advise =====
    printf("=== Unified Memory with Hints ===\n");

    // Give hints to the CUDA driver
    CHECK_CUDA(cudaMemAdvise(um_a, bytes, cudaMemAdviseSetReadMostly, 0));
    CHECK_CUDA(cudaMemAdvise(um_b, bytes, cudaMemAdviseSetReadMostly, 0));
    CHECK_CUDA(cudaMemAdvise(um_c, bytes, cudaMemAdviseSetPreferredLocation, 0));

    CHECK_CUDA(cudaEventRecord(start));

    vectorAdd<<<gridSize, blockSize>>>(um_a, um_b, um_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_advise = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_advise, start, stop));

    printf("Time: %.3f ms\n\n", ms_advise);

    // ===== Concurrent Access Example =====
    printf("=== Concurrent CPU/GPU Access ===\n");

    float *concurrent_data;
    CHECK_CUDA(cudaMallocManaged(&concurrent_data, bytes));

    // Initialize on GPU
    matrixInit<<<gridSize, blockSize>>>(concurrent_data, 1, N);

    // With concurrent access, CPU can read while GPU writes (if supported)
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("First 10 values: ");
    for (int i = 0; i < 10; i++) {
        printf("%.0f ", concurrent_data[i]);
    }
    printf("\n\n");

    // ===== Performance Summary =====
    printf("=== Performance Comparison ===\n");
    printf("Traditional:         %.3f ms (1.00x)\n", ms_traditional);
    printf("Unified Memory:      %.3f ms (%.2fx)\n",
           ms_unified, ms_traditional / ms_unified);
    printf("With Prefetching:    %.3f ms (%.2fx)\n",
           ms_prefetch, ms_traditional / ms_prefetch);
    printf("With Advise:         %.3f ms (%.2fx)\n",
           ms_advise, ms_traditional / ms_advise);

    printf("\n=== Benefits of Unified Memory ===\n");
    printf("✓ Simpler code (no explicit cudaMemcpy)\n");
    printf("✓ Automatic data migration\n");
    printf("✓ Oversubscription support\n");
    printf("✓ Better for prototyping\n");
    printf("✓ Can optimize with prefetch/advise\n");

    // Cleanup
    CHECK_CUDA(cudaFree(um_a));
    CHECK_CUDA(cudaFree(um_b));
    CHECK_CUDA(cudaFree(um_c));
    CHECK_CUDA(cudaFree(concurrent_data));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nUnified Memory examples completed successfully!\n");
    return 0;
}
