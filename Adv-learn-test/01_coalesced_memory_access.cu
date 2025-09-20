#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define N 1024*1024
#define BLOCK_SIZE 256

__global__ void uncoalesced_access(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Bad: strided access pattern
        output[idx] = input[idx * 32] + 1.0f;
    }
}

__global__ void coalesced_access(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Good: consecutive access pattern
        output[idx] = input[idx] + 1.0f;
    }
}

float measure_time(void (*kernel)(float*, float*, int), 
                   float *d_input, float *d_output, int n, const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        if (kernel == (void(*)(float*,float*,int))uncoalesced_access) {
            uncoalesced_access<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
        } else {
            coalesced_access<<<grid_size, BLOCK_SIZE>>>(d_input, d_output, n);
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("%s: %.3f ms\n", name, time / 100.0f);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time / 100.0f;
}

int main() {
    printf("=== Memory Coalescing Optimization Demo ===\n");
    printf("This example demonstrates the impact of memory access patterns on performance.\n");
    printf("Coalesced access (consecutive memory locations) vs uncoalesced (strided) access.\n\n");
    
    size_t bytes = N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    
    // Initialize input data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes * 32); // Extra space for strided access
    cudaMalloc(&d_output, bytes);
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Measure uncoalesced access
    float uncoalesced_time = measure_time((void(*)(float*,float*,int))uncoalesced_access, 
                                        d_input, d_output, N, "Uncoalesced Access");
    
    // Measure coalesced access  
    float coalesced_time = measure_time((void(*)(float*,float*,int))coalesced_access,
                                      d_input, d_output, N, "Coalesced Access");
    
    printf("\nSpeedup with coalescing: %.2fx\n", uncoalesced_time / coalesced_time);
    printf("\nKey Learning:\n");
    printf("- Coalesced memory access can provide significant performance improvements\n");
    printf("- Always try to access consecutive memory locations within a warp\n");
    printf("- Memory bandwidth is often the bottleneck in GPU kernels\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}