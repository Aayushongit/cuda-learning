#include <stdio.h>
#include <cuda_runtime.h>

__global__ void histogram(unsigned char *data, unsigned int *hist, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    
    __shared__ unsigned int hist_private[256];
    
    if (threadIdx.x < 256) {
        hist_private[threadIdx.x] = 0;
    }
    __syncthreads();
    
    while (tid < n) {
        atomicAdd(&hist_private[data[tid]], 1);
        tid += offset;
    }
    __syncthreads();
    
    if (threadIdx.x < 256) {
        atomicAdd(&hist[threadIdx.x], hist_private[threadIdx.x]);
    }
}

int main() {
    const int n = 1024 * 1024;
    
    unsigned char *h_data = (unsigned char*)malloc(n * sizeof(unsigned char));
    unsigned int *h_hist = (unsigned int*)calloc(256, sizeof(unsigned int));
    
    for (int i = 0; i < n; i++) {
        h_data[i] = i % 256;
    }
    
    unsigned char *d_data;
    unsigned int *d_hist;
    cudaMalloc(&d_data, n * sizeof(unsigned char));
    cudaMalloc(&d_hist, 256 * sizeof(unsigned int));
    
    cudaMemcpy(d_data, h_data, n * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, 256 * sizeof(unsigned int));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = 64;
    
    histogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, n);
    
    cudaMemcpy(h_hist, d_hist, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    printf("Histogram with thread synchronization and atomics:\n");
    printf("First 10 bins:\n");
    for (int i = 0; i < 10; i++) {
        printf("Bin %d: %u\n", i, h_hist[i]);
    }
    
    unsigned int total = 0;
    for (int i = 0; i < 256; i++) {
        total += h_hist[i];
    }
    printf("Total count: %u (expected: %d)\n", total, n);
    
    cudaFree(d_data);
    cudaFree(d_hist);
    free(h_data);
    free(h_hist);
    
    return 0;
}