#include <stdio.h>
#include <cuda_runtime.h>

__global__ void child_kernel(float *data, int start, int end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < end) {
        data[idx] = data[idx] * data[idx];
    }
}

__global__ void parent_kernel(float *data, int n, int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int chunk_size = n / (gridDim.x * blockDim.x);
    int start = idx * chunk_size;
    int end = min(start + chunk_size, n);
    
    if (idx < gridDim.x * blockDim.x && start < n) {
        if (chunk_size > threshold) {
            int child_blocks = (chunk_size + 255) / 256;
            child_kernel<<<child_blocks, 256>>>(data, start, end);
            cudaDeviceSynchronize();
        } else {
            for (int i = start; i < end; i++) {
                data[i] = data[i] * data[i];
            }
        }
    }
}

int main() {
    const int n = 4096;
    const int threshold = 512;
    
    float *h_data = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = i + 1.0f;
    }
    
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    
    parent_kernel<<<4, 64>>>(d_data, n, threshold);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Dynamic parallelism example (squares):\n");
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.0f^2 = %.0f\n", (float)(i + 1), h_data[i]);
    }
    
    cudaFree(d_data);
    free(h_data);
    
    return 0;
}
