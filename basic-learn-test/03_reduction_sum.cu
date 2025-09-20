#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reduce(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    const int n = 1024;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    float *h_input = (float*)malloc(n * sizeof(float));
    float *h_output = (float*)malloc(blocksPerGrid * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, blocksPerGrid * sizeof(float));
    
    cudaMemcpy(d_input, h_input, n * sizeof(float), cudaMemcpyHostToDevice);
    
    reduce<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, n);
    
    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_sum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        total_sum += h_output[i];
    }
    
    printf("Reduction sum of %d elements: %.1f\n", n, total_sum);
    printf("Expected: %.1f\n", (float)n);
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}