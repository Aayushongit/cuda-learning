#include <stdio.h>
#include <cuda_runtime.h>

__constant__ float c_filter[9];

__global__ void convolution(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
        float sum = 0.0f;
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int idx = (y + i) * width + (x + j);
                int filter_idx = (i + 1) * 3 + (j + 1);
                sum += input[idx] * c_filter[filter_idx];
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    const int width = 32;
    const int height = 32;
    const int size = width * height * sizeof(float);
    
    float h_filter[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
    
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);
    
    for (int i = 0; i < width * height; i++) {
        h_input[i] = 1.0f;
    }
    
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_filter, h_filter, 9 * sizeof(float));
    
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    convolution<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height);
    
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);
    
    printf("Convolution using constant memory:\n");
    printf("Filter coefficients:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.0f ", h_filter[i * 3 + j]);
        }
        printf("\n");
    }
    
    printf("Result (center 4x4):\n");
    int start = height / 2 - 2;
    for (int i = start; i < start + 4; i++) {
        for (int j = start; j < start + 4; j++) {
            printf("%.1f ", h_output[i * width + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}