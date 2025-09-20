#include <stdio.h>
#include <cuda_runtime.h>

__global__ void transpose_naive(float *odata, float *idata, int width) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < width) {
        odata[x * width + y] = idata[y * width + x];
    }
}

__global__ void transpose_coalesced(float *odata, float *idata, int width) {
    __shared__ float tile[32][33];
    
    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    }
    
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        odata[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

int main() {
    const int width = 1024;
    const int size = width * width * sizeof(float);
    
    float *h_idata = (float*)malloc(size);
    float *h_odata1 = (float*)malloc(size);
    float *h_odata2 = (float*)malloc(size);
    
    for (int i = 0; i < width * width; i++) {
        h_idata[i] = i;
    }
    
    float *d_idata, *d_odata1, *d_odata2;
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata1, size);
    cudaMalloc(&d_odata2, size);
    
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    
    dim3 dimGrid((width + 31) / 32, (width + 31) / 32);
    dim3 dimBlock(32, 32);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    transpose_naive<<<dimGrid, dimBlock>>>(d_odata1, d_idata, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naive_time;
    cudaEventElapsedTime(&naive_time, start, stop);
    
    cudaEventRecord(start);
    transpose_coalesced<<<dimGrid, dimBlock>>>(d_odata2, d_idata, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float coalesced_time;
    cudaEventElapsedTime(&coalesced_time, start, stop);
    
    cudaMemcpy(h_odata1, d_odata1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_odata2, d_odata2, size, cudaMemcpyDeviceToHost);
    
    printf("Memory coalescing comparison:\n");
    printf("Naive transpose time: %.3f ms\n", naive_time);
    printf("Coalesced transpose time: %.3f ms\n", coalesced_time);
    printf("Speedup: %.2fx\n", naive_time / coalesced_time);
    
    bool results_match = true;
    for (int i = 0; i < 100 && results_match; i++) {
        if (h_odata1[i] != h_odata2[i]) {
            results_match = false;
        }
    }
    printf("Results match: %s\n", results_match ? "Yes" : "No");
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_idata);
    cudaFree(d_odata1);
    cudaFree(d_odata2);
    free(h_idata);
    free(h_odata1);
    free(h_odata2);
    
    return 0;
}
