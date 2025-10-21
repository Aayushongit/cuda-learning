#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(float *odata, float *idata, int width, int height) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
        }
    }
    
    __syncthreads();
    
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            odata[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

int main() {
    const int width = 64;
    const int height = 64;
    const int size = width * height * sizeof(float);
    
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);
    
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            h_idata[i * width + j] = i * width + j;
        }
    }
    
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);
    
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM, 1);
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS, 1);
    
    transpose<<<dimGrid, dimBlock>>>(d_odata, d_idata, width, height);
    
    cudaMemcpy(h_odata, d_odata, size, cudaMemcpyDeviceToHost);
    
    printf("Matrix transpose using shared memory:\n");
    printf("Original (4x4 corner):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.0f ", h_idata[i * width + j]);
        }
        printf("\n");
    }
    
    printf("Transposed (4x4 corner):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.0f ", h_odata[i * height + j]);
        }
        printf("\n");
    }
    
    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    
    return 0;
}
