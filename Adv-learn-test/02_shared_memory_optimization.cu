#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define TILE_SIZE 32

__global__ void matrix_multiply_naive(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void matrix_multiply_shared(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        if (row < n && a_col < n) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * n + a_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (b_row < n && col < n) {
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * n + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum using shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

__global__ void shared_memory_bank_conflicts_demo() {
    __shared__ float shared_data[32][33]; // +1 to avoid bank conflicts
    __shared__ float shared_data_conflict[32][32]; // Will have bank conflicts
    
    int tid = threadIdx.x;
    
    // Good: no bank conflicts (different banks)
    shared_data[tid][0] = tid;
    __syncthreads();
    
    // Bad: bank conflicts (same bank for multiple threads)
    shared_data_conflict[0][tid] = tid;
    __syncthreads();
    
    printf("Thread %d: Good access pattern demonstrates no bank conflicts\n", tid);
}

float measure_kernel_time(void (*kernel_func)(), const char *name) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    kernel_func();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("%s: %.3f ms\n", name, time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

int main() {
    printf("=== Shared Memory Optimization Demo ===\n");
    printf("This example demonstrates shared memory usage and bank conflict avoidance.\n\n");
    
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Measure naive implementation
    auto naive_kernel = [=]() {
        matrix_multiply_naive<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    };
    float naive_time = measure_kernel_time(naive_kernel, "Naive Matrix Multiplication");
    
    // Measure shared memory implementation
    auto shared_kernel = [=]() {
        matrix_multiply_shared<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    };
    float shared_time = measure_kernel_time(shared_kernel, "Shared Memory Matrix Multiplication");
    
    printf("\nSpeedup with shared memory: %.2fx\n", naive_time / shared_time);
    
    // Demo bank conflicts
    printf("\n=== Bank Conflict Demo ===\n");
    shared_memory_bank_conflicts_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\nKey Learnings:\n");
    printf("- Shared memory is much faster than global memory (~100x)\n");
    printf("- Use tiling to reuse data loaded into shared memory\n");
    printf("- Avoid bank conflicts by padding arrays or careful indexing\n");
    printf("- Always synchronize threads when using shared memory\n");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}