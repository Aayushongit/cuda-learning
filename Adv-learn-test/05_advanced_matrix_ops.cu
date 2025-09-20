#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define TILE_SIZE 32
#define N 1024

// Optimized matrix multiplication with multiple techniques
__global__ void matrix_multiply_optimized(float *A, float *B, float *C, int n) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    // Process tiles
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Coalesced loading into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        shared_A[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? 
            A[row * n + a_col] : 0.0f;
        shared_B[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? 
            B[b_row * n + col] : 0.0f;
        
        __syncthreads();
        
        // Compute using shared memory
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < n && col < n) {
        C[row * n + col] = sum;
    }
}

// Matrix transpose with shared memory optimization
__global__ void matrix_transpose_optimized(float *input, float *output, int n) {
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Read from global memory (coalesced)
    if (x < n && y < n) {
        tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    }
    
    __syncthreads();
    
    // Write to global memory (coalesced for transposed matrix)
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < n && y < n) {
        output[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Vectorized matrix operations using float4
__global__ void matrix_add_vectorized(float4 *A, float4 *B, float4 *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * n / 4; // Since we're using float4
    
    if (idx < total_elements) {
        float4 a = A[idx];
        float4 b = B[idx];
        
        C[idx] = make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
}

// Strided matrix multiplication (useful for batch operations)
__global__ void matrix_multiply_strided(float *A, float *B, float *C, 
                                       int n, int batch_size, 
                                       int stride_A, int stride_B, int stride_C) {
    int batch_id = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_id < batch_size && row < n && col < n) {
        float *A_batch = A + batch_id * stride_A;
        float *B_batch = B + batch_id * stride_B;
        float *C_batch = C + batch_id * stride_C;
        
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A_batch[row * n + k] * B_batch[k * n + col];
        }
        C_batch[row * n + col] = sum;
    }
}

// Matrix multiplication with register tiling
__global__ void matrix_multiply_register_tiling(float *A, float *B, float *C, int n) {
    const int REG_TILE = 4;
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Register arrays for accumulation
    float sum[REG_TILE][REG_TILE] = {0};
    
    for (int tile = 0; tile < (n + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Load tile into shared memory
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        
        shared_A[threadIdx.y][threadIdx.x] = (row < n && a_col < n) ? 
            A[row * n + a_col] : 0.0f;
        shared_B[threadIdx.y][threadIdx.x] = (b_row < n && col < n) ? 
            B[b_row * n + col] : 0.0f;
        
        __syncthreads();
        
        // Compute using register tiling
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < REG_TILE && (threadIdx.y * REG_TILE + i) < TILE_SIZE; i++) {
                for (int j = 0; j < REG_TILE && (threadIdx.x * REG_TILE + j) < TILE_SIZE; j++) {
                    sum[i][j] += shared_A[threadIdx.y * REG_TILE + i][k] * 
                                shared_B[k][threadIdx.x * REG_TILE + j];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back
    for (int i = 0; i < REG_TILE; i++) {
        for (int j = 0; j < REG_TILE; j++) {
            int write_row = row * REG_TILE + i;
            int write_col = col * REG_TILE + j;
            if (write_row < n && write_col < n) {
                C[write_row * n + write_col] = sum[i][j];
            }
        }
    }
}

float benchmark_kernel(const char *name, void (*setup_and_run)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    setup_and_run();
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
    printf("=== Advanced Matrix Operations Optimization Demo ===\n");
    printf("This example demonstrates various matrix operation optimizations.\n\n");
    
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C, *d_temp;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_temp, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // 1. Optimized Matrix Multiplication
    auto matmul_test = [&]() {
        matrix_multiply_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
        cudaDeviceSynchronize();
    };
    benchmark_kernel("Optimized Matrix Multiplication", matmul_test);
    
    // 2. Matrix Transpose
    auto transpose_test = [&]() {
        matrix_transpose_optimized<<<gridDim, blockDim>>>(d_A, d_temp, N);
        cudaDeviceSynchronize();
    };
    benchmark_kernel("Optimized Matrix Transpose", transpose_test);
    
    // 3. Vectorized Operations
    auto vectorized_test = [&]() {
        int vec_elements = (N * N + 3) / 4; // Round up for float4
        int vec_blocks = (vec_elements + 256 - 1) / 256;
        matrix_add_vectorized<<<vec_blocks, 256>>>((float4*)d_A, (float4*)d_B, (float4*)d_C, N);
        cudaDeviceSynchronize();
    };
    benchmark_kernel("Vectorized Matrix Addition", vectorized_test);
    
    // 4. cuBLAS comparison
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    auto cublas_test = [&]() {
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, 
                   &alpha, d_A, N, d_B, N, &beta, d_C, N);
        cudaDeviceSynchronize();
    };
    benchmark_kernel("cuBLAS SGEMM", cublas_test);
    
    cublasDestroy(handle);
    
    printf("\nKey Learnings:\n");
    printf("- Shared memory tiling reduces global memory accesses\n");
    printf("- Bank conflict avoidance improves shared memory performance\n");
    printf("- Vectorized operations (float4) can improve memory throughput\n");
    printf("- Register tiling can further optimize computation\n");
    printf("- cuBLAS provides highly optimized implementations\n");
    printf("- Consider strided operations for batch processing\n");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_temp);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}