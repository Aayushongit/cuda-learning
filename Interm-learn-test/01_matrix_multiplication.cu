#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

// Optimized matrix multiplication kernel with shared memory
__global__ void matmul_shared(const float* A, const float* B, float* C,
                             int M, int N, int K, int tile_size = 16) {
    
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + tile_size - 1) / tile_size; t++) {
        // Load tiles into shared memory
        if (row < M && t * tile_size + tx < K)
            As[ty][tx] = A[row * K + t * tile_size + tx];
        else
            As[ty][tx] = 0.0f;
            
        if (col < N && t * tile_size + ty < K)
            Bs[ty][tx] = B[(t * tile_size + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < tile_size; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Fused matrix multiplication with bias and ReLU activation
__global__ void matmul_bias_relu(const float* A, const float* B, const float* bias,
                                float* C, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idx < N && idy < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[idy * K + k] * B[k * N + idx];
        }
        sum += bias[idx]; // Add bias
        C[idy * N + idx] = fmaxf(0.0f, sum); // ReLU activation
    }
}

// Batch matrix multiplication for neural networks
__global__ void batch_matmul(const float* A, const float* B, float* C,
                            int batch_size, int M, int N, int K) {
    int batch_idx = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && row < M && col < N) {
        float sum = 0.0f;
        
        int A_offset = batch_idx * M * K;
        int B_offset = batch_idx * K * N;
        int C_offset = batch_idx * M * N;
        
        for (int k = 0; k < K; k++) {
            sum += A[A_offset + row * K + k] * B[B_offset + k * N + col];
        }
        
        C[C_offset + row * N + col] = sum;
    }
}

void initialize_matrix(float* matrix, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        matrix[i] = dis(gen);
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int batch_size = 4;
    
    // Host memory allocation
    std::vector<float> h_A(M * K), h_B(K * N), h_C(M * N);
    std::vector<float> h_bias(N);
    std::vector<float> h_batch_A(batch_size * M * K);
    std::vector<float> h_batch_B(batch_size * K * N);
    std::vector<float> h_batch_C(batch_size * M * N);
    
    // Initialize matrices
    initialize_matrix(h_A.data(), M * K);
    initialize_matrix(h_B.data(), K * N);
    initialize_matrix(h_bias.data(), N);
    initialize_matrix(h_batch_A.data(), batch_size * M * K);
    initialize_matrix(h_batch_B.data(), batch_size * K * N);
    
    // Device memory allocation
    float *d_A, *d_B, *d_C, *d_bias;
    float *d_batch_A, *d_batch_B, *d_batch_C;
    
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_bias, N * sizeof(float));
    cudaMalloc(&d_batch_A, batch_size * M * K * sizeof(float));
    cudaMalloc(&d_batch_B, batch_size * K * N * sizeof(float));
    cudaMalloc(&d_batch_C, batch_size * M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_A, h_batch_A.data(), batch_size * M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_batch_B, h_batch_B.data(), batch_size * K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel configurations
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    dim3 batch_grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y, batch_size);
    
    // Test shared memory matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    matmul_shared<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto shared_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test fused matrix multiplication with bias and ReLU
    start = std::chrono::high_resolution_clock::now();
    matmul_bias_relu<<<grid, block>>>(d_A, d_B, d_bias, d_C, M, N, K);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto fused_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test batch matrix multiplication
    start = std::chrono::high_resolution_clock::now();
    batch_matmul<<<batch_grid, block>>>(d_batch_A, d_batch_B, d_batch_C, batch_size, M, N, K);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto batch_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Matrix Multiplication Performance:\n";
    std::cout << "Shared Memory GEMM: " << shared_time << " ms\n";
    std::cout << "Fused Bias+ReLU GEMM: " << fused_time << " ms\n";
    std::cout << "Batch GEMM (" << batch_size << " batches): " << batch_time << " ms\n";
    std::cout << "First element of result: " << h_C[0] << std::endl;
    
    // Cleanup
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_bias);
    cudaFree(d_batch_A); cudaFree(d_batch_B); cudaFree(d_batch_C);
    
    return 0;
}
