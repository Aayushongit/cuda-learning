#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Mixed precision matrix multiplication (FP16 with FP32 accumulation)
__global__ void mixed_precision_matmul(const __half* A, const __half* B, float* C,
                                      int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        
        for (int k = 0; k < K; k++) {
            float a_val = __half2float(A[row * K + k]);
            float b_val = __half2float(B[k * N + col]);
            sum += a_val * b_val;
        }
        
        C[row * N + col] = sum;
    }
}

// Tensor Core mixed precision GEMM simulation
__global__ void tensorcore_mixed_precision_matmul(const __half* A, const __half* B, float* C,
                                                 int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Use half2 for vectorized operations
        float sum = 0.0f;
        
        // Process pairs of elements for better throughput
        for (int k = 0; k < K; k += 2) {
            if (k + 1 < K) {
                __half2 a_vec = *(__half2*)&A[row * K + k];
                __half2 b_vec = make_half2(B[k * N + col], B[(k + 1) * N + col]);
                
                float2 a_float = __half22float2(a_vec);
                float2 b_float = __half22float2(b_vec);
                
                sum += a_float.x * b_float.x + a_float.y * b_float.y;
            } else {
                // Handle odd K
                float a_val = __half2float(A[row * K + k]);
                float b_val = __half2float(B[k * N + col]);
                sum += a_val * b_val;
            }
        }
        
        C[row * N + col] = sum;
    }
}

// Convert FP32 to FP16 with loss scaling
__global__ void fp32_to_fp16_scaled(const float* input, __half* output, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float scaled_val = input[idx] * scale;
        output[idx] = __float2half(scaled_val);
    }
}

// Convert FP16 to FP32 with loss unscaling
__global__ void fp16_to_fp32_unscaled(const __half* input, float* output, float inv_scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float val = __half2float(input[idx]);
        output[idx] = val * inv_scale;
    }
}

// Mixed precision layer normalization
__global__ void mixed_precision_layer_norm(const __half* input, const float* weight, const float* bias,
                                          __half* output, float* mean, float* variance,
                                          int batch_size, int seq_length, int hidden_size,
                                          float epsilon = 1e-5f) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    const __half* x = input + (batch_idx * seq_length + seq_idx) * hidden_size;
    __half* y = output + (batch_idx * seq_length + seq_idx) * hidden_size;
    
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    
    // Compute statistics in FP32 for numerical stability
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __half2float(x[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float batch_mean = s_sum[0] / hidden_size;
    float batch_var = (s_sum_sq[0] / hidden_size) - (batch_mean * batch_mean);
    
    if (tid == 0) {
        int stat_idx = batch_idx * seq_length + seq_idx;
        mean[stat_idx] = batch_mean;
        variance[stat_idx] = batch_var;
    }
    
    __syncthreads();
    
    // Normalize and apply scale/shift (compute in FP32, store as FP16)
    float inv_std = rsqrtf(batch_var + epsilon);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_val = __half2float(x[i]);
        float normalized = (x_val - batch_mean) * inv_std;
        float result = weight[i] * normalized + bias[i];
        y[i] = __float2half(result);
    }
}

// Loss scaling for mixed precision training
__global__ void dynamic_loss_scaling(float* loss, float* loss_scale, float* growth_factor,
                                    int* growth_interval, int* growth_count,
                                    bool* overflow_detected, float max_loss_scale) {
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (*overflow_detected) {
            // Reduce loss scale on overflow
            *loss_scale = fmaxf(*loss_scale / 2.0f, 1.0f);
            *growth_count = 0;
            *overflow_detected = false;
        } else {
            // Increase loss scale gradually
            (*growth_count)++;
            if (*growth_count >= *growth_interval) {
                *loss_scale = fminf(*loss_scale * *growth_factor, max_loss_scale);
                *growth_count = 0;
            }
        }
        
        // Apply loss scaling
        *loss = *loss * (*loss_scale);
    }
}

// Gradient overflow detection
__global__ void detect_gradient_overflow(const float* gradients, bool* overflow_detected,
                                        int size, float threshold = 65504.0f) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad_val = gradients[idx];
        if (!isfinite(grad_val) || fabsf(grad_val) > threshold) {
            *overflow_detected = true;
        }
    }
}

// Mixed precision GELU activation
__global__ void mixed_precision_gelu(const __half* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float x = __half2float(input[idx]);
        
        // GELU approximation
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        float gelu_val = 0.5f * x * (1.0f + tanhf(inner));
        
        output[idx] = __float2half(gelu_val);
    }
}

// Mixed precision softmax
__global__ void mixed_precision_softmax(const __half* input, __half* output,
                                       int batch_size, int seq_length, int vocab_size) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    const __half* x = input + (batch_idx * seq_length + seq_idx) * vocab_size;
    __half* y = output + (batch_idx * seq_length + seq_idx) * vocab_size;
    
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;
    
    // Find maximum in FP32 for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        local_max = fmaxf(local_max, __half2float(x[i]));
    }
    s_max[tid] = local_max;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = s_max[0];
    __syncthreads();
    
    // Compute sum of exponentials in FP32
    float local_sum = 0.0f;
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float exp_val = expf(__half2float(x[i]) - max_val);
        local_sum += exp_val;
    }
    s_sum[tid] = local_sum;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = s_sum[0];
    __syncthreads();
    
    // Normalize and convert back to FP16
    for (int i = tid; i < vocab_size; i += blockDim.x) {
        float exp_val = expf(__half2float(x[i]) - max_val);
        float prob = exp_val / sum_val;
        y[i] = __float2half(prob);
    }
}

// Auto-cast operations for mixed precision
__global__ void autocast_operations(const float* fp32_input, __half* fp16_weights,
                                   float* fp32_output, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Simulate autocast behavior
        float input_val = fp32_input[idx];
        float weight_val = __half2float(fp16_weights[idx]);
        
        // Perform operation in appropriate precision
        float result = input_val * weight_val; // This would be autocasted to appropriate precision
        
        fp32_output[idx] = result;
    }
}

// Memory bandwidth test for different precisions
__global__ void memory_bandwidth_test_fp16(const __half* input, __half* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx];
    }
}

__global__ void memory_bandwidth_test_fp32(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        output[idx] = input[idx];
    }
}

void initialize_random_fp32(float* data, int size, float mean = 0.0f, float std = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, std);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

void fp32_to_fp16_cpu(const float* fp32_data, __half* fp16_data, int size) {
    for (int i = 0; i < size; i++) {
        fp16_data[i] = __float2half(fp32_data[i]);
    }
}

int main() {
    // Model parameters
    const int batch_size = 32;
    const int seq_length = 512;
    const int hidden_size = 768;
    const int vocab_size = 50000;
    const int M = 1024, N = 1024, K = 1024;
    
    // Mixed precision parameters
    const float loss_scale = 1024.0f;
    const float growth_factor = 2.0f;
    const int growth_interval = 2000;
    const float max_loss_scale = 65536.0f;
    
    std::cout << "Mixed Precision Training Benchmark\n";
    std::cout << "Matrix size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Sequence processing: " << batch_size << "x" << seq_length << "x" << hidden_size << std::endl;
    
    // Memory sizes
    int matrix_size = M * K; // A matrix
    int weight_size = K * N; // B matrix
    int output_size = M * N; // C matrix
    int sequence_size = batch_size * seq_length * hidden_size;
    int softmax_size = batch_size * seq_length * vocab_size;
    
    // Host memory
    std::vector<float> h_fp32_A(matrix_size), h_fp32_B(weight_size), h_fp32_C(output_size);
    std::vector<__half> h_fp16_A(matrix_size), h_fp16_B(weight_size);
    std::vector<float> h_sequence_fp32(sequence_size);
    std::vector<__half> h_sequence_fp16(sequence_size);
    
    initialize_random_fp32(h_fp32_A.data(), matrix_size, 0.0f, 1.0f);
    initialize_random_fp32(h_fp32_B.data(), weight_size, 0.0f, 1.0f);
    initialize_random_fp32(h_sequence_fp32.data(), sequence_size, 0.0f, 1.0f);
    
    // Convert to FP16
    fp32_to_fp16_cpu(h_fp32_A.data(), h_fp16_A.data(), matrix_size);
    fp32_to_fp16_cpu(h_fp32_B.data(), h_fp16_B.data(), weight_size);
    fp32_to_fp16_cpu(h_sequence_fp32.data(), h_sequence_fp16.data(), sequence_size);
    
    // Device memory
    float *d_fp32_A, *d_fp32_B, *d_fp32_C;
    __half *d_fp16_A, *d_fp16_B;
    float *d_sequence_fp32, *d_layer_norm_weights, *d_layer_norm_bias;
    __half *d_sequence_fp16, *d_sequence_output;
    float *d_mean, *d_variance;
    
    cudaMalloc(&d_fp32_A, matrix_size * sizeof(float));
    cudaMalloc(&d_fp32_B, weight_size * sizeof(float));
    cudaMalloc(&d_fp32_C, output_size * sizeof(float));
    cudaMalloc(&d_fp16_A, matrix_size * sizeof(__half));
    cudaMalloc(&d_fp16_B, weight_size * sizeof(__half));
    
    cudaMalloc(&d_sequence_fp32, sequence_size * sizeof(float));
    cudaMalloc(&d_sequence_fp16, sequence_size * sizeof(__half));
    cudaMalloc(&d_sequence_output, sequence_size * sizeof(__half));
    cudaMalloc(&d_layer_norm_weights, hidden_size * sizeof(float));
    cudaMalloc(&d_layer_norm_bias, hidden_size * sizeof(float));
    cudaMalloc(&d_mean, batch_size * seq_length * sizeof(float));
    cudaMalloc(&d_variance, batch_size * seq_length * sizeof(float));
    
    // Initialize layer norm parameters
    std::vector<float> h_weights(hidden_size, 1.0f), h_bias(hidden_size, 0.0f);
    cudaMemcpy(d_layer_norm_weights, h_weights.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_norm_bias, h_bias.data(), hidden_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_fp32_A, h_fp32_A.data(), matrix_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp32_B, h_fp32_B.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp16_A, h_fp16_A.data(), matrix_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fp16_B, h_fp16_B.data(), weight_size * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sequence_fp16, h_sequence_fp16.data(), sequence_size * sizeof(__half), cudaMemcpyHostToDevice);
    
    // Kernel configurations
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    // Benchmark FP32 vs Mixed Precision GEMM
    auto start = std::chrono::high_resolution_clock::now();
    
    // Simulate FP32 GEMM (using existing FP32 data)
    // This would normally use cuBLAS SGEMM
    cudaDeviceSynchronize();
    auto fp32_gemm_end = std::chrono::high_resolution_clock::now();
    
    // Mixed precision GEMM
    start = std::chrono::high_resolution_clock::now();
    mixed_precision_matmul<<<grid, block>>>(d_fp16_A, d_fp16_B, d_fp32_C, M, N, K);
    cudaDeviceSynchronize();
    auto mixed_gemm_end = std::chrono::high_resolution_clock::now();
    
    auto mixed_gemm_time = std::chrono::duration<float, std::milli>(mixed_gemm_end - start).count();
    
    // Tensor Core simulation
    start = std::chrono::high_resolution_clock::now();
    tensorcore_mixed_precision_matmul<<<grid, block>>>(d_fp16_A, d_fp16_B, d_fp32_C, M, N, K);
    cudaDeviceSynchronize();
    auto tensorcore_end = std::chrono::high_resolution_clock::now();
    auto tensorcore_time = std::chrono::duration<float, std::milli>(tensorcore_end - start).count();
    
    // Mixed precision layer normalization
    dim3 ln_block(min(256, hidden_size));
    dim3 ln_grid(seq_length, batch_size);
    int shared_mem_size = 2 * ln_block.x * sizeof(float);
    
    start = std::chrono::high_resolution_clock::now();
    mixed_precision_layer_norm<<<ln_grid, ln_block, shared_mem_size>>>(
        d_sequence_fp16, d_layer_norm_weights, d_layer_norm_bias, d_sequence_output,
        d_mean, d_variance, batch_size, seq_length, hidden_size);
    cudaDeviceSynchronize();
    auto ln_end = std::chrono::high_resolution_clock::now();
    auto ln_time = std::chrono::duration<float, std::milli>(ln_end - start).count();
    
    // Mixed precision GELU
    dim3 gelu_block(256);
    dim3 gelu_grid((sequence_size + gelu_block.x - 1) / gelu_block.x);
    
    start = std::chrono::high_resolution_clock::now();
    mixed_precision_gelu<<<gelu_grid, gelu_block>>>(d_sequence_fp16, d_sequence_output, sequence_size);
    cudaDeviceSynchronize();
    auto gelu_end = std::chrono::high_resolution_clock::now();
    auto gelu_time = std::chrono::duration<float, std::milli>(gelu_end - start).count();
    
    // Loss scaling test
    float *d_loss, *d_loss_scale, *d_growth_factor;
    int *d_growth_interval, *d_growth_count;
    bool *d_overflow_detected;
    
    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_loss_scale, sizeof(float));
    cudaMalloc(&d_growth_factor, sizeof(float));
    cudaMalloc(&d_growth_interval, sizeof(int));
    cudaMalloc(&d_growth_count, sizeof(int));
    cudaMalloc(&d_overflow_detected, sizeof(bool));
    
    float h_loss = 1.5f;
    bool h_overflow = false;
    int h_growth_count = 0;
    
    cudaMemcpy(d_loss, &h_loss, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_loss_scale, &loss_scale, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_growth_factor, &growth_factor, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_growth_interval, &growth_interval, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_growth_count, &h_growth_count, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_overflow_detected, &h_overflow, sizeof(bool), cudaMemcpyHostToDevice);
    
    start = std::chrono::high_resolution_clock::now();
    dynamic_loss_scaling<<<1, 1>>>(d_loss, d_loss_scale, d_growth_factor,
                                   d_growth_interval, d_growth_count, d_overflow_detected, max_loss_scale);
    cudaDeviceSynchronize();
    auto scaling_end = std::chrono::high_resolution_clock::now();
    auto scaling_time = std::chrono::duration<float, std::milli>(scaling_end - start).count();
    
    // Memory bandwidth comparison
    float *d_fp32_mem_test;
    __half *d_fp16_mem_test;
    cudaMalloc(&d_fp32_mem_test, sequence_size * sizeof(float));
    cudaMalloc(&d_fp16_mem_test, sequence_size * sizeof(__half));
    
    start = std::chrono::high_resolution_clock::now();
    memory_bandwidth_test_fp32<<<gelu_grid, gelu_block>>>(d_sequence_fp32, d_fp32_mem_test, sequence_size);
    cudaDeviceSynchronize();
    auto fp32_mem_end = std::chrono::high_resolution_clock::now();
    auto fp32_mem_time = std::chrono::duration<float, std::milli>(fp32_mem_end - start).count();
    
    start = std::chrono::high_resolution_clock::now();
    memory_bandwidth_test_fp16<<<gelu_grid, gelu_block>>>(d_sequence_fp16, d_fp16_mem_test, sequence_size);
    cudaDeviceSynchronize();
    auto fp16_mem_end = std::chrono::high_resolution_clock::now();
    auto fp16_mem_time = std::chrono::duration<float, std::milli>(fp16_mem_end - start).count();
    
    // Copy results back for verification
    cudaMemcpy(h_fp32_C.data(), d_fp32_C, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float h_final_loss_scale;
    cudaMemcpy(&h_final_loss_scale, d_loss_scale, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Mixed Precision GEMM: " << mixed_gemm_time << " ms\n";
    std::cout << "Tensor Core GEMM (simulated): " << tensorcore_time << " ms\n";
    std::cout << "Mixed Precision Layer Norm: " << ln_time << " ms\n";
    std::cout << "Mixed Precision GELU: " << gelu_time << " ms\n";
    std::cout << "Loss Scaling: " << scaling_time << " ms\n";
    
    std::cout << "\nMemory Bandwidth Comparison:\n";
    std::cout << "FP32 Memory Copy: " << fp32_mem_time << " ms\n";
    std::cout << "FP16 Memory Copy: " << fp16_mem_time << " ms\n";
    
    // Calculate memory bandwidth
    long long fp32_bytes = sequence_size * sizeof(float) * 2; // Read + Write
    long long fp16_bytes = sequence_size * sizeof(__half) * 2;
    
    float fp32_bandwidth = (fp32_bytes / (1024.0f * 1024.0f * 1024.0f)) / (fp32_mem_time / 1000.0f);
    float fp16_bandwidth = (fp16_bytes / (1024.0f * 1024.0f * 1024.0f)) / (fp16_mem_time / 1000.0f);
    
    std::cout << "FP32 Bandwidth: " << fp32_bandwidth << " GB/s\n";
    std::cout << "FP16 Bandwidth: " << fp16_bandwidth << " GB/s\n";
    std::cout << "Bandwidth Improvement: " << (fp16_bandwidth / fp32_bandwidth) << "x\n";
    
    std::cout << "\nMixed Precision Benefits:\n";
    std::cout << "Memory Usage Reduction: ~50% (FP16 vs FP32)\n";
    std::cout << "Potential Speed Improvement: " << (fp32_mem_time / mixed_gemm_time) << "x (operation dependent)\n";
    std::cout << "Final Loss Scale: " << h_final_loss_scale << std::endl;
    
    // Verify numerical results
    float result_mean = 0.0f;
    for (int i = 0; i < 100; i++) { // Check first 100 elements
        result_mean += h_fp32_C[i];
    }
    result_mean /= 100.0f;
    
    std::cout << "\nNumerical Verification:\n";
    std::cout << "Mean of first 100 GEMM results: " << result_mean << std::endl;
    std::cout << "First few GEMM results: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_fp32_C[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_fp32_A); cudaFree(d_fp32_B); cudaFree(d_fp32_C);
    cudaFree(d_fp16_A); cudaFree(d_fp16_B);
    cudaFree(d_sequence_fp32); cudaFree(d_sequence_fp16); cudaFree(d_sequence_output);
    cudaFree(d_layer_norm_weights); cudaFree(d_layer_norm_bias);
    cudaFree(d_mean); cudaFree(d_variance);
    cudaFree(d_loss); cudaFree(d_loss_scale); cudaFree(d_growth_factor);
    cudaFree(d_growth_interval); cudaFree(d_growth_count); cudaFree(d_overflow_detected);
    cudaFree(d_fp32_mem_test); cudaFree(d_fp16_mem_test);
    
    return 0;
}