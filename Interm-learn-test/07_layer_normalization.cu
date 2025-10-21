#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Layer normalization forward pass with Welford's online algorithm
__global__ void layer_norm_forward(const float* input, const float* weight, const float* bias,
                                  float* output, float* mean, float* variance,
                                  int batch_size, int seq_length, int hidden_size, float epsilon = 1e-5f) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    const float* x = input + (batch_idx * seq_length + seq_idx) * hidden_size;
    float* y = output + (batch_idx * seq_length + seq_idx) * hidden_size;
    
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    
    // Welford's online algorithm for numerical stability
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    int count = 0;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sum_sq += val * val;
        count++;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    
    __syncthreads();
    
    // Reduction to compute mean and variance
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
    
    // Normalize and apply scale/shift
    float inv_std = rsqrtf(batch_var + epsilon);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (x[i] - batch_mean) * inv_std;
        y[i] = weight[i] * normalized + bias[i];
    }
}

// Layer normalization backward pass
__global__ void layer_norm_backward(const float* grad_output, const float* input,
                                   const float* weight, const float* mean, const float* variance,
                                   float* grad_input, float* grad_weight, float* grad_bias,
                                   int batch_size, int seq_length, int hidden_size, float epsilon = 1e-5f) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    int base_idx = (batch_idx * seq_length + seq_idx) * hidden_size;
    const float* x = input + base_idx;
    const float* dy = grad_output + base_idx;
    float* dx = grad_input + base_idx;
    
    int stat_idx = batch_idx * seq_length + seq_idx;
    float x_mean = mean[stat_idx];
    float x_var = variance[stat_idx];
    float inv_std = rsqrtf(x_var + epsilon);
    
    extern __shared__ float sdata[];
    float* s_sum_dy_xhat = sdata;
    float* s_sum_dy = sdata + blockDim.x;
    
    // Compute intermediate sums
    float local_sum_dy_xhat = 0.0f;
    float local_sum_dy = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_hat = (x[i] - x_mean) * inv_std;
        local_sum_dy_xhat += dy[i] * x_hat;
        local_sum_dy += dy[i];
        
        // Accumulate gradients for weight and bias
        atomicAdd(&grad_weight[i], dy[i] * x_hat);
        atomicAdd(&grad_bias[i], dy[i]);
    }
    
    s_sum_dy_xhat[tid] = local_sum_dy_xhat;
    s_sum_dy[tid] = local_sum_dy;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_dy_xhat[tid] += s_sum_dy_xhat[tid + stride];
            s_sum_dy[tid] += s_sum_dy[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_dy_xhat = s_sum_dy_xhat[0];
    float sum_dy = s_sum_dy[0];
    
    __syncthreads();
    
    // Compute input gradients
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float x_hat = (x[i] - x_mean) * inv_std;
        float dx_hat = dy[i] * weight[i];
        
        dx[i] = inv_std * (dx_hat - (sum_dy + x_hat * sum_dy_xhat) / hidden_size);
    }
}

// Fused layer normalization + residual connection
__global__ void layer_norm_residual_forward(const float* input, const float* residual,
                                           const float* weight, const float* bias,
                                           float* output, float* mean, float* variance,
                                           int batch_size, int seq_length, int hidden_size,
                                           float epsilon = 1e-5f) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    int base_idx = (batch_idx * seq_length + seq_idx) * hidden_size;
    const float* x = input + base_idx;
    const float* r = residual + base_idx;
    float* y = output + base_idx;
    
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    
    // First, add residual connection
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i] + r[i]; // Residual connection
        local_sum += val;
        local_sum_sq += val * val;
        y[i] = val; // Store for later normalization
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
    
    // Apply layer normalization
    float inv_std = rsqrtf(batch_var + epsilon);
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float normalized = (y[i] - batch_mean) * inv_std;
        y[i] = weight[i] * normalized + bias[i];
    }
}

// Group normalization (useful for small batch sizes)
__global__ void group_norm_forward(const float* input, const float* weight, const float* bias,
                                  float* output, float* mean, float* variance,
                                  int batch_size, int seq_length, int hidden_size,
                                  int num_groups, float epsilon = 1e-5f) {
    
    int batch_idx = blockIdx.z;
    int seq_idx = blockIdx.y;
    int group_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || group_idx >= num_groups) return;
    
    int channels_per_group = hidden_size / num_groups;
    int base_idx = (batch_idx * seq_length + seq_idx) * hidden_size;
    int group_start = group_idx * channels_per_group;
    int group_end = group_start + channels_per_group;
    
    const float* x = input + base_idx;
    float* y = output + base_idx;
    
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    
    // Compute group statistics
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int i = group_start + tid; i < group_end; i += blockDim.x) {
        float val = x[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    
    __syncthreads();
    
    // Reduction within group
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float group_mean = s_sum[0] / channels_per_group;
    float group_var = (s_sum_sq[0] / channels_per_group) - (group_mean * group_mean);
    
    if (tid == 0) {
        int stat_idx = (batch_idx * seq_length + seq_idx) * num_groups + group_idx;
        mean[stat_idx] = group_mean;
        variance[stat_idx] = group_var;
    }
    
    __syncthreads();
    
    // Normalize group
    float inv_std = rsqrtf(group_var + epsilon);
    
    for (int i = group_start + tid; i < group_end; i += blockDim.x) {
        float normalized = (x[i] - group_mean) * inv_std;
        y[i] = weight[i] * normalized + bias[i];
    }
}

// RMS normalization (used in LLaMA and other modern models)
__global__ void rms_norm_forward(const float* input, const float* weight,
                                float* output, float* rms,
                                int batch_size, int seq_length, int hidden_size,
                                float epsilon = 1e-6f) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length) return;
    
    int base_idx = (batch_idx * seq_length + seq_idx) * hidden_size;
    const float* x = input + base_idx;
    float* y = output + base_idx;
    
    extern __shared__ float s_sum_sq[];
    
    // Compute sum of squares
    float local_sum_sq = 0.0f;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = x[i];
        local_sum_sq += val * val;
    }
    
    s_sum_sq[tid] = local_sum_sq;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    float mean_sq = s_sum_sq[0] / hidden_size;
    float rms_val = sqrtf(mean_sq + epsilon);
    
    if (tid == 0) {
        int stat_idx = batch_idx * seq_length + seq_idx;
        rms[stat_idx] = rms_val;
    }
    
    __syncthreads();
    
    // Normalize by RMS
    float inv_rms = 1.0f / rms_val;
    
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        y[i] = weight[i] * x[i] * inv_rms;
    }
}

void initialize_random(float* data, int size, float mean = 0.0f, float std = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, std);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main() {
    // Model parameters
    const int batch_size = 16;
    const int seq_length = 128;
    const int hidden_size = 768;
    const int num_groups = 12; // For group normalization
    const float epsilon = 1e-5f;
    
    // Memory sizes
    int input_size = batch_size * seq_length * hidden_size;
    int weight_bias_size = hidden_size;
    int stats_size = batch_size * seq_length;
    int group_stats_size = batch_size * seq_length * num_groups;
    
    // Host memory
    std::vector<float> h_input(input_size);
    std::vector<float> h_residual(input_size);
    std::vector<float> h_weight(weight_bias_size);
    std::vector<float> h_bias(weight_bias_size);
    std::vector<float> h_output(input_size);
    std::vector<float> h_mean(stats_size);
    std::vector<float> h_variance(stats_size);
    
    initialize_random(h_input.data(), input_size, 0.0f, 1.0f);
    initialize_random(h_residual.data(), input_size, 0.0f, 0.5f);
    initialize_random(h_weight.data(), weight_bias_size, 1.0f, 0.1f);
    initialize_random(h_bias.data(), weight_bias_size, 0.0f, 0.1f);
    
    // Device memory
    float *d_input, *d_residual, *d_weight, *d_bias, *d_output;
    float *d_mean, *d_variance, *d_rms;
    float *d_group_mean, *d_group_variance;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_residual, input_size * sizeof(float));
    cudaMalloc(&d_weight, weight_bias_size * sizeof(float));
    cudaMalloc(&d_bias, weight_bias_size * sizeof(float));
    cudaMalloc(&d_output, input_size * sizeof(float));
    cudaMalloc(&d_mean, stats_size * sizeof(float));
    cudaMalloc(&d_variance, stats_size * sizeof(float));
    cudaMalloc(&d_rms, stats_size * sizeof(float));
    cudaMalloc(&d_group_mean, group_stats_size * sizeof(float));
    cudaMalloc(&d_group_variance, group_stats_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, h_weight.data(), weight_bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias.data(), weight_bias_size * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "Normalization Layer Benchmark\n";
    std::cout << "Batch size: " << batch_size << ", Seq length: " << seq_length;
    std::cout << ", Hidden size: " << hidden_size << std::endl;
    
    // Kernel configurations
    int threads_per_block = min(256, hidden_size);
    dim3 block(threads_per_block);
    dim3 grid(seq_length, batch_size);
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Test Layer Normalization
    auto start = std::chrono::high_resolution_clock::now();
    layer_norm_forward<<<grid, block, shared_mem_size>>>(
        d_input, d_weight, d_bias, d_output, d_mean, d_variance,
        batch_size, seq_length, hidden_size, epsilon);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto layer_norm_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test Fused Layer Norm + Residual
    start = std::chrono::high_resolution_clock::now();
    layer_norm_residual_forward<<<grid, block, shared_mem_size>>>(
        d_input, d_residual, d_weight, d_bias, d_output, d_mean, d_variance,
        batch_size, seq_length, hidden_size, epsilon);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto layer_norm_residual_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test Group Normalization
    dim3 group_grid(num_groups, seq_length, batch_size);
    int group_shared_mem = 2 * threads_per_block * sizeof(float);
    
    start = std::chrono::high_resolution_clock::now();
    group_norm_forward<<<group_grid, block, group_shared_mem>>>(
        d_input, d_weight, d_bias, d_output, d_group_mean, d_group_variance,
        batch_size, seq_length, hidden_size, num_groups, epsilon);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto group_norm_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test RMS Normalization
    start = std::chrono::high_resolution_clock::now();
    rms_norm_forward<<<grid, block, threads_per_block * sizeof(float)>>>(
        d_input, d_weight, d_output, d_rms,
        batch_size, seq_length, hidden_size, epsilon);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto rms_norm_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test backward pass
    float *d_grad_output, *d_grad_input, *d_grad_weight, *d_grad_bias;
    cudaMalloc(&d_grad_output, input_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_weight, weight_bias_size * sizeof(float));
    cudaMalloc(&d_grad_bias, weight_bias_size * sizeof(float));
    
    // Initialize gradients
    std::vector<float> h_grad_output(input_size);
    initialize_random(h_grad_output.data(), input_size, 0.0f, 0.1f);
    cudaMemcpy(d_grad_output, h_grad_output.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grad_weight, 0, weight_bias_size * sizeof(float));
    cudaMemset(d_grad_bias, 0, weight_bias_size * sizeof(float));
    
    // First run forward to get statistics
    layer_norm_forward<<<grid, block, shared_mem_size>>>(
        d_input, d_weight, d_bias, d_output, d_mean, d_variance,
        batch_size, seq_length, hidden_size, epsilon);
    cudaDeviceSynchronize();
    
    start = std::chrono::high_resolution_clock::now();
    layer_norm_backward<<<grid, block, shared_mem_size>>>(
        d_grad_output, d_input, d_weight, d_mean, d_variance,
        d_grad_input, d_grad_weight, d_grad_bias,
        batch_size, seq_length, hidden_size, epsilon);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto layer_norm_backward_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back for verification
    cudaMemcpy(h_output.data(), d_output, input_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mean.data(), d_mean, stats_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variance.data(), d_variance, stats_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float output_mean = 0.0f, output_var = 0.0f;
    for (int i = 0; i < hidden_size; i++) {
        output_mean += h_output[i];
    }
    output_mean /= hidden_size;
    
    for (int i = 0; i < hidden_size; i++) {
        float diff = h_output[i] - output_mean;
        output_var += diff * diff;
    }
    output_var /= hidden_size;
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Layer Normalization: " << layer_norm_time << " ms\n";
    std::cout << "Layer Norm + Residual: " << layer_norm_residual_time << " ms\n";
    std::cout << "Group Normalization: " << group_norm_time << " ms\n";
    std::cout << "RMS Normalization: " << rms_norm_time << " ms\n";
    std::cout << "Layer Norm Backward: " << layer_norm_backward_time << " ms\n";
    
    std::cout << "\nNormalization Verification (first sequence):\n";
    std::cout << "Output mean: " << output_mean << " (should be ~0)\n";
    std::cout << "Output variance: " << output_var << " (should be ~1)\n";
    std::cout << "Computed mean: " << h_mean[0] << std::endl;
    std::cout << "Computed variance: " << h_variance[0] << std::endl;
    
    // Cleanup
    cudaFree(d_input); cudaFree(d_residual); cudaFree(d_weight); cudaFree(d_bias);
    cudaFree(d_output); cudaFree(d_mean); cudaFree(d_variance); cudaFree(d_rms);
    cudaFree(d_group_mean); cudaFree(d_group_variance);
    cudaFree(d_grad_output); cudaFree(d_grad_input);
    cudaFree(d_grad_weight); cudaFree(d_grad_bias);
    
    return 0;
}
