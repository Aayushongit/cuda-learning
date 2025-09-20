#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

// Numerically stable softmax with shared memory reduction
__global__ void softmax_stable(const float* input, float* output, int batch_size, int features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * features;
    float* y = output + batch_idx * features;
    
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;
    
    // Find maximum value for numerical stability
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }
    s_max[tid] = local_max;
    
    __syncthreads();
    
    // Reduction to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_val = s_max[0];
    __syncthreads();
    
    // Compute exp(x - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        float exp_val = expf(x[i] - max_val);
        y[i] = exp_val;
        local_sum += exp_val;
    }
    s_sum[tid] = local_sum;
    
    __syncthreads();
    
    // Reduction to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float sum_val = s_sum[0];
    __syncthreads();
    
    // Normalize
    for (int i = tid; i < features; i += blockDim.x) {
        y[i] = y[i] / sum_val;
    }
}

// Softmax backward pass for gradient computation
__global__ void softmax_backward(const float* grad_output, const float* softmax_output,
                                float* grad_input, int batch_size, int features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* grad_out = grad_output + batch_idx * features;
    const float* softmax_out = softmax_output + batch_idx * features;
    float* grad_in = grad_input + batch_idx * features;
    
    extern __shared__ float s_dot_product[];
    
    // Compute dot product of grad_output and softmax_output
    float local_dot = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        local_dot += grad_out[i] * softmax_out[i];
    }
    s_dot_product[tid] = local_dot;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_dot_product[tid] += s_dot_product[tid + stride];
        }
        __syncthreads();
    }
    
    float dot_product = s_dot_product[0];
    __syncthreads();
    
    // Compute gradient: softmax_output * (grad_output - dot_product)
    for (int i = tid; i < features; i += blockDim.x) {
        grad_in[i] = softmax_out[i] * (grad_out[i] - dot_product);
    }
}

// Log-softmax for numerical stability in loss computation
__global__ void log_softmax(const float* input, float* output, int batch_size, int features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * features;
    float* y = output + batch_idx * features;
    
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;
    
    // Find maximum
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
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
    
    // Compute sum of exp(x - max)
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        local_sum += expf(x[i] - max_val);
    }
    s_sum[tid] = local_sum;
    
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }
    
    float log_sum_exp = logf(s_sum[0]);
    __syncthreads();
    
    // Compute log-softmax: x - max - log(sum_exp)
    for (int i = tid; i < features; i += blockDim.x) {
        y[i] = x[i] - max_val - log_sum_exp;
    }
}

// Softmax with temperature for LLM sampling
__global__ void softmax_temperature(const float* input, float* output, 
                                   int batch_size, int features, float temperature) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    const float* x = input + batch_idx * features;
    float* y = output + batch_idx * features;
    
    extern __shared__ float sdata[];
    float* s_max = sdata;
    float* s_sum = sdata + blockDim.x;
    
    // Find maximum of x/temperature
    float local_max = -INFINITY;
    for (int i = tid; i < features; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i] / temperature);
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
    
    // Compute exp((x/temperature) - max) and sum
    float local_sum = 0.0f;
    for (int i = tid; i < features; i += blockDim.x) {
        float exp_val = expf((x[i] / temperature) - max_val);
        y[i] = exp_val;
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
    
    // Normalize
    for (int i = tid; i < features; i += blockDim.x) {
        y[i] = y[i] / sum_val;
    }
}

void initialize_random(float* data, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main() {
    const int batch_size = 32;
    const int features = 1000;
    const float temperature = 0.8f;
    
    // Host memory
    std::vector<float> h_input(batch_size * features);
    std::vector<float> h_output(batch_size * features);
    std::vector<float> h_grad_output(batch_size * features);
    std::vector<float> h_grad_input(batch_size * features);
    
    initialize_random(h_input.data(), batch_size * features);
    initialize_random(h_grad_output.data(), batch_size * features);
    
    // Device memory
    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, batch_size * features * sizeof(float));
    cudaMalloc(&d_output, batch_size * features * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * features * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * features * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), batch_size * features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output.data(), batch_size * features * sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel configuration
    int threads_per_block = 256;
    int shared_mem_size = 2 * threads_per_block * sizeof(float);
    
    // Test standard softmax
    auto start = std::chrono::high_resolution_clock::now();
    softmax_stable<<<batch_size, threads_per_block, shared_mem_size>>>(d_input, d_output, batch_size, features);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto softmax_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test softmax backward
    start = std::chrono::high_resolution_clock::now();
    softmax_backward<<<batch_size, threads_per_block, threads_per_block * sizeof(float)>>>(
        d_grad_output, d_output, d_grad_input, batch_size, features);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto backward_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test log-softmax
    start = std::chrono::high_resolution_clock::now();
    log_softmax<<<batch_size, threads_per_block, shared_mem_size>>>(d_input, d_output, batch_size, features);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto log_softmax_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test temperature softmax
    start = std::chrono::high_resolution_clock::now();
    softmax_temperature<<<batch_size, threads_per_block, shared_mem_size>>>(
        d_input, d_output, batch_size, features, temperature);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto temp_softmax_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back
    cudaMemcpy(h_output.data(), d_output, batch_size * features * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Verify softmax properties (sum should be ~1.0)
    float sum = 0.0f;
    for (int i = 0; i < features; i++) {
        sum += h_output[i];
    }
    
    std::cout << "Softmax Performance Results:\n";
    std::cout << "Standard Softmax: " << softmax_time << " ms\n";
    std::cout << "Softmax Backward: " << backward_time << " ms\n";
    std::cout << "Log-Softmax: " << log_softmax_time << " ms\n";
    std::cout << "Temperature Softmax: " << temp_softmax_time << " ms\n";
    std::cout << "First batch sum verification: " << sum << " (should be ~1.0)\n";
    std::cout << "First element: " << h_output[0] << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    
    return 0;
}