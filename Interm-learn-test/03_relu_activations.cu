#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Basic ReLU activation function
__global__ void relu_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

// ReLU backward pass (derivative)
__global__ void relu_backward(const float* grad_output, const float* input, 
                             float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

// Leaky ReLU with configurable slope
__global__ void leaky_relu_forward(const float* input, float* output, 
                                  int size, float negative_slope = 0.01f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : negative_slope * input[idx];
    }
}

// Leaky ReLU backward pass
__global__ void leaky_relu_backward(const float* grad_output, const float* input,
                                   float* grad_input, int size, float negative_slope = 0.01f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : negative_slope * grad_output[idx];
    }
}

// ELU (Exponential Linear Unit) activation
__global__ void elu_forward(const float* input, float* output, int size, float alpha = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (input[idx] > 0.0f) ? input[idx] : alpha * (expf(input[idx]) - 1.0f);
    }
}

// ELU backward pass
__global__ void elu_backward(const float* grad_output, const float* input, const float* output,
                            float* grad_input, int size, float alpha = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (input[idx] > 0.0f) {
            grad_input[idx] = grad_output[idx];
        } else {
            grad_input[idx] = grad_output[idx] * (output[idx] + alpha);
        }
    }
}

// GELU (Gaussian Error Linear Unit) - approximation for faster computation
__global__ void gelu_forward_approx(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
        float x3 = x * x * x;
        float inner = sqrtf(2.0f / M_PI) * (x + 0.044715f * x3);
        output[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

// GELU backward pass - approximation
__global__ void gelu_backward_approx(const float* grad_output, const float* input,
                                    float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float x2 = x * x;
        float x3 = x2 * x;
        
        float sqrt_2_pi = sqrtf(2.0f / M_PI);
        float inner = sqrt_2_pi * (x + 0.044715f * x3);
        float tanh_inner = tanhf(inner);
        float sech2_inner = 1.0f - tanh_inner * tanh_inner; // sech^2(x) = 1 - tanh^2(x)
        
        float d_inner_dx = sqrt_2_pi * (1.0f + 3.0f * 0.044715f * x2);
        
        float grad = 0.5f * (1.0f + tanh_inner) + 0.5f * x * sech2_inner * d_inner_dx;
        grad_input[idx] = grad_output[idx] * grad;
    }
}

// Swish activation (x * sigmoid(x)) - used in modern architectures
__global__ void swish_forward(const float* input, float* output, int size, float beta = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-beta * x));
        output[idx] = x * sigmoid;
    }
}

// Swish backward pass
__global__ void swish_backward(const float* grad_output, const float* input,
                              float* grad_input, int size, float beta = 1.0f) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float sigmoid = 1.0f / (1.0f + expf(-beta * x));
        float swish = x * sigmoid;
        
        // Derivative: sigmoid + swish * beta * (1 - sigmoid)
        float grad = sigmoid + swish * beta * (1.0f - sigmoid);
        grad_input[idx] = grad_output[idx] * grad;
    }
}

// Mish activation (x * tanh(softplus(x))) - state-of-the-art activation
__global__ void mish_forward(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float softplus = logf(1.0f + expf(x));
        output[idx] = x * tanhf(softplus);
    }
}

// Mish backward pass
__global__ void mish_backward(const float* grad_output, const float* input,
                             float* grad_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        float exp_x = expf(x);
        float exp_2x = exp_x * exp_x;
        float omega = (4.0f * (x + 1.0f) + 4.0f * exp_2x + exp_x * (4.0f * x + 6.0f)) / 
                     ((2.0f * exp_x + exp_2x + 2.0f) * (2.0f * exp_x + exp_2x + 2.0f));
        grad_input[idx] = grad_output[idx] * omega;
    }
}

// In-place ReLU for memory efficiency
__global__ void relu_inplace(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
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

template<typename Func>
float benchmark_kernel(Func kernel_func, const std::string& name) {
    auto start = std::chrono::high_resolution_clock::now();
    kernel_func();
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration<float, std::milli>(end - start).count();
    std::cout << name << ": " << duration << " ms" << std::endl;
    return duration;
}

int main() {
    const int size = 1024 * 1024; // 1M elements
    const float alpha = 1.0f;
    const float negative_slope = 0.01f;
    const float beta = 1.0f;
    
    // Host memory
    std::vector<float> h_input(size);
    std::vector<float> h_output(size);
    std::vector<float> h_grad_output(size);
    std::vector<float> h_grad_input(size);
    
    initialize_random(h_input.data(), size, 0.0f, 2.0f); // Mix of positive and negative
    initialize_random(h_grad_output.data(), size, 0.0f, 1.0f);
    
    // Device memory
    float *d_input, *d_output, *d_grad_output, *d_grad_input;
    cudaMalloc(&d_input, size * sizeof(float));
    cudaMalloc(&d_output, size * sizeof(float));
    cudaMalloc(&d_grad_output, size * sizeof(float));
    cudaMalloc(&d_grad_input, size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel configuration
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    std::cout << "Activation Function Performance Benchmarks:\n";
    std::cout << "Input size: " << size << " elements\n\n";
    
    // Benchmark forward passes
    std::cout << "Forward Pass Benchmarks:\n";
    benchmark_kernel([&]() {
        relu_forward<<<blocks, threads_per_block>>>(d_input, d_output, size);
    }, "ReLU Forward");
    
    benchmark_kernel([&]() {
        leaky_relu_forward<<<blocks, threads_per_block>>>(d_input, d_output, size, negative_slope);
    }, "Leaky ReLU Forward");
    
    benchmark_kernel([&]() {
        elu_forward<<<blocks, threads_per_block>>>(d_input, d_output, size, alpha);
    }, "ELU Forward");
    
    benchmark_kernel([&]() {
        gelu_forward_approx<<<blocks, threads_per_block>>>(d_input, d_output, size);
    }, "GELU Forward (Approx)");
    
    benchmark_kernel([&]() {
        swish_forward<<<blocks, threads_per_block>>>(d_input, d_output, size, beta);
    }, "Swish Forward");
    
    benchmark_kernel([&]() {
        mish_forward<<<blocks, threads_per_block>>>(d_input, d_output, size);
    }, "Mish Forward");
    
    std::cout << "\nBackward Pass Benchmarks:\n";
    
    // First compute forward pass for ELU (needed for backward)
    elu_forward<<<blocks, threads_per_block>>>(d_input, d_output, size, alpha);
    cudaDeviceSynchronize();
    
    benchmark_kernel([&]() {
        relu_backward<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_grad_input, size);
    }, "ReLU Backward");
    
    benchmark_kernel([&]() {
        leaky_relu_backward<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_grad_input, size, negative_slope);
    }, "Leaky ReLU Backward");
    
    benchmark_kernel([&]() {
        elu_backward<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_output, d_grad_input, size, alpha);
    }, "ELU Backward");
    
    benchmark_kernel([&]() {
        gelu_backward_approx<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_grad_input, size);
    }, "GELU Backward (Approx)");
    
    benchmark_kernel([&]() {
        swish_backward<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_grad_input, size, beta);
    }, "Swish Backward");
    
    benchmark_kernel([&]() {
        mish_backward<<<blocks, threads_per_block>>>(d_grad_output, d_input, d_grad_input, size);
    }, "Mish Backward");
    
    // Test in-place operation
    float *d_inplace_data;
    cudaMalloc(&d_inplace_data, size * sizeof(float));
    cudaMemcpy(d_inplace_data, d_input, size * sizeof(float), cudaMemcpyDeviceToDevice);
    
    std::cout << "\nIn-place Operation:\n";
    benchmark_kernel([&]() {
        relu_inplace<<<blocks, threads_per_block>>>(d_inplace_data, size);
    }, "ReLU In-place");
    
    // Verify results
    cudaMemcpy(h_output.data(), d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Check some statistics
    int positive_count = 0, negative_count = 0, zero_count = 0;
    for (int i = 0; i < std::min(1000, size); i++) {
        if (h_output[i] > 0.0f) positive_count++;
        else if (h_output[i] < 0.0f) negative_count++;
        else zero_count++;
    }
    
    std::cout << "\nResult Statistics (first 1000 elements):\n";
    std::cout << "Positive: " << positive_count << ", Negative: " << negative_count 
              << ", Zero: " << zero_count << std::endl;
    std::cout << "First few elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_grad_output);
    cudaFree(d_grad_input);
    cudaFree(d_inplace_data);
    
    return 0;
}