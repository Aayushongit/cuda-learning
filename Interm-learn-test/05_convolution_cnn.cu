#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Basic 2D convolution kernel (without padding)
__global__ void conv2d_basic(const float* input, const float* kernel, float* output,
                            int batch_size, int in_channels, int out_channels,
                            int input_height, int input_width,
                            int kernel_height, int kernel_width,
                            int output_height, int output_width) {
    
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;
    
    if (batch_idx < batch_size && out_ch < out_channels && 
        out_y < output_height && out_x < output_width) {
        
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int in_y = out_y + ky;
                    int in_x = out_x + kx;
                    
                    if (in_y < input_height && in_x < input_width) {
                        int input_idx = ((batch_idx * in_channels + in_ch) * input_height + in_y) 
                                       * input_width + in_x;
                        int kernel_idx = ((out_ch * in_channels + in_ch) * kernel_height + ky) 
                                        * kernel_width + kx;
                        
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * out_channels + out_ch) * output_height + out_y) 
                        * output_width + out_x;
        output[output_idx] = sum;
    }
}

// Optimized convolution with shared memory
__global__ void conv2d_shared_memory(const float* input, const float* kernel, float* output,
                                    int batch_size, int in_channels, int out_channels,
                                    int input_height, int input_width,
                                    int kernel_height, int kernel_width,
                                    int output_height, int output_width,
                                    int stride = 1, int padding = 0) {
    
    extern __shared__ float shared_mem[];
    float* s_input = shared_mem;
    float* s_kernel = shared_mem + blockDim.x * blockDim.y * in_channels;
    
    int batch_idx = blockIdx.z;
    int out_ch = blockIdx.y;
    int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;
    
    // Load kernel into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int kernel_size = kernel_height * kernel_width * in_channels;
    
    for (int i = tid; i < kernel_size; i += blockDim.x * blockDim.y) {
        int kernel_idx = out_ch * kernel_size + i;
        s_kernel[i] = kernel[kernel_idx];
    }
    
    __syncthreads();
    
    if (batch_idx < batch_size && out_ch < out_channels && 
        out_y < output_height && out_x < output_width) {
        
        float sum = 0.0f;
        
        for (int in_ch = 0; in_ch < in_channels; in_ch++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int in_y = out_y * stride + ky - padding;
                    int in_x = out_x * stride + kx - padding;
                    
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_idx = ((batch_idx * in_channels + in_ch) * input_height + in_y) 
                                       * input_width + in_x;
                        int kernel_local_idx = (in_ch * kernel_height + ky) * kernel_width + kx;
                        
                        sum += input[input_idx] * s_kernel[kernel_local_idx];
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * out_channels + out_ch) * output_height + out_y) 
                        * output_width + out_x;
        output[output_idx] = sum;
    }
}

// Convolution backward pass - gradients w.r.t. input
__global__ void conv2d_backward_input(const float* grad_output, const float* kernel, float* grad_input,
                                     int batch_size, int in_channels, int out_channels,
                                     int input_height, int input_width,
                                     int kernel_height, int kernel_width,
                                     int output_height, int output_width,
                                     int stride = 1, int padding = 0) {
    
    int batch_idx = blockIdx.z;
    int in_ch = blockIdx.y;
    int in_y = blockIdx.x * blockDim.y + threadIdx.y;
    int in_x = threadIdx.x;
    
    if (batch_idx < batch_size && in_ch < in_channels && 
        in_y < input_height && in_x < input_width) {
        
        float grad = 0.0f;
        
        for (int out_ch = 0; out_ch < out_channels; out_ch++) {
            for (int ky = 0; ky < kernel_height; ky++) {
                for (int kx = 0; kx < kernel_width; kx++) {
                    int out_y = (in_y + padding - ky) / stride;
                    int out_x = (in_x + padding - kx) / stride;
                    
                    // Check if output position is valid and aligned
                    if (out_y >= 0 && out_y < output_height && out_x >= 0 && out_x < output_width &&
                        (in_y + padding - ky) % stride == 0 && (in_x + padding - kx) % stride == 0) {
                        
                        int grad_output_idx = ((batch_idx * out_channels + out_ch) * output_height + out_y) 
                                             * output_width + out_x;
                        int kernel_idx = ((out_ch * in_channels + in_ch) * kernel_height + ky) 
                                        * kernel_width + kx;
                        
                        grad += grad_output[grad_output_idx] * kernel[kernel_idx];
                    }
                }
            }
        }
        
        int grad_input_idx = ((batch_idx * in_channels + in_ch) * input_height + in_y) 
                            * input_width + in_x;
        grad_input[grad_input_idx] = grad;
    }
}

// Convolution backward pass - gradients w.r.t. kernel weights
__global__ void conv2d_backward_kernel(const float* input, const float* grad_output, float* grad_kernel,
                                      int batch_size, int in_channels, int out_channels,
                                      int input_height, int input_width,
                                      int kernel_height, int kernel_width,
                                      int output_height, int output_width,
                                      int stride = 1, int padding = 0) {
    
    int out_ch = blockIdx.z;
    int in_ch = blockIdx.y;
    int ky = blockIdx.x * blockDim.y + threadIdx.y;
    int kx = threadIdx.x;
    
    if (out_ch < out_channels && in_ch < in_channels && 
        ky < kernel_height && kx < kernel_width) {
        
        float grad = 0.0f;
        
        for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
            for (int out_y = 0; out_y < output_height; out_y++) {
                for (int out_x = 0; out_x < output_width; out_x++) {
                    int in_y = out_y * stride + ky - padding;
                    int in_x = out_x * stride + kx - padding;
                    
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                        int input_idx = ((batch_idx * in_channels + in_ch) * input_height + in_y) 
                                       * input_width + in_x;
                        int grad_output_idx = ((batch_idx * out_channels + out_ch) * output_height + out_y) 
                                             * output_width + out_x;
                        
                        grad += input[input_idx] * grad_output[grad_output_idx];
                    }
                }
            }
        }
        
        int grad_kernel_idx = ((out_ch * in_channels + in_ch) * kernel_height + ky) 
                             * kernel_width + kx;
        grad_kernel[grad_kernel_idx] = grad / batch_size;
    }
}

// Max pooling forward pass
__global__ void max_pool2d_forward(const float* input, float* output, int* indices,
                                  int batch_size, int channels,
                                  int input_height, int input_width,
                                  int output_height, int output_width,
                                  int pool_height, int pool_width,
                                  int stride_y, int stride_x) {
    
    int batch_idx = blockIdx.z;
    int ch = blockIdx.y;
    int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;
    
    if (batch_idx < batch_size && ch < channels && 
        out_y < output_height && out_x < output_width) {
        
        float max_val = -FLT_MAX;
        int max_idx = -1;
        
        for (int py = 0; py < pool_height; py++) {
            for (int px = 0; px < pool_width; px++) {
                int in_y = out_y * stride_y + py;
                int in_x = out_x * stride_x + px;
                
                if (in_y < input_height && in_x < input_width) {
                    int input_idx = ((batch_idx * channels + ch) * input_height + in_y) 
                                   * input_width + in_x;
                    
                    if (input[input_idx] > max_val) {
                        max_val = input[input_idx];
                        max_idx = input_idx;
                    }
                }
            }
        }
        
        int output_idx = ((batch_idx * channels + ch) * output_height + out_y) 
                        * output_width + out_x;
        output[output_idx] = max_val;
        indices[output_idx] = max_idx; // Store index for backward pass
    }
}

// Max pooling backward pass
__global__ void max_pool2d_backward(const float* grad_output, const int* indices,
                                   float* grad_input, int batch_size, int channels,
                                   int input_height, int input_width,
                                   int output_height, int output_width) {
    
    int batch_idx = blockIdx.z;
    int ch = blockIdx.y;
    int out_y = blockIdx.x * blockDim.y + threadIdx.y;
    int out_x = threadIdx.x;
    
    if (batch_idx < batch_size && ch < channels && 
        out_y < output_height && out_x < output_width) {
        
        int output_idx = ((batch_idx * channels + ch) * output_height + out_y) 
                        * output_width + out_x;
        int max_input_idx = indices[output_idx];
        
        if (max_input_idx >= 0) {
            atomicAdd(&grad_input[max_input_idx], grad_output[output_idx]);
        }
    }
}

// Batch normalization forward pass
__global__ void batch_norm_forward(const float* input, const float* scale, const float* bias,
                                  const float* running_mean, const float* running_var,
                                  float* output, float* save_mean, float* save_var,
                                  int batch_size, int channels, int spatial_size,
                                  float epsilon = 1e-5f, float momentum = 0.1f, bool training = true) {
    
    int ch = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (ch < channels) {
        float mean_val, var_val;
        
        if (training) {
            // Compute batch statistics
            float sum = 0.0f, sum_sq = 0.0f;
            int total_elements = batch_size * spatial_size;
            
            for (int b = 0; b < batch_size; b++) {
                for (int s = 0; s < spatial_size; s++) {
                    int idx = (b * channels + ch) * spatial_size + s;
                    float val = input[idx];
                    sum += val;
                    sum_sq += val * val;
                }
            }
            
            mean_val = sum / total_elements;
            var_val = (sum_sq / total_elements) - (mean_val * mean_val);
            
            // Save for backward pass
            save_mean[ch] = mean_val;
            save_var[ch] = var_val;
            
            // Update running statistics
            // This should be done atomically in practice
            // running_mean[ch] = (1 - momentum) * running_mean[ch] + momentum * mean_val;
            // running_var[ch] = (1 - momentum) * running_var[ch] + momentum * var_val;
        } else {
            mean_val = running_mean[ch];
            var_val = running_var[ch];
        }
        
        // Normalize
        float std_val = sqrtf(var_val + epsilon);
        for (int b = 0; b < batch_size; b++) {
            for (int s = 0; s < spatial_size; s++) {
                int idx = (b * channels + ch) * spatial_size + s;
                float normalized = (input[idx] - mean_val) / std_val;
                output[idx] = scale[ch] * normalized + bias[ch];
            }
        }
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
    // CNN parameters
    const int batch_size = 8;
    const int in_channels = 3;
    const int out_channels = 64;
    const int input_height = 32, input_width = 32;
    const int kernel_height = 3, kernel_width = 3;
    const int stride = 1, padding = 1;
    
    const int output_height = (input_height + 2 * padding - kernel_height) / stride + 1;
    const int output_width = (input_width + 2 * padding - kernel_width) / stride + 1;
    
    // Memory sizes
    int input_size = batch_size * in_channels * input_height * input_width;
    int kernel_size = out_channels * in_channels * kernel_height * kernel_width;
    int output_size = batch_size * out_channels * output_height * output_width;
    
    // Host memory
    std::vector<float> h_input(input_size);
    std::vector<float> h_kernel(kernel_size);
    std::vector<float> h_output(output_size);
    std::vector<float> h_grad_output(output_size);
    
    initialize_random(h_input.data(), input_size, 0.0f, 1.0f);
    
    // Xavier initialization for kernel
    float xavier_std = sqrtf(2.0f / (in_channels * kernel_height * kernel_width));
    initialize_random(h_kernel.data(), kernel_size, 0.0f, xavier_std);
    initialize_random(h_grad_output.data(), output_size, 0.0f, 0.1f);
    
    // Device memory
    float *d_input, *d_kernel, *d_output, *d_grad_output, *d_grad_input, *d_grad_kernel;
    
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    cudaMalloc(&d_grad_input, input_size * sizeof(float));
    cudaMalloc(&d_grad_kernel, kernel_size * sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel.data(), kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_output, h_grad_output.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "CNN Layer Benchmark\n";
    std::cout << "Input: " << batch_size << "x" << in_channels << "x" << input_height << "x" << input_width << std::endl;
    std::cout << "Kernel: " << out_channels << "x" << in_channels << "x" << kernel_height << "x" << kernel_width << std::endl;
    std::cout << "Output: " << batch_size << "x" << out_channels << "x" << output_height << "x" << output_width << std::endl;
    
    // Kernel configurations
    dim3 block(16, 16);
    dim3 grid((output_width + block.x - 1) / block.x, out_channels, batch_size);
    
    // Test basic convolution
    auto start = std::chrono::high_resolution_clock::now();
    conv2d_basic<<<grid, block>>>(d_input, d_kernel, d_output,
                                 batch_size, in_channels, out_channels,
                                 input_height, input_width,
                                 kernel_height, kernel_width,
                                 output_height, output_width);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto basic_conv_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test shared memory convolution
    int shared_mem_size = (block.x * block.y * in_channels + kernel_height * kernel_width * in_channels) * sizeof(float);
    
    start = std::chrono::high_resolution_clock::now();
    conv2d_shared_memory<<<grid, block, shared_mem_size>>>(d_input, d_kernel, d_output,
                                                          batch_size, in_channels, out_channels,
                                                          input_height, input_width,
                                                          kernel_height, kernel_width,
                                                          output_height, output_width,
                                                          stride, padding);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto shared_conv_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test backward passes
    dim3 input_grid((input_width + block.x - 1) / block.x, in_channels, batch_size);
    
    start = std::chrono::high_resolution_clock::now();
    conv2d_backward_input<<<input_grid, block>>>(d_grad_output, d_kernel, d_grad_input,
                                                 batch_size, in_channels, out_channels,
                                                 input_height, input_width,
                                                 kernel_height, kernel_width,
                                                 output_height, output_width,
                                                 stride, padding);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto backward_input_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    dim3 kernel_grid((kernel_width + block.x - 1) / block.x, in_channels, out_channels);
    
    start = std::chrono::high_resolution_clock::now();
    conv2d_backward_kernel<<<kernel_grid, block>>>(d_input, d_grad_output, d_grad_kernel,
                                                   batch_size, in_channels, out_channels,
                                                   input_height, input_width,
                                                   kernel_height, kernel_width,
                                                   output_height, output_width,
                                                   stride, padding);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto backward_kernel_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test max pooling
    const int pool_height = 2, pool_width = 2;
    const int pool_stride = 2;
    const int pool_output_height = output_height / pool_stride;
    const int pool_output_width = output_width / pool_stride;
    
    int pool_output_size = batch_size * out_channels * pool_output_height * pool_output_width;
    
    float *d_pool_output;
    int *d_pool_indices;
    cudaMalloc(&d_pool_output, pool_output_size * sizeof(float));
    cudaMalloc(&d_pool_indices, pool_output_size * sizeof(int));
    
    dim3 pool_grid((pool_output_width + block.x - 1) / block.x, out_channels, batch_size);
    
    start = std::chrono::high_resolution_clock::now();
    max_pool2d_forward<<<pool_grid, block>>>(d_output, d_pool_output, d_pool_indices,
                                            batch_size, out_channels,
                                            output_height, output_width,
                                            pool_output_height, pool_output_width,
                                            pool_height, pool_width,
                                            pool_stride, pool_stride);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto pool_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back for verification
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Basic Convolution: " << basic_conv_time << " ms\n";
    std::cout << "Shared Memory Convolution: " << shared_conv_time << " ms\n";
    std::cout << "Backward Input: " << backward_input_time << " ms\n";
    std::cout << "Backward Kernel: " << backward_kernel_time << " ms\n";
    std::cout << "Max Pooling: " << pool_time << " ms\n";
    
    // Output statistics
    float output_mean = 0.0f, output_abs_mean = 0.0f;
    for (int i = 0; i < output_size; i++) {
        output_mean += h_output[i];
        output_abs_mean += std::abs(h_output[i]);
    }
    output_mean /= output_size;
    output_abs_mean /= output_size;
    
    std::cout << "\nOutput Statistics:\n";
    std::cout << "Mean: " << output_mean << std::endl;
    std::cout << "Mean Absolute: " << output_abs_mean << std::endl;
    std::cout << "First few elements: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_input); cudaFree(d_kernel); cudaFree(d_output);
    cudaFree(d_grad_output); cudaFree(d_grad_input); cudaFree(d_grad_kernel);
    cudaFree(d_pool_output); cudaFree(d_pool_indices);
    
    return 0;
}