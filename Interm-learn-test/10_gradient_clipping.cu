#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Global gradient norm computation kernel
__global__ void compute_gradient_norm_squared(const float** gradients, float* partial_norms,
                                             const int* sizes, int num_tensors, int block_offset) {
    
    int tid = threadIdx.x;
    int tensor_idx = blockIdx.y;
    int block_start = blockIdx.x * blockDim.x;
    
    extern __shared__ float sdata[];
    
    float local_norm_sq = 0.0f;
    
    if (tensor_idx < num_tensors) {
        const float* grad = gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        // Compute sum of squares for this block
        for (int i = block_start + tid; i < tensor_size; i += blockDim.x * gridDim.x) {
            float val = grad[i];
            local_norm_sq += val * val;
        }
    }
    
    sdata[tid] = local_norm_sq;
    __syncthreads();
    
    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Store partial result
    if (tid == 0) {
        int output_idx = tensor_idx * gridDim.x + blockIdx.x;
        partial_norms[output_idx] = sdata[0];
    }
}

// Gradient clipping by global norm
__global__ void clip_gradients_by_global_norm(float** gradients, const int* sizes,
                                             float clip_norm, float total_norm,
                                             int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx < num_tensors) {
        float* grad = gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        if (idx < tensor_size) {
            // Apply clipping if total norm exceeds threshold
            if (total_norm > clip_norm) {
                grad[idx] = grad[idx] * (clip_norm / total_norm);
            }
        }
    }
}

// Gradient clipping by value (element-wise)
__global__ void clip_gradients_by_value(float** gradients, const int* sizes,
                                       float clip_min, float clip_max, int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx < num_tensors) {
        float* grad = gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        if (idx < tensor_size) {
            grad[idx] = fmaxf(clip_min, fminf(clip_max, grad[idx]));
        }
    }
}

// Adaptive gradient clipping (per-layer)
__global__ void adaptive_gradient_clipping(float** gradients, float** parameters,
                                          const int* sizes, float clip_factor,
                                          int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx >= num_tensors) return;
    
    float* grad = gradients[tensor_idx];
    float* param = parameters[tensor_idx];
    int tensor_size = sizes[tensor_idx];
    
    extern __shared__ float sdata[];
    float* s_grad_norm = sdata;
    float* s_param_norm = sdata + blockDim.x;
    
    // Compute norms for this tensor
    float local_grad_norm = 0.0f;
    float local_param_norm = 0.0f;
    
    for (int i = idx; i < tensor_size; i += blockDim.x * gridDim.x) {
        float g = grad[i];
        float p = param[i];
        local_grad_norm += g * g;
        local_param_norm += p * p;
    }
    
    s_grad_norm[threadIdx.x] = local_grad_norm;
    s_param_norm[threadIdx.x] = local_param_norm;
    
    __syncthreads();
    
    // Reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_grad_norm[threadIdx.x] += s_grad_norm[threadIdx.x + stride];
            s_param_norm[threadIdx.x] += s_param_norm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    float grad_norm = sqrtf(s_grad_norm[0]);
    float param_norm = sqrtf(s_param_norm[0]);
    
    // Compute adaptive clipping threshold
    float max_norm = clip_factor * fmaxf(param_norm, 1e-3f);
    
    // Apply clipping
    if (grad_norm > max_norm) {
        float scale = max_norm / grad_norm;
        for (int i = idx; i < tensor_size; i += blockDim.x * gridDim.x) {
            grad[i] *= scale;
        }
    }
}

// Gradient accumulation with overflow detection
__global__ void accumulate_gradients_with_overflow_check(float** src_gradients, float** dst_gradients,
                                                        const int* sizes, bool* overflow_detected,
                                                        int accumulation_steps, int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx < num_tensors) {
        float* src_grad = src_gradients[tensor_idx];
        float* dst_grad = dst_gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        if (idx < tensor_size) {
            float src_val = src_grad[idx];
            float dst_val = dst_grad[idx];
            
            // Check for overflow/underflow
            if (!isfinite(src_val) || !isfinite(dst_val)) {
                *overflow_detected = true;
                return;
            }
            
            // Accumulate gradient
            float accumulated = dst_val + src_val / accumulation_steps;
            
            if (!isfinite(accumulated)) {
                *overflow_detected = true;
                return;
            }
            
            dst_grad[idx] = accumulated;
        }
    }
}

// Stochastic gradient clipping (randomly clip some gradients)
__global__ void stochastic_gradient_clipping(float** gradients, const int* sizes,
                                            const float* random_values, float clip_probability,
                                            float clip_norm, int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx < num_tensors) {
        float* grad = gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        if (idx < tensor_size) {
            float random_val = random_values[tensor_idx * tensor_size + idx];
            
            if (random_val < clip_probability) {
                // Apply clipping to this gradient
                float grad_val = grad[idx];
                if (fabsf(grad_val) > clip_norm) {
                    grad[idx] = (grad_val > 0) ? clip_norm : -clip_norm;
                }
            }
        }
    }
}

// Percentile-based gradient clipping
__global__ void percentile_gradient_clipping(float** gradients, const int* sizes,
                                            float percentile, int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    
    if (tensor_idx >= num_tensors) return;
    
    float* grad = gradients[tensor_idx];
    int tensor_size = sizes[tensor_idx];
    
    // This is a simplified version - in practice, you'd need to sort gradients
    // or use a more sophisticated percentile computation
    
    extern __shared__ float s_values[];
    int tid = threadIdx.x;
    
    // Load gradients into shared memory (simplified for demo)
    float local_max = 0.0f;
    for (int i = tid; i < tensor_size; i += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(grad[i]));
    }
    
    s_values[tid] = local_max;
    __syncthreads();
    
    // Reduction to find maximum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_values[tid] = fmaxf(s_values[tid], s_values[tid + stride]);
        }
        __syncthreads();
    }
    
    float max_grad = s_values[0];
    float threshold = max_grad * percentile; // Simplified percentile
    
    // Apply clipping
    for (int i = tid; i < tensor_size; i += blockDim.x) {
        if (fabsf(grad[i]) > threshold) {
            grad[i] = (grad[i] > 0) ? threshold : -threshold;
        }
    }
}

// Gradient norm scaling for mixed precision training
__global__ void scale_gradients(float** gradients, const int* sizes, float scale_factor,
                               int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tensor_idx < num_tensors) {
        float* grad = gradients[tensor_idx];
        int tensor_size = sizes[tensor_idx];
        
        if (idx < tensor_size) {
            grad[idx] *= scale_factor;
        }
    }
}

// Gradient statistics collection
__global__ void collect_gradient_stats(const float** gradients, const int* sizes,
                                      float* means, float* variances, float* max_vals,
                                      int num_tensors) {
    
    int tensor_idx = blockIdx.y;
    int tid = threadIdx.x;
    
    if (tensor_idx >= num_tensors) return;
    
    const float* grad = gradients[tensor_idx];
    int tensor_size = sizes[tensor_idx];
    
    extern __shared__ float sdata[];
    float* s_sum = sdata;
    float* s_sum_sq = sdata + blockDim.x;
    float* s_max = sdata + 2 * blockDim.x;
    
    // Compute local statistics
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    float local_max = 0.0f;
    
    for (int i = tid; i < tensor_size; i += blockDim.x) {
        float val = grad[i];
        local_sum += val;
        local_sum_sq += val * val;
        local_max = fmaxf(local_max, fabsf(val));
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    s_max[tid] = local_max;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float mean = s_sum[0] / tensor_size;
        float variance = (s_sum_sq[0] / tensor_size) - (mean * mean);
        
        means[tensor_idx] = mean;
        variances[tensor_idx] = variance;
        max_vals[tensor_idx] = s_max[0];
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
    const int num_layers = 6;
    const std::vector<int> layer_sizes = {1024*768, 768*3072, 3072*768, 768*768, 768*768, 768*50000};
    const float clip_norm = 1.0f;
    const float clip_value = 5.0f;
    const float adaptive_clip_factor = 0.01f;
    const float percentile = 0.9f;
    const float scale_factor = 1.0f / 1024.0f; // For mixed precision
    
    int total_parameters = 0;
    for (int size : layer_sizes) {
        total_parameters += size;
    }
    
    std::cout << "Gradient Clipping Benchmark\n";
    std::cout << "Number of layers: " << num_layers << std::endl;
    std::cout << "Total parameters: " << total_parameters << std::endl;
    
    // Host memory
    std::vector<std::vector<float>> h_gradients(num_layers);
    std::vector<std::vector<float>> h_parameters(num_layers);
    std::vector<float*> h_grad_ptrs(num_layers);
    std::vector<float*> h_param_ptrs(num_layers);
    
    for (int i = 0; i < num_layers; i++) {
        h_gradients[i].resize(layer_sizes[i]);
        h_parameters[i].resize(layer_sizes[i]);
        initialize_random(h_gradients[i].data(), layer_sizes[i], 0.0f, 0.1f);
        initialize_random(h_parameters[i].data(), layer_sizes[i], 0.0f, 1.0f);
    }
    
    // Device memory
    std::vector<float*> d_gradients(num_layers);
    std::vector<float*> d_parameters(num_layers);
    std::vector<int> h_sizes(layer_sizes);
    
    int *d_sizes;
    float **d_grad_ptrs, **d_param_ptrs;
    float *d_partial_norms;
    bool *d_overflow;
    
    cudaMalloc(&d_sizes, num_layers * sizeof(int));
    cudaMalloc(&d_grad_ptrs, num_layers * sizeof(float*));
    cudaMalloc(&d_param_ptrs, num_layers * sizeof(float*));
    cudaMalloc(&d_partial_norms, num_layers * 256 * sizeof(float)); // Assuming max 256 blocks per layer
    cudaMalloc(&d_overflow, sizeof(bool));
    
    // Allocate and copy gradient data
    for (int i = 0; i < num_layers; i++) {
        cudaMalloc(&d_gradients[i], layer_sizes[i] * sizeof(float));
        cudaMalloc(&d_parameters[i], layer_sizes[i] * sizeof(float));
        
        cudaMemcpy(d_gradients[i], h_gradients[i].data(), layer_sizes[i] * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_parameters[i], h_parameters[i].data(), layer_sizes[i] * sizeof(float), cudaMemcpyHostToDevice);
        
        h_grad_ptrs[i] = d_gradients[i];
        h_param_ptrs[i] = d_parameters[i];
    }
    
    cudaMemcpy(d_sizes, h_sizes.data(), num_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_ptrs, h_grad_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_param_ptrs, h_param_ptrs.data(), num_layers * sizeof(float*), cudaMemcpyHostToDevice);
    
    // Test gradient norm computation
    auto start = std::chrono::high_resolution_clock::now();
    
    const int blocks_per_tensor = 256;
    dim3 norm_grid(blocks_per_tensor, num_layers);
    dim3 norm_block(256);
    
    compute_gradient_norm_squared<<<norm_grid, norm_block, norm_block.x * sizeof(float)>>>(
        (const float**)d_grad_ptrs, d_partial_norms, d_sizes, num_layers, 0);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto norm_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy back partial norms and compute total
    std::vector<float> h_partial_norms(num_layers * blocks_per_tensor);
    cudaMemcpy(h_partial_norms.data(), d_partial_norms, num_layers * blocks_per_tensor * sizeof(float), cudaMemcpyDeviceToHost);
    
    float total_norm_sq = 0.0f;
    for (float partial : h_partial_norms) {
        total_norm_sq += partial;
    }
    float total_norm = sqrtf(total_norm_sq);
    
    // Test global gradient clipping
    start = std::chrono::high_resolution_clock::now();
    
    // Find maximum tensor size for grid configuration
    int max_size = *std::max_element(layer_sizes.begin(), layer_sizes.end());
    int blocks_per_layer = (max_size + 255) / 256;
    
    dim3 clip_grid(blocks_per_layer, num_layers);
    dim3 clip_block(256);
    
    clip_gradients_by_global_norm<<<clip_grid, clip_block>>>(
        d_grad_ptrs, d_sizes, clip_norm, total_norm, num_layers);
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto global_clip_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test value clipping
    start = std::chrono::high_resolution_clock::now();
    
    clip_gradients_by_value<<<clip_grid, clip_block>>>(
        d_grad_ptrs, d_sizes, -clip_value, clip_value, num_layers);
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto value_clip_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test adaptive clipping
    start = std::chrono::high_resolution_clock::now();
    
    adaptive_gradient_clipping<<<clip_grid, clip_block, 2 * clip_block.x * sizeof(float)>>>(
        d_grad_ptrs, d_param_ptrs, d_sizes, adaptive_clip_factor, num_layers);
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto adaptive_clip_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test gradient scaling
    start = std::chrono::high_resolution_clock::now();
    
    scale_gradients<<<clip_grid, clip_block>>>(
        d_grad_ptrs, d_sizes, scale_factor, num_layers);
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto scale_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test gradient statistics collection
    float *d_means, *d_variances, *d_max_vals;
    cudaMalloc(&d_means, num_layers * sizeof(float));
    cudaMalloc(&d_variances, num_layers * sizeof(float));
    cudaMalloc(&d_max_vals, num_layers * sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    
    dim3 stats_grid(1, num_layers);
    dim3 stats_block(256);
    
    collect_gradient_stats<<<stats_grid, stats_block, 3 * stats_block.x * sizeof(float)>>>(
        (const float**)d_grad_ptrs, d_sizes, d_means, d_variances, d_max_vals, num_layers);
    
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto stats_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy back results
    std::vector<float> h_means(num_layers), h_variances(num_layers), h_max_vals(num_layers);
    cudaMemcpy(h_means.data(), d_means, num_layers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_variances.data(), d_variances, num_layers * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_max_vals.data(), d_max_vals, num_layers * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Copy back final gradients for verification
    for (int i = 0; i < num_layers; i++) {
        cudaMemcpy(h_gradients[i].data(), d_gradients[i], layer_sizes[i] * sizeof(float), cudaMemcpyDeviceToHost);
    }
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Gradient Norm Computation: " << norm_time << " ms\n";
    std::cout << "Global Gradient Clipping: " << global_clip_time << " ms\n";
    std::cout << "Value Gradient Clipping: " << value_clip_time << " ms\n";
    std::cout << "Adaptive Gradient Clipping: " << adaptive_clip_time << " ms\n";
    std::cout << "Gradient Scaling: " << scale_time << " ms\n";
    std::cout << "Gradient Statistics: " << stats_time << " ms\n";
    
    std::cout << "\nGradient Statistics:\n";
    std::cout << "Total gradient norm: " << total_norm << std::endl;
    std::cout << "Clipped: " << (total_norm > clip_norm ? "Yes" : "No") << std::endl;
    
    std::cout << "\nPer-layer Statistics:\n";
    for (int i = 0; i < num_layers; i++) {
        std::cout << "Layer " << i << " - Mean: " << h_means[i] 
                  << ", Std: " << sqrtf(h_variances[i]) 
                  << ", Max: " << h_max_vals[i] << std::endl;
    }
    
    // Verify some gradients
    std::cout << "\nSample gradients from first layer: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_gradients[0][i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    for (int i = 0; i < num_layers; i++) {
        cudaFree(d_gradients[i]);
        cudaFree(d_parameters[i]);
    }
    
    cudaFree(d_sizes);
    cudaFree(d_grad_ptrs);
    cudaFree(d_param_ptrs);
    cudaFree(d_partial_norms);
    cudaFree(d_overflow);
    cudaFree(d_means);
    cudaFree(d_variances);
    cudaFree(d_max_vals);
    
    return 0;
}