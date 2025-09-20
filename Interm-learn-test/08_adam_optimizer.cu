#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// ADAM optimizer kernel with bias correction
__global__ void adam_optimizer(float* weights, float* gradients, float* m, float* v,
                              float learning_rate, float beta1, float beta2, float epsilon,
                              int step, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        
        // Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update weights
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ADAM optimizer with weight decay (AdamW)
__global__ void adamw_optimizer(float* weights, float* gradients, float* m, float* v,
                               float learning_rate, float beta1, float beta2, float epsilon,
                               float weight_decay, int step, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        float weight = weights[idx];
        
        // Add weight decay to gradient
        grad += weight_decay * weight;
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update weights
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// ADAM optimizer with learning rate scheduling and gradient clipping
__global__ void adam_optimizer_advanced(float* weights, float* gradients, float* m, float* v,
                                       float base_lr, float lr_schedule_factor, float beta1, float beta2, 
                                       float epsilon, float grad_clip_norm, int step, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Gradient clipping
        float grad_norm = fabsf(grad);
        if (grad_norm > grad_clip_norm) {
            grad = grad * (grad_clip_norm / grad_norm);
        }
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Apply learning rate schedule (e.g., step decay)
        float current_lr = base_lr * lr_schedule_factor;
        
        // Update weights
        weights[idx] -= current_lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Mixed precision ADAM optimizer (FP16 gradients, FP32 optimizer state)
__global__ void adam_mixed_precision(float* weights, __half* gradients_fp16, float* gradients_fp32,
                                    float* m, float* v, float learning_rate, float beta1, float beta2,
                                    float epsilon, float loss_scale, int step, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Convert FP16 gradient to FP32 and unscale
        float grad = __half2float(gradients_fp16[idx]) / loss_scale;
        gradients_fp32[idx] = grad;
        
        // Check for gradient overflow/underflow
        if (!isfinite(grad)) {
            return; // Skip update if gradient is not finite
        }
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update weights
        weights[idx] -= learning_rate * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// Layer-wise Adaptive Rate Scaling (LARS) optimizer
__global__ void lars_optimizer(float* weights, float* gradients, float* m, float* v,
                              float learning_rate, float beta1, float beta2, float epsilon,
                              float lars_coeff, float weight_decay, int step, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float sdata[];
    float* s_weight_norm = sdata;
    float* s_grad_norm = sdata + blockDim.x;
    
    if (idx < size) {
        float weight = weights[idx];
        float grad = gradients[idx] + weight_decay * weight;
        
        // Compute local contributions to norms
        s_weight_norm[threadIdx.x] = weight * weight;
        s_grad_norm[threadIdx.x] = grad * grad;
    } else {
        s_weight_norm[threadIdx.x] = 0.0f;
        s_grad_norm[threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction to compute norms
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_weight_norm[threadIdx.x] += s_weight_norm[threadIdx.x + stride];
            s_grad_norm[threadIdx.x] += s_grad_norm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // This is simplified - in practice, you'd need global reduction across all blocks
    float weight_norm = sqrtf(s_weight_norm[0]);
    float grad_norm = sqrtf(s_grad_norm[0]);
    
    if (idx < size) {
        float grad = gradients[idx] + weight_decay * weights[idx];
        
        // Compute layer-wise learning rate
        float local_lr = learning_rate;
        if (grad_norm > 0 && weight_norm > 0) {
            local_lr = learning_rate * lars_coeff * weight_norm / grad_norm;
        }
        
        // Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1.0f - beta1) * grad;
        
        // Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1.0f - beta2) * grad * grad;
        
        // Compute bias-corrected estimates
        float m_hat = m[idx] / (1.0f - powf(beta1, step));
        float v_hat = v[idx] / (1.0f - powf(beta2, step));
        
        // Update weights with layer-wise learning rate
        weights[idx] -= local_lr * m_hat / (sqrtf(v_hat) + epsilon);
    }
}

// RMSprop optimizer (for comparison)
__global__ void rmsprop_optimizer(float* weights, float* gradients, float* v,
                                 float learning_rate, float decay, float epsilon, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float grad = gradients[idx];
        
        // Update running average of squared gradients
        v[idx] = decay * v[idx] + (1.0f - decay) * grad * grad;
        
        // Update weights
        weights[idx] -= learning_rate * grad / (sqrtf(v[idx]) + epsilon);
    }
}

// Lookahead optimizer wrapper (slow weights)
__global__ void lookahead_update(float* fast_weights, float* slow_weights, float alpha, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // Update slow weights: slow = slow + alpha * (fast - slow)
        slow_weights[idx] = slow_weights[idx] + alpha * (fast_weights[idx] - slow_weights[idx]);
        
        // Reset fast weights to slow weights
        fast_weights[idx] = slow_weights[idx];
    }
}

// Compute gradient norms for monitoring
__global__ void compute_gradient_norm(const float* gradients, float* partial_norms, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    extern __shared__ float sdata[];
    
    // Load data into shared memory
    if (idx < size) {
        float grad = gradients[idx];
        sdata[tid] = grad * grad;
    } else {
        sdata[tid] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    
    // Store partial norm for this block
    if (tid == 0) {
        partial_norms[blockIdx.x] = sdata[0];
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

float compute_total_norm(const std::vector<float>& partial_norms) {
    float total = 0.0f;
    for (float norm : partial_norms) {
        total += norm;
    }
    return sqrtf(total);
}

int main() {
    // Model parameters
    const int num_parameters = 1024 * 1024; // 1M parameters
    const int num_steps = 100;
    const int num_blocks = (num_parameters + 255) / 256;
    
    // Optimizer hyperparameters
    const float learning_rate = 0.001f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float epsilon = 1e-8f;
    const float weight_decay = 0.01f;
    const float grad_clip_norm = 1.0f;
    const float lars_coeff = 0.001f;
    const float lookahead_alpha = 0.5f;
    const int lookahead_k = 5;
    const float loss_scale = 1024.0f;
    
    // Host memory
    std::vector<float> h_weights(num_parameters);
    std::vector<float> h_gradients(num_parameters);
    std::vector<float> h_m(num_parameters, 0.0f);
    std::vector<float> h_v(num_parameters, 0.0f);
    std::vector<float> h_partial_norms(num_blocks);
    
    initialize_random(h_weights.data(), num_parameters, 0.0f, 0.1f);
    
    // Device memory
    float *d_weights, *d_gradients, *d_m, *d_v, *d_slow_weights, *d_partial_norms;
    __half *d_gradients_fp16;
    float *d_gradients_fp32;
    
    cudaMalloc(&d_weights, num_parameters * sizeof(float));
    cudaMalloc(&d_gradients, num_parameters * sizeof(float));
    cudaMalloc(&d_m, num_parameters * sizeof(float));
    cudaMalloc(&d_v, num_parameters * sizeof(float));
    cudaMalloc(&d_slow_weights, num_parameters * sizeof(float));
    cudaMalloc(&d_gradients_fp16, num_parameters * sizeof(__half));
    cudaMalloc(&d_gradients_fp32, num_parameters * sizeof(float));
    cudaMalloc(&d_partial_norms, num_blocks * sizeof(float));
    
    cudaMemcpy(d_weights, h_weights.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_slow_weights, h_weights.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_m, 0, num_parameters * sizeof(float));
    cudaMemset(d_v, 0, num_parameters * sizeof(float));
    
    std::cout << "Optimizer Benchmark\n";
    std::cout << "Number of parameters: " << num_parameters << std::endl;
    std::cout << "Number of steps: " << num_steps << std::endl;
    
    // Kernel configuration
    dim3 block(256);
    dim3 grid(num_blocks);
    
    // Benchmark different optimizers
    std::vector<float> step_times;
    
    std::cout << "\nBenchmarking ADAM optimizer:\n";
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (int step = 1; step <= num_steps; step++) {
        // Generate random gradients
        initialize_random(h_gradients.data(), num_parameters, 0.0f, 0.01f);
        cudaMemcpy(d_gradients, h_gradients.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
        
        auto step_start = std::chrono::high_resolution_clock::now();
        
        // Compute gradient norm for monitoring
        compute_gradient_norm<<<grid, block, block.x * sizeof(float)>>>(d_gradients, d_partial_norms, num_parameters);
        
        // Apply ADAM optimizer
        adam_optimizer<<<grid, block>>>(d_weights, d_gradients, d_m, d_v,
                                       learning_rate, beta1, beta2, epsilon, step, num_parameters);
        
        cudaDeviceSynchronize();
        auto step_end = std::chrono::high_resolution_clock::now();
        
        float step_time = std::chrono::duration<float, std::milli>(step_end - step_start).count();
        step_times.push_back(step_time);
        
        // Monitor gradient norm every 10 steps
        if (step % 10 == 0) {
            cudaMemcpy(h_partial_norms.data(), d_partial_norms, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);
            float grad_norm = compute_total_norm(h_partial_norms);
            std::cout << "Step " << step << ", Gradient norm: " << grad_norm 
                      << ", Step time: " << step_time << " ms" << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
    
    float avg_step_time = 0.0f;
    for (float time : step_times) {
        avg_step_time += time;
    }
    avg_step_time /= step_times.size();
    
    std::cout << "\nADAM Performance:\n";
    std::cout << "Total time: " << total_time << " ms\n";
    std::cout << "Average step time: " << avg_step_time << " ms\n";
    std::cout << "Parameters/sec: " << (num_parameters * num_steps) / (total_time / 1000.0f) << std::endl;
    
    // Test AdamW
    cudaMemset(d_m, 0, num_parameters * sizeof(float));
    cudaMemset(d_v, 0, num_parameters * sizeof(float));
    
    auto adamw_start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= 10; step++) {
        initialize_random(h_gradients.data(), num_parameters, 0.0f, 0.01f);
        cudaMemcpy(d_gradients, h_gradients.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
        
        adamw_optimizer<<<grid, block>>>(d_weights, d_gradients, d_m, d_v,
                                        learning_rate, beta1, beta2, epsilon, weight_decay, 
                                        step, num_parameters);
    }
    cudaDeviceSynchronize();
    auto adamw_end = std::chrono::high_resolution_clock::now();
    auto adamw_time = std::chrono::duration<float, std::milli>(adamw_end - adamw_start).count();
    
    // Test RMSprop
    cudaMemset(d_v, 0, num_parameters * sizeof(float));
    
    auto rmsprop_start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= 10; step++) {
        initialize_random(h_gradients.data(), num_parameters, 0.0f, 0.01f);
        cudaMemcpy(d_gradients, h_gradients.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
        
        rmsprop_optimizer<<<grid, block>>>(d_weights, d_gradients, d_v,
                                          learning_rate, 0.9f, epsilon, num_parameters);
    }
    cudaDeviceSynchronize();
    auto rmsprop_end = std::chrono::high_resolution_clock::now();
    auto rmsprop_time = std::chrono::duration<float, std::milli>(rmsprop_end - rmsprop_start).count();
    
    std::cout << "\nOptimizer Comparison (10 steps):\n";
    std::cout << "AdamW: " << adamw_time << " ms\n";
    std::cout << "RMSprop: " << rmsprop_time << " ms\n";
    
    // Test Lookahead (apply every k steps)
    auto lookahead_start = std::chrono::high_resolution_clock::now();
    for (int step = 1; step <= 20; step++) {
        initialize_random(h_gradients.data(), num_parameters, 0.0f, 0.01f);
        cudaMemcpy(d_gradients, h_gradients.data(), num_parameters * sizeof(float), cudaMemcpyHostToDevice);
        
        adam_optimizer<<<grid, block>>>(d_weights, d_gradients, d_m, d_v,
                                       learning_rate, beta1, beta2, epsilon, step, num_parameters);
        
        if (step % lookahead_k == 0) {
            lookahead_update<<<grid, block>>>(d_weights, d_slow_weights, lookahead_alpha, num_parameters);
        }
    }
    cudaDeviceSynchronize();
    auto lookahead_end = std::chrono::high_resolution_clock::now();
    auto lookahead_time = std::chrono::duration<float, std::milli>(lookahead_end - lookahead_start).count();
    
    std::cout << "Lookahead (20 steps): " << lookahead_time << " ms\n";
    
    // Copy final weights back to verify
    cudaMemcpy(h_weights.data(), d_weights, num_parameters * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate weight statistics
    float weight_mean = 0.0f, weight_abs_mean = 0.0f;
    for (int i = 0; i < num_parameters; i++) {
        weight_mean += h_weights[i];
        weight_abs_mean += std::abs(h_weights[i]);
    }
    weight_mean /= num_parameters;
    weight_abs_mean /= num_parameters;
    
    std::cout << "\nFinal Weight Statistics:\n";
    std::cout << "Mean: " << weight_mean << std::endl;
    std::cout << "Mean Absolute: " << weight_abs_mean << std::endl;
    std::cout << "First few weights: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_weights[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_weights); cudaFree(d_gradients); cudaFree(d_m); cudaFree(d_v);
    cudaFree(d_slow_weights); cudaFree(d_gradients_fp16); cudaFree(d_gradients_fp32);
    cudaFree(d_partial_norms);
    
    return 0;
}