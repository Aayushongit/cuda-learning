#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Simple fully connected layer forward pass
__global__ void fc_forward(const float* input, const float* weights, const float* bias,
                          float* output, int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && output_idx < output_size) {
        float sum = bias[output_idx];
        for (int i = 0; i < input_size; i++) {
            sum += input[batch_idx * input_size + i] * weights[i * output_size + output_idx];
        }
        output[batch_idx * output_size + output_idx] = fmaxf(0.0f, sum); // ReLU
    }
}

// Fully connected layer backward pass - compute gradients w.r.t. weights
__global__ void fc_backward_weights(const float* input, const float* grad_output,
                                   float* grad_weights, int batch_size, int input_size, int output_size) {
    int input_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (input_idx < input_size && output_idx < output_size) {
        float grad_w = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            float input_val = input[b * input_size + input_idx];
            float grad_out_val = grad_output[b * output_size + output_idx];
            // Only propagate if ReLU was active (assuming forward pass stored activations)
            grad_w += input_val * grad_out_val;
        }
        grad_weights[input_idx * output_size + output_idx] = grad_w / batch_size;
    }
}

// Fully connected layer backward pass - compute gradients w.r.t. bias
__global__ void fc_backward_bias(const float* grad_output, float* grad_bias,
                                int batch_size, int output_size) {
    int output_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (output_idx < output_size) {
        float grad_b = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            grad_b += grad_output[b * output_size + output_idx];
        }
        grad_bias[output_idx] = grad_b / batch_size;
    }
}

// Fully connected layer backward pass - compute gradients w.r.t. input
__global__ void fc_backward_input(const float* weights, const float* grad_output, const float* forward_output,
                                 float* grad_input, int batch_size, int input_size, int output_size) {
    int batch_idx = blockIdx.y;
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && input_idx < input_size) {
        float grad_in = 0.0f;
        for (int o = 0; o < output_size; o++) {
            float grad_out_val = grad_output[batch_idx * output_size + o];
            // Apply ReLU derivative (gradient flows only if forward output > 0)
            if (forward_output[batch_idx * output_size + o] > 0.0f) {
                grad_in += weights[input_idx * output_size + o] * grad_out_val;
            }
        }
        grad_input[batch_idx * input_size + input_idx] = grad_in;
    }
}

// Mean Squared Error loss forward pass
__global__ void mse_loss_forward(const float* predictions, const float* targets, float* loss,
                                int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    
    extern __shared__ float sdata[];
    
    float local_loss = 0.0f;
    if (idx < total_elements) {
        float diff = predictions[idx] - targets[idx];
        local_loss = diff * diff;
    }
    
    sdata[threadIdx.x] = local_loss;
    __syncthreads();
    
    // Reduction to compute total loss
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(loss, sdata[0] / (2.0f * batch_size));
    }
}

// Mean Squared Error loss backward pass
__global__ void mse_loss_backward(const float* predictions, const float* targets, float* grad_output,
                                 int batch_size, int output_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * output_size;
    
    if (idx < total_elements) {
        grad_output[idx] = (predictions[idx] - targets[idx]) / batch_size;
    }
}

// Cross-entropy loss forward pass (for classification)
__global__ void cross_entropy_loss_forward(const float* log_probs, const int* targets, float* loss,
                                          int batch_size, int num_classes) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int target_class = targets[batch_idx];
        float neg_log_prob = -log_probs[batch_idx * num_classes + target_class];
        atomicAdd(loss, neg_log_prob / batch_size);
    }
}

// Cross-entropy loss backward pass
__global__ void cross_entropy_loss_backward(const float* softmax_probs, const int* targets,
                                           float* grad_output, int batch_size, int num_classes) {
    int batch_idx = blockIdx.y;
    int class_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        int target_class = targets[batch_idx];
        float grad = softmax_probs[batch_idx * num_classes + class_idx];
        if (class_idx == target_class) {
            grad -= 1.0f;
        }
        grad_output[batch_idx * num_classes + class_idx] = grad / batch_size;
    }
}

// Multi-layer perceptron forward pass
class MLPLayer {
public:
    int input_size, output_size, batch_size;
    float *d_weights, *d_bias, *d_output, *d_activations;
    
    MLPLayer(int in_size, int out_size, int batch) 
        : input_size(in_size), output_size(out_size), batch_size(batch) {
        
        cudaMalloc(&d_weights, input_size * output_size * sizeof(float));
        cudaMalloc(&d_bias, output_size * sizeof(float));
        cudaMalloc(&d_output, batch_size * output_size * sizeof(float));
        cudaMalloc(&d_activations, batch_size * output_size * sizeof(float));
        
        // Initialize weights with Xavier initialization
        std::vector<float> h_weights(input_size * output_size);
        std::vector<float> h_bias(output_size, 0.0f);
        
        std::random_device rd;
        std::mt19937 gen(rd());
        float xavier_std = sqrtf(2.0f / (input_size + output_size));
        std::normal_distribution<float> dis(0.0f, xavier_std);
        
        for (int i = 0; i < input_size * output_size; i++) {
            h_weights[i] = dis(gen);
        }
        
        cudaMemcpy(d_weights, h_weights.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bias, h_bias.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ~MLPLayer() {
        cudaFree(d_weights);
        cudaFree(d_bias);
        cudaFree(d_output);
        cudaFree(d_activations);
    }
    
    void forward(const float* input) {
        dim3 block(16, 16);
        dim3 grid((output_size + block.x - 1) / block.x, (batch_size + block.y - 1) / block.y);
        
        fc_forward<<<grid, block>>>(input, d_weights, d_bias, d_activations, 
                                   batch_size, input_size, output_size);
        
        // Store pre-activation for backward pass
        cudaMemcpy(d_output, d_activations, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    void backward(const float* input, const float* grad_output, 
                 float* grad_weights, float* grad_bias, float* grad_input) {
        
        dim3 block_weights(16, 16);
        dim3 grid_weights((output_size + block_weights.x - 1) / block_weights.x,
                         (input_size + block_weights.y - 1) / block_weights.y);
        
        fc_backward_weights<<<grid_weights, block_weights>>>(
            input, grad_output, grad_weights, batch_size, input_size, output_size);
        
        dim3 block_bias(256);
        dim3 grid_bias((output_size + block_bias.x - 1) / block_bias.x);
        
        fc_backward_bias<<<grid_bias, block_bias>>>(grad_output, grad_bias, batch_size, output_size);
        
        dim3 block_input(256);
        dim3 grid_input((input_size + block_input.x - 1) / block_input.x, batch_size);
        
        fc_backward_input<<<grid_input, block_input>>>(
            d_weights, grad_output, d_output, grad_input, batch_size, input_size, output_size);
    }
};

void initialize_random(float* data, int size, float mean = 0.0f, float std = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(mean, std);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main() {
    const int batch_size = 64;
    const int input_size = 784;  // 28x28 image
    const int hidden_size = 256;
    const int output_size = 10;  // 10 classes
    
    // Create MLP layers
    MLPLayer layer1(input_size, hidden_size, batch_size);
    MLPLayer layer2(hidden_size, output_size, batch_size);
    
    // Host memory
    std::vector<float> h_input(batch_size * input_size);
    std::vector<float> h_targets(batch_size * output_size);
    std::vector<int> h_target_labels(batch_size);
    
    initialize_random(h_input.data(), batch_size * input_size, 0.0f, 1.0f);
    
    // Create one-hot targets
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> label_dis(0, output_size - 1);
    
    for (int i = 0; i < batch_size; i++) {
        int label = label_dis(gen);
        h_target_labels[i] = label;
        for (int j = 0; j < output_size; j++) {
            h_targets[i * output_size + j] = (j == label) ? 1.0f : 0.0f;
        }
    }
    
    // Device memory
    float *d_input, *d_targets, *d_hidden_output, *d_final_output;
    int *d_target_labels;
    float *d_loss;
    
    cudaMalloc(&d_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_targets, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_target_labels, batch_size * sizeof(int));
    cudaMalloc(&d_hidden_output, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_final_output, batch_size * output_size * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));
    
    cudaMemcpy(d_input, h_input.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets.data(), batch_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target_labels, h_target_labels.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    
    // Gradient storage
    float *d_grad_weights1, *d_grad_bias1, *d_grad_weights2, *d_grad_bias2;
    float *d_grad_hidden, *d_grad_input, *d_grad_output;
    
    cudaMalloc(&d_grad_weights1, input_size * hidden_size * sizeof(float));
    cudaMalloc(&d_grad_bias1, hidden_size * sizeof(float));
    cudaMalloc(&d_grad_weights2, hidden_size * output_size * sizeof(float));
    cudaMalloc(&d_grad_bias2, output_size * sizeof(float));
    cudaMalloc(&d_grad_hidden, batch_size * hidden_size * sizeof(float));
    cudaMalloc(&d_grad_input, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_grad_output, batch_size * output_size * sizeof(float));
    
    std::cout << "Running Forward and Backward Pass Benchmark\n";
    std::cout << "Network: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    
    // Benchmark forward pass
    auto start = std::chrono::high_resolution_clock::now();
    
    // Forward pass
    layer1.forward(d_input);
    layer2.forward(layer1.d_activations);
    
    // Compute loss
    cudaMemset(d_loss, 0, sizeof(float));
    dim3 loss_block(256);
    dim3 loss_grid((batch_size * output_size + loss_block.x - 1) / loss_block.x);
    
    mse_loss_forward<<<loss_grid, loss_block, loss_block.x * sizeof(float)>>>(
        layer2.d_activations, d_targets, d_loss, batch_size, output_size);
    
    cudaDeviceSynchronize();
    auto forward_end = std::chrono::high_resolution_clock::now();
    
    // Backward pass
    auto backward_start = std::chrono::high_resolution_clock::now();
    
    // Loss gradient
    mse_loss_backward<<<loss_grid, loss_block>>>(
        layer2.d_activations, d_targets, d_grad_output, batch_size, output_size);
    
    // Layer 2 backward
    layer2.backward(layer1.d_activations, d_grad_output, d_grad_weights2, d_grad_bias2, d_grad_hidden);
    
    // Layer 1 backward
    layer1.backward(d_input, d_grad_hidden, d_grad_weights1, d_grad_bias1, d_grad_input);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    // Calculate timings
    auto forward_time = std::chrono::duration<float, std::milli>(forward_end - start).count();
    auto backward_time = std::chrono::duration<float, std::milli>(end - backward_start).count();
    auto total_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Get loss value
    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Forward Pass: " << forward_time << " ms\n";
    std::cout << "Backward Pass: " << backward_time << " ms\n";
    std::cout << "Total Time: " << total_time << " ms\n";
    std::cout << "Loss: " << h_loss << std::endl;
    
    // Verify gradients by checking some statistics
    std::vector<float> h_grad_weights1(input_size * hidden_size);
    cudaMemcpy(h_grad_weights1.data(), d_grad_weights1, input_size * hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float grad_sum = 0.0f, grad_abs_sum = 0.0f;
    for (int i = 0; i < input_size * hidden_size; i++) {
        grad_sum += h_grad_weights1[i];
        grad_abs_sum += std::abs(h_grad_weights1[i]);
    }
    
    std::cout << "Gradient Statistics (Layer 1 weights):\n";
    std::cout << "Mean: " << grad_sum / (input_size * hidden_size) << std::endl;
    std::cout << "Mean Absolute: " << grad_abs_sum / (input_size * hidden_size) << std::endl;
    
    // Cleanup
    cudaFree(d_input); cudaFree(d_targets); cudaFree(d_target_labels);
    cudaFree(d_hidden_output); cudaFree(d_final_output); cudaFree(d_loss);
    cudaFree(d_grad_weights1); cudaFree(d_grad_bias1);
    cudaFree(d_grad_weights2); cudaFree(d_grad_bias2);
    cudaFree(d_grad_hidden); cudaFree(d_grad_input); cudaFree(d_grad_output);
    
    return 0;
}
