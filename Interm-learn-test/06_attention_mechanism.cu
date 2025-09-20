#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>

// Scaled dot-product attention kernel
__global__ void scaled_dot_product_attention(const float* query, const float* key, const float* value,
                                           float* output, float* attention_weights,
                                           int batch_size, int seq_length, int d_model, int d_k,
                                           float scale_factor, const float* mask = nullptr) {
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_i = blockIdx.x * blockDim.x + threadIdx.x;
    int seq_j = threadIdx.y;
    
    if (batch_idx >= batch_size || head_idx >= 1 || seq_i >= seq_length) return;
    
    extern __shared__ float sdata[];
    float* s_scores = sdata;
    float* s_values = sdata + seq_length;
    
    // Compute attention scores: Q * K^T
    if (seq_j < seq_length) {
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            int q_idx = ((batch_idx * seq_length + seq_i) * d_model + d);
            int k_idx = ((batch_idx * seq_length + seq_j) * d_model + d);
            score += query[q_idx] * key[k_idx];
        }
        score *= scale_factor;
        
        // Apply mask if provided
        if (mask != nullptr && mask[seq_i * seq_length + seq_j] == 0.0f) {
            score = -INFINITY;
        }
        
        s_scores[seq_j] = score;
    }
    
    __syncthreads();
    
    // Apply softmax to get attention weights
    float max_score = -INFINITY;
    for (int j = 0; j < seq_length; j++) {
        max_score = fmaxf(max_score, s_scores[j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_length; j++) {
        s_scores[j] = expf(s_scores[j] - max_score);
        sum_exp += s_scores[j];
    }
    
    for (int j = 0; j < seq_length; j++) {
        s_scores[j] /= sum_exp;
        if (seq_j == j) {
            attention_weights[batch_idx * seq_length * seq_length + seq_i * seq_length + j] = s_scores[j];
        }
    }
    
    __syncthreads();
    
    // Compute weighted sum of values
    for (int d = threadIdx.y; d < d_model; d += blockDim.y) {
        float output_val = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            int v_idx = ((batch_idx * seq_length + j) * d_model + d);
            output_val += s_scores[j] * value[v_idx];
        }
        
        int out_idx = ((batch_idx * seq_length + seq_i) * d_model + d);
        output[out_idx] = output_val;
    }
}

// Multi-head attention with separate Q, K, V projections
__global__ void multi_head_attention(const float* input, const float* W_q, const float* W_k, const float* W_v,
                                    const float* W_o, float* output, float* attention_weights,
                                    int batch_size, int seq_length, int d_model, int num_heads,
                                    const float* mask = nullptr) {
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int seq_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads || seq_i >= seq_length) return;
    
    int d_k = d_model / num_heads;
    float scale_factor = 1.0f / sqrtf(float(d_k));
    
    extern __shared__ float shared_memory[];
    float* s_q = shared_memory;
    float* s_k = s_q + d_k;
    float* s_v = s_k + d_k;
    float* s_scores = s_v + d_k;
    
    // Project input to Q, K, V for this head
    for (int d = threadIdx.y; d < d_k; d += blockDim.y) {
        float q_val = 0.0f, k_val = 0.0f, v_val = 0.0f;
        
        for (int i = 0; i < d_model; i++) {
            int input_idx = batch_idx * seq_length * d_model + seq_i * d_model + i;
            int q_weight_idx = head_idx * d_model * d_k + i * d_k + d;
            int k_weight_idx = head_idx * d_model * d_k + i * d_k + d;
            int v_weight_idx = head_idx * d_model * d_k + i * d_k + d;
            
            q_val += input[input_idx] * W_q[q_weight_idx];
            k_val += input[input_idx] * W_k[k_weight_idx];
            v_val += input[input_idx] * W_v[v_weight_idx];
        }
        
        s_q[d] = q_val;
        s_k[d] = k_val;
        s_v[d] = v_val;
    }
    
    __syncthreads();
    
    // Compute attention scores for each position
    for (int seq_j = 0; seq_j < seq_length; seq_j++) {
        // Load key for position seq_j
        float* s_k_j = s_scores + seq_j * d_k; // Reuse part of shared memory
        
        for (int d = threadIdx.y; d < d_k; d += blockDim.y) {
            float k_val = 0.0f;
            for (int i = 0; i < d_model; i++) {
                int input_idx = batch_idx * seq_length * d_model + seq_j * d_model + i;
                int k_weight_idx = head_idx * d_model * d_k + i * d_k + d;
                k_val += input[input_idx] * W_k[k_weight_idx];
            }
            s_k_j[d] = k_val;
        }
        
        __syncthreads();
        
        // Compute dot product
        float score = 0.0f;
        for (int d = 0; d < d_k; d++) {
            score += s_q[d] * s_k_j[d];
        }
        score *= scale_factor;
        
        // Apply mask
        if (mask != nullptr && mask[seq_i * seq_length + seq_j] == 0.0f) {
            score = -INFINITY;
        }
        
        s_scores[seq_j] = score;
    }
    
    // Apply softmax
    float max_score = -INFINITY;
    for (int j = 0; j < seq_length; j++) {
        max_score = fmaxf(max_score, s_scores[j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_length; j++) {
        s_scores[j] = expf(s_scores[j] - max_score);
        sum_exp += s_scores[j];
    }
    
    for (int j = 0; j < seq_length; j++) {
        s_scores[j] /= sum_exp;
        // Store attention weights
        attention_weights[((batch_idx * num_heads + head_idx) * seq_length + seq_i) * seq_length + j] = s_scores[j];
    }
    
    // Compute attention output for this head
    for (int d = threadIdx.y; d < d_k; d += blockDim.y) {
        float head_output = 0.0f;
        
        for (int seq_j = 0; seq_j < seq_length; seq_j++) {
            // Load value for position seq_j
            float v_val = 0.0f;
            for (int i = 0; i < d_model; i++) {
                int input_idx = batch_idx * seq_length * d_model + seq_j * d_model + i;
                int v_weight_idx = head_idx * d_model * d_k + i * d_k + d;
                v_val += input[input_idx] * W_v[v_weight_idx];
            }
            
            head_output += s_scores[seq_j] * v_val;
        }
        
        // Store head output (will be concatenated later)
        int head_out_idx = ((batch_idx * seq_length + seq_i) * num_heads + head_idx) * d_k + d;
        // This is a simplified version - in practice, you'd store in temp memory first
        // then apply final projection W_o
    }
}

// Flash Attention - memory efficient attention
__global__ void flash_attention_forward(const float* Q, const float* K, const float* V,
                                      float* O, float* L, float* M,
                                      int batch_size, int num_heads, int seq_length, int d_k,
                                      int block_size_M, int block_size_N) {
    
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int block_i = blockIdx.x;
    
    if (batch_idx >= batch_size || head_idx >= num_heads) return;
    
    extern __shared__ float sram[];
    float* s_Q = sram;
    float* s_K = s_Q + block_size_M * d_k;
    float* s_V = s_K + block_size_N * d_k;
    float* s_S = s_V + block_size_N * d_k;
    
    int start_i = block_i * block_size_M;
    int end_i = min(start_i + block_size_M, seq_length);
    
    float scale = 1.0f / sqrtf(float(d_k));
    
    // Initialize output accumulator
    for (int i = threadIdx.x; i < (end_i - start_i) * d_k; i += blockDim.x) {
        int local_i = i / d_k;
        int local_d = i % d_k;
        int global_i = start_i + local_i;
        
        if (global_i < seq_length) {
            int out_idx = ((batch_idx * num_heads + head_idx) * seq_length + global_i) * d_k + local_d;
            O[out_idx] = 0.0f;
            if (local_d == 0) {
                L[out_idx / d_k] = 0.0f;
                M[out_idx / d_k] = -INFINITY;
            }
        }
    }
    
    // Load Q block
    for (int i = threadIdx.x; i < (end_i - start_i) * d_k; i += blockDim.x) {
        int local_i = i / d_k;
        int local_d = i % d_k;
        int global_i = start_i + local_i;
        
        if (global_i < seq_length) {
            int q_idx = ((batch_idx * num_heads + head_idx) * seq_length + global_i) * d_k + local_d;
            s_Q[local_i * d_k + local_d] = Q[q_idx];
        }
    }
    
    __syncthreads();
    
    // Process K, V blocks
    for (int block_j = 0; block_j < (seq_length + block_size_N - 1) / block_size_N; block_j++) {
        int start_j = block_j * block_size_N;
        int end_j = min(start_j + block_size_N, seq_length);
        
        // Load K and V blocks
        for (int i = threadIdx.x; i < (end_j - start_j) * d_k; i += blockDim.x) {
            int local_j = i / d_k;
            int local_d = i % d_k;
            int global_j = start_j + local_j;
            
            if (global_j < seq_length) {
                int kv_idx = ((batch_idx * num_heads + head_idx) * seq_length + global_j) * d_k + local_d;
                s_K[local_j * d_k + local_d] = K[kv_idx];
                s_V[local_j * d_k + local_d] = V[kv_idx];
            }
        }
        
        __syncthreads();
        
        // Compute attention scores S = Q @ K^T
        for (int i = threadIdx.x; i < (end_i - start_i) * (end_j - start_j); i += blockDim.x) {
            int local_i = i / (end_j - start_j);
            int local_j = i % (end_j - start_j);
            
            float score = 0.0f;
            for (int d = 0; d < d_k; d++) {
                score += s_Q[local_i * d_k + d] * s_K[local_j * d_k + d];
            }
            s_S[local_i * block_size_N + local_j] = score * scale;
        }
        
        __syncthreads();
        
        // Update output using online softmax algorithm
        for (int local_i = 0; local_i < end_i - start_i; local_i++) {
            int global_i = start_i + local_i;
            if (global_i >= seq_length) continue;
            
            // Find row maximum
            float row_max = -INFINITY;
            for (int local_j = 0; local_j < end_j - start_j; local_j++) {
                row_max = fmaxf(row_max, s_S[local_i * block_size_N + local_j]);
            }
            
            // Compute row sum
            float row_sum = 0.0f;
            for (int local_j = 0; local_j < end_j - start_j; local_j++) {
                s_S[local_i * block_size_N + local_j] = expf(s_S[local_i * block_size_N + local_j] - row_max);
                row_sum += s_S[local_i * block_size_N + local_j];
            }
            
            // Online update of output
            int base_idx = ((batch_idx * num_heads + head_idx) * seq_length + global_i);
            float old_M = M[base_idx];
            float new_M = fmaxf(old_M, row_max);
            float old_L = L[base_idx];
            float new_L = expf(old_M - new_M) * old_L + expf(row_max - new_M) * row_sum;
            
            for (int d = 0; d < d_k; d++) {
                float old_O = O[base_idx * d_k + d];
                float new_O_part = 0.0f;
                
                for (int local_j = 0; local_j < end_j - start_j; local_j++) {
                    new_O_part += s_S[local_i * block_size_N + local_j] * s_V[local_j * d_k + d];
                }
                
                O[base_idx * d_k + d] = (expf(old_M - new_M) * old_L * old_O + 
                                        expf(row_max - new_M) * new_O_part) / new_L;
            }
            
            M[base_idx] = new_M;
            L[base_idx] = new_L;
        }
        
        __syncthreads();
    }
}

// Relative positional encoding attention
__global__ void relative_position_attention(const float* query, const float* key, const float* value,
                                           const float* pos_encodings, float* output,
                                           int batch_size, int seq_length, int d_model,
                                           int max_relative_position) {
    
    int batch_idx = blockIdx.z;
    int seq_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size || seq_i >= seq_length) return;
    
    extern __shared__ float s_scores[];
    
    // Compute attention scores with relative position bias
    for (int seq_j = 0; seq_j < seq_length; seq_j++) {
        float score = 0.0f;
        
        // Standard attention score
        for (int d = 0; d < d_model; d++) {
            int q_idx = batch_idx * seq_length * d_model + seq_i * d_model + d;
            int k_idx = batch_idx * seq_length * d_model + seq_j * d_model + d;
            score += query[q_idx] * key[k_idx];
        }
        
        // Add relative position bias
        int relative_pos = seq_j - seq_i;
        int clipped_pos = max(-max_relative_position, min(max_relative_position, relative_pos));
        int pos_idx = clipped_pos + max_relative_position; // Shift to positive index
        
        score += pos_encodings[pos_idx];
        s_scores[seq_j] = score / sqrtf(float(d_model));
    }
    
    __syncthreads();
    
    // Apply softmax
    float max_score = -INFINITY;
    for (int j = 0; j < seq_length; j++) {
        max_score = fmaxf(max_score, s_scores[j]);
    }
    
    float sum_exp = 0.0f;
    for (int j = 0; j < seq_length; j++) {
        s_scores[j] = expf(s_scores[j] - max_score);
        sum_exp += s_scores[j];
    }
    
    // Compute weighted sum
    for (int d = 0; d < d_model; d++) {
        float output_val = 0.0f;
        for (int j = 0; j < seq_length; j++) {
            float weight = s_scores[j] / sum_exp;
            int v_idx = batch_idx * seq_length * d_model + j * d_model + d;
            output_val += weight * value[v_idx];
        }
        
        int out_idx = batch_idx * seq_length * d_model + seq_i * d_model + d;
        output[out_idx] = output_val;
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
    // Attention parameters
    const int batch_size = 4;
    const int seq_length = 128;
    const int d_model = 512;
    const int num_heads = 8;
    const int d_k = d_model / num_heads;
    const int max_relative_position = 32;
    
    // Memory sizes
    int qkv_size = batch_size * seq_length * d_model;
    int attention_weights_size = batch_size * seq_length * seq_length;
    int multi_head_weights_size = batch_size * num_heads * seq_length * seq_length;
    
    // Host memory
    std::vector<float> h_query(qkv_size), h_key(qkv_size), h_value(qkv_size);
    std::vector<float> h_output(qkv_size);
    std::vector<float> h_attention_weights(attention_weights_size);
    std::vector<float> h_pos_encodings(2 * max_relative_position + 1);
    
    initialize_random(h_query.data(), qkv_size, 0.0f, 1.0f / sqrtf(d_model));
    initialize_random(h_key.data(), qkv_size, 0.0f, 1.0f / sqrtf(d_model));
    initialize_random(h_value.data(), qkv_size, 0.0f, 1.0f / sqrtf(d_model));
    initialize_random(h_pos_encodings.data(), 2 * max_relative_position + 1, 0.0f, 0.1f);
    
    // Device memory
    float *d_query, *d_key, *d_value, *d_output, *d_attention_weights, *d_pos_encodings;
    float *d_L, *d_M; // For Flash Attention
    
    cudaMalloc(&d_query, qkv_size * sizeof(float));
    cudaMalloc(&d_key, qkv_size * sizeof(float));
    cudaMalloc(&d_value, qkv_size * sizeof(float));
    cudaMalloc(&d_output, qkv_size * sizeof(float));
    cudaMalloc(&d_attention_weights, attention_weights_size * sizeof(float));
    cudaMalloc(&d_pos_encodings, (2 * max_relative_position + 1) * sizeof(float));
    cudaMalloc(&d_L, batch_size * seq_length * sizeof(float));
    cudaMalloc(&d_M, batch_size * seq_length * sizeof(float));
    
    cudaMemcpy(d_query, h_query.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_key, h_key.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, h_value.data(), qkv_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_encodings, h_pos_encodings.data(), (2 * max_relative_position + 1) * sizeof(float), cudaMemcpyHostToDevice);
    
    std::cout << "Attention Mechanism Benchmark\n";
    std::cout << "Batch size: " << batch_size << ", Seq length: " << seq_length;
    std::cout << ", d_model: " << d_model << ", Num heads: " << num_heads << std::endl;
    
    // Test standard scaled dot-product attention
    dim3 block(32, min(32, seq_length));
    dim3 grid((seq_length + block.x - 1) / block.x, 1, batch_size);
    int shared_mem_size = 2 * seq_length * sizeof(float);
    
    auto start = std::chrono::high_resolution_clock::now();
    scaled_dot_product_attention<<<grid, block, shared_mem_size>>>(
        d_query, d_key, d_value, d_output, d_attention_weights,
        batch_size, seq_length, d_model, d_k, 1.0f / sqrtf(d_k));
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto attention_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test Flash Attention
    const int block_size_M = 32, block_size_N = 32;
    dim3 flash_block(min(256, block_size_M * d_k));
    dim3 flash_grid((seq_length + block_size_M - 1) / block_size_M, num_heads, batch_size);
    int flash_shared_mem = (block_size_M * d_k + 2 * block_size_N * d_k + block_size_M * block_size_N) * sizeof(float);
    
    start = std::chrono::high_resolution_clock::now();
    flash_attention_forward<<<flash_grid, flash_block, flash_shared_mem>>>(
        d_query, d_key, d_value, d_output, d_L, d_M,
        batch_size, num_heads, seq_length, d_k, block_size_M, block_size_N);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto flash_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test relative position attention
    dim3 rel_block(min(128, seq_length));
    dim3 rel_grid((seq_length + rel_block.x - 1) / rel_block.x, 1, batch_size);
    int rel_shared_mem = seq_length * sizeof(float);
    
    start = std::chrono::high_resolution_clock::now();
    relative_position_attention<<<rel_grid, rel_block, rel_shared_mem>>>(
        d_query, d_key, d_value, d_pos_encodings, d_output,
        batch_size, seq_length, d_model, max_relative_position);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto rel_pos_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back
    cudaMemcpy(h_output.data(), d_output, qkv_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_attention_weights.data(), d_attention_weights, attention_weights_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate statistics
    float output_mean = 0.0f, attention_entropy = 0.0f;
    for (int i = 0; i < qkv_size; i++) {
        output_mean += h_output[i];
    }
    output_mean /= qkv_size;
    
    // Calculate attention entropy (first batch, first sequence position)
    for (int j = 0; j < seq_length; j++) {
        float p = h_attention_weights[j];
        if (p > 1e-10f) {
            attention_entropy -= p * logf(p);
        }
    }
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Scaled Dot-Product Attention: " << attention_time << " ms\n";
    std::cout << "Flash Attention: " << flash_time << " ms\n";
    std::cout << "Relative Position Attention: " << rel_pos_time << " ms\n";
    
    std::cout << "\nOutput Statistics:\n";
    std::cout << "Output mean: " << output_mean << std::endl;
    std::cout << "Attention entropy (first position): " << attention_entropy << std::endl;
    std::cout << "First few attention weights: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_attention_weights[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_query); cudaFree(d_key); cudaFree(d_value);
    cudaFree(d_output); cudaFree(d_attention_weights); cudaFree(d_pos_encodings);
    cudaFree(d_L); cudaFree(d_M);
    
    return 0;
}