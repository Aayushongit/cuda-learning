#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <algorithm>

// Basic embedding lookup kernel
__global__ void embedding_lookup(const float* embedding_table, const int* indices,
                                float* output, int batch_size, int seq_length,
                                int vocab_size, int embedding_dim) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    int token_id = indices[batch_idx * seq_length + seq_idx];
    
    // Bounds check for token_id
    if (token_id >= 0 && token_id < vocab_size) {
        int embedding_idx = token_id * embedding_dim + dim_idx;
        int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
        
        output[output_idx] = embedding_table[embedding_idx];
    } else {
        // Handle out-of-bounds tokens (set to zero or use UNK token)
        int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
        output[output_idx] = 0.0f;
    }
}

// Embedding lookup with positional encoding
__global__ void embedding_with_positional(const float* embedding_table, const float* pos_encodings,
                                         const int* indices, float* output,
                                         int batch_size, int seq_length, int vocab_size,
                                         int embedding_dim, int max_seq_length) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    int token_id = indices[batch_idx * seq_length + seq_idx];
    int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
    
    float embed_val = 0.0f;
    if (token_id >= 0 && token_id < vocab_size) {
        int embedding_idx = token_id * embedding_dim + dim_idx;
        embed_val = embedding_table[embedding_idx];
    }
    
    // Add positional encoding
    float pos_val = 0.0f;
    if (seq_idx < max_seq_length) {
        int pos_idx = seq_idx * embedding_dim + dim_idx;
        pos_val = pos_encodings[pos_idx];
    }
    
    output[output_idx] = embed_val + pos_val;
}

// Embedding lookup backward pass (gradient accumulation)
__global__ void embedding_backward(const float* grad_output, const int* indices,
                                  float* grad_embedding_table, int batch_size, int seq_length,
                                  int vocab_size, int embedding_dim) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    int token_id = indices[batch_idx * seq_length + seq_idx];
    
    if (token_id >= 0 && token_id < vocab_size) {
        int grad_output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
        int embedding_idx = token_id * embedding_dim + dim_idx;
        
        // Accumulate gradients using atomic operations
        atomicAdd(&grad_embedding_table[embedding_idx], grad_output[grad_output_idx]);
    }
}

// Learned positional embedding lookup
__global__ void positional_embedding_lookup(const float* pos_embedding_table, float* output,
                                           int batch_size, int seq_length, int embedding_dim,
                                           int max_positions) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    // Use position index directly (could also add offset for absolute positions)
    int pos_id = min(seq_idx, max_positions - 1);
    
    int pos_embedding_idx = pos_id * embedding_dim + dim_idx;
    int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
    
    output[output_idx] = pos_embedding_table[pos_embedding_idx];
}

// Sinusoidal positional encoding generation (for Transformer models)
__global__ void generate_sinusoidal_positions(float* pos_encodings, int max_seq_length,
                                             int embedding_dim) {
    
    int pos = blockIdx.x;
    int dim = threadIdx.x;
    
    if (pos >= max_seq_length || dim >= embedding_dim) return;
    
    int output_idx = pos * embedding_dim + dim;
    
    if (dim % 2 == 0) {
        // Even dimensions use sine
        float angle = pos / powf(10000.0f, (2.0f * (dim / 2)) / embedding_dim);
        pos_encodings[output_idx] = sinf(angle);
    } else {
        // Odd dimensions use cosine
        float angle = pos / powf(10000.0f, (2.0f * ((dim - 1) / 2)) / embedding_dim);
        pos_encodings[output_idx] = cosf(angle);
    }
}

// Subword embedding with BPE/WordPiece support
__global__ void subword_embedding_lookup(const float* embedding_table, const int* token_ids,
                                        const int* token_lengths, float* output,
                                        int batch_size, int max_tokens, int vocab_size,
                                        int embedding_dim) {
    
    int batch_idx = blockIdx.y;
    int token_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || token_idx >= max_tokens || dim_idx >= embedding_dim) return;
    
    int base_token_idx = batch_idx * max_tokens;
    int token_length = token_lengths[batch_idx];
    
    if (token_idx >= token_length) {
        // Padding tokens
        int output_idx = (batch_idx * max_tokens + token_idx) * embedding_dim + dim_idx;
        output[output_idx] = 0.0f;
        return;
    }
    
    int token_id = token_ids[base_token_idx + token_idx];
    
    if (token_id >= 0 && token_id < vocab_size) {
        int embedding_idx = token_id * embedding_dim + dim_idx;
        int output_idx = (batch_idx * max_tokens + token_idx) * embedding_dim + dim_idx;
        output[output_idx] = embedding_table[embedding_idx];
    }
}

// Multi-lingual embedding with language-specific scaling
__global__ void multilingual_embedding_lookup(const float* embedding_table, const float* lang_scales,
                                             const int* token_ids, const int* lang_ids,
                                             float* output, int batch_size, int seq_length,
                                             int vocab_size, int embedding_dim, int num_languages) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    int token_id = token_ids[batch_idx * seq_length + seq_idx];
    int lang_id = lang_ids[batch_idx]; // Language ID per batch
    
    float embed_val = 0.0f;
    if (token_id >= 0 && token_id < vocab_size) {
        int embedding_idx = token_id * embedding_dim + dim_idx;
        embed_val = embedding_table[embedding_idx];
    }
    
    // Apply language-specific scaling
    float scale = 1.0f;
    if (lang_id >= 0 && lang_id < num_languages) {
        scale = lang_scales[lang_id * embedding_dim + dim_idx];
    }
    
    int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
    output[output_idx] = embed_val * scale;
}

// Cached embedding lookup for frequently used tokens
__global__ void cached_embedding_lookup(const float* embedding_table, const int* indices,
                                       float* output, int* cache_hits, int batch_size,
                                       int seq_length, int vocab_size, int embedding_dim) {
    
    int batch_idx = blockIdx.y;
    int seq_idx = blockIdx.x;
    int dim_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_length || dim_idx >= embedding_dim) return;
    
    int token_id = indices[batch_idx * seq_length + seq_idx];
    
    if (token_id >= 0 && token_id < vocab_size) {
        // Check if this is a frequent token (simplified cache logic)
        bool is_cached = (token_id < 1000); // Cache first 1000 tokens
        
        if (dim_idx == 0 && is_cached) {
            atomicAdd(&cache_hits[batch_idx * seq_length + seq_idx], 1);
        }
        
        int embedding_idx = token_id * embedding_dim + dim_idx;
        int output_idx = (batch_idx * seq_length + seq_idx) * embedding_dim + dim_idx;
        
        output[output_idx] = embedding_table[embedding_idx];
    }
}

// Dropout applied to embeddings
__global__ void embedding_dropout(float* embeddings, const float* dropout_mask, float dropout_prob,
                                 int batch_size, int seq_length, int embedding_dim) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = batch_size * seq_length * embedding_dim;
    
    if (idx < total_size) {
        float mask_val = dropout_mask[idx];
        if (mask_val < dropout_prob) {
            embeddings[idx] = 0.0f;
        } else {
            embeddings[idx] = embeddings[idx] / (1.0f - dropout_prob); // Scale by keep probability
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

void initialize_random_int(int* data, int size, int min_val, int max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(min_val, max_val - 1);
    
    for (int i = 0; i < size; i++) {
        data[i] = dis(gen);
    }
}

int main() {
    // Model parameters
    const int batch_size = 32;
    const int seq_length = 512;
    const int vocab_size = 50000;
    const int embedding_dim = 768;
    const int max_seq_length = 1024;
    const int max_positions = 1024;
    const int num_languages = 100;
    const float dropout_prob = 0.1f;
    
    // Memory sizes
    int embedding_table_size = vocab_size * embedding_dim;
    int pos_table_size = max_positions * embedding_dim;
    int indices_size = batch_size * seq_length;
    int output_size = batch_size * seq_length * embedding_dim;
    int pos_encodings_size = max_seq_length * embedding_dim;
    
    std::cout << "Embedding Lookup Benchmark\n";
    std::cout << "Vocab size: " << vocab_size << ", Embedding dim: " << embedding_dim << std::endl;
    std::cout << "Batch size: " << batch_size << ", Seq length: " << seq_length << std::endl;
    
    // Host memory
    std::vector<float> h_embedding_table(embedding_table_size);
    std::vector<float> h_pos_table(pos_table_size);
    std::vector<float> h_pos_encodings(pos_encodings_size);
    std::vector<int> h_indices(indices_size);
    std::vector<int> h_lang_ids(batch_size);
    std::vector<float> h_output(output_size);
    std::vector<float> h_lang_scales(num_languages * embedding_dim);
    std::vector<float> h_dropout_mask(output_size);
    
    // Initialize data
    float embed_std = 1.0f / sqrtf(embedding_dim);
    initialize_random(h_embedding_table.data(), embedding_table_size, 0.0f, embed_std);
    initialize_random(h_pos_table.data(), pos_table_size, 0.0f, embed_std);
    initialize_random_int(h_indices.data(), indices_size, 0, vocab_size);
    initialize_random_int(h_lang_ids.data(), batch_size, 0, num_languages);
    initialize_random(h_lang_scales.data(), num_languages * embedding_dim, 1.0f, 0.1f);
    initialize_random(h_dropout_mask.data(), output_size, 0.0f, 1.0f);
    
    // Device memory
    float *d_embedding_table, *d_pos_table, *d_pos_encodings, *d_output;
    float *d_lang_scales, *d_dropout_mask, *d_grad_embedding_table;
    int *d_indices, *d_lang_ids, *d_cache_hits;
    
    cudaMalloc(&d_embedding_table, embedding_table_size * sizeof(float));
    cudaMalloc(&d_pos_table, pos_table_size * sizeof(float));
    cudaMalloc(&d_pos_encodings, pos_encodings_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_indices, indices_size * sizeof(int));
    cudaMalloc(&d_lang_ids, batch_size * sizeof(int));
    cudaMalloc(&d_lang_scales, num_languages * embedding_dim * sizeof(float));
    cudaMalloc(&d_dropout_mask, output_size * sizeof(float));
    cudaMalloc(&d_grad_embedding_table, embedding_table_size * sizeof(float));
    cudaMalloc(&d_cache_hits, indices_size * sizeof(int));
    
    cudaMemcpy(d_embedding_table, h_embedding_table.data(), embedding_table_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_table, h_pos_table.data(), pos_table_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices.data(), indices_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang_ids, h_lang_ids.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_lang_scales, h_lang_scales.data(), num_languages * embedding_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dropout_mask, h_dropout_mask.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Generate sinusoidal positional encodings
    dim3 pos_block(min(embedding_dim, 256));
    dim3 pos_grid(max_seq_length);
    
    generate_sinusoidal_positions<<<pos_grid, pos_block>>>(d_pos_encodings, max_seq_length, embedding_dim);
    cudaDeviceSynchronize();
    
    // Kernel configurations
    dim3 block(min(embedding_dim, 256));
    dim3 grid(seq_length, batch_size);
    
    // Benchmark basic embedding lookup
    auto start = std::chrono::high_resolution_clock::now();
    embedding_lookup<<<grid, block>>>(d_embedding_table, d_indices, d_output,
                                     batch_size, seq_length, vocab_size, embedding_dim);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto basic_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Benchmark embedding with positional encoding
    start = std::chrono::high_resolution_clock::now();
    embedding_with_positional<<<grid, block>>>(d_embedding_table, d_pos_encodings, d_indices, d_output,
                                              batch_size, seq_length, vocab_size, embedding_dim, max_seq_length);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto pos_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Benchmark multi-lingual embedding
    start = std::chrono::high_resolution_clock::now();
    multilingual_embedding_lookup<<<grid, block>>>(d_embedding_table, d_lang_scales, d_indices, d_lang_ids, d_output,
                                                  batch_size, seq_length, vocab_size, embedding_dim, num_languages);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto multilingual_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Benchmark cached embedding lookup
    cudaMemset(d_cache_hits, 0, indices_size * sizeof(int));
    
    start = std::chrono::high_resolution_clock::now();
    cached_embedding_lookup<<<grid, block>>>(d_embedding_table, d_indices, d_output, d_cache_hits,
                                            batch_size, seq_length, vocab_size, embedding_dim);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto cached_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Benchmark embedding backward pass
    std::vector<float> h_grad_output(output_size);
    initialize_random(h_grad_output.data(), output_size, 0.0f, 0.1f);
    
    float *d_grad_output;
    cudaMalloc(&d_grad_output, output_size * sizeof(float));
    cudaMemcpy(d_grad_output, h_grad_output.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_grad_embedding_table, 0, embedding_table_size * sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    embedding_backward<<<grid, block>>>(d_grad_output, d_indices, d_grad_embedding_table,
                                       batch_size, seq_length, vocab_size, embedding_dim);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto backward_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Test dropout
    dim3 dropout_block(256);
    dim3 dropout_grid((output_size + dropout_block.x - 1) / dropout_block.x);
    
    start = std::chrono::high_resolution_clock::now();
    embedding_dropout<<<dropout_grid, dropout_block>>>(d_output, d_dropout_mask, dropout_prob,
                                                       batch_size, seq_length, embedding_dim);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto dropout_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back for verification
    cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Get cache hit statistics
    std::vector<int> h_cache_hits(indices_size);
    cudaMemcpy(h_cache_hits.data(), d_cache_hits, indices_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    int total_cache_hits = 0;
    for (int hit : h_cache_hits) {
        total_cache_hits += hit;
    }
    
    // Calculate throughput
    long long total_lookups = (long long)batch_size * seq_length;
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Basic Embedding Lookup: " << basic_time << " ms\n";
    std::cout << "Embedding + Positional: " << pos_time << " ms\n";
    std::cout << "Multi-lingual Embedding: " << multilingual_time << " ms\n";
    std::cout << "Cached Embedding: " << cached_time << " ms\n";
    std::cout << "Embedding Backward: " << backward_time << " ms\n";
    std::cout << "Embedding Dropout: " << dropout_time << " ms\n";
    
    std::cout << "\nThroughput:\n";
    std::cout << "Basic lookups/sec: " << (total_lookups / (basic_time / 1000.0f)) << std::endl;
    std::cout << "Cache hit rate: " << (float)total_cache_hits / total_lookups * 100.0f << "%" << std::endl;
    
    // Verify embedding statistics
    float embedding_mean = 0.0f, embedding_std = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        embedding_mean += h_output[i];
    }
    embedding_mean /= embedding_dim;
    
    for (int i = 0; i < embedding_dim; i++) {
        float diff = h_output[i] - embedding_mean;
        embedding_std += diff * diff;
    }
    embedding_std = sqrtf(embedding_std / embedding_dim);
    
    std::cout << "\nEmbedding Statistics (first token):\n";
    std::cout << "Mean: " << embedding_mean << std::endl;
    std::cout << "Std: " << embedding_std << std::endl;
    std::cout << "First few values: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup
    cudaFree(d_embedding_table); cudaFree(d_pos_table); cudaFree(d_pos_encodings);
    cudaFree(d_output); cudaFree(d_indices); cudaFree(d_lang_ids);
    cudaFree(d_lang_scales); cudaFree(d_dropout_mask); cudaFree(d_grad_embedding_table);
    cudaFree(d_cache_hits); cudaFree(d_grad_output);
    
    return 0;
}