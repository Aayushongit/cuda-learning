#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

// Standard cross-entropy loss forward pass
__global__ void cross_entropy_loss_forward(const float* predictions, const int* targets,
                                          float* losses, int batch_size, int num_classes) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int target_class = targets[batch_idx];
        
        if (target_class >= 0 && target_class < num_classes) {
            float pred_prob = predictions[batch_idx * num_classes + target_class];
            // Clamp to prevent log(0)
            pred_prob = fmaxf(pred_prob, 1e-7f);
            losses[batch_idx] = -logf(pred_prob);
        } else {
            losses[batch_idx] = 0.0f; // Ignore invalid targets
        }
    }
}

// Cross-entropy loss with label smoothing
__global__ void cross_entropy_loss_label_smoothing(const float* predictions, const int* targets,
                                                   float* losses, int batch_size, int num_classes,
                                                   float smoothing = 0.1f) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int target_class = targets[batch_idx];
        
        if (target_class >= 0 && target_class < num_classes) {
            float loss = 0.0f;
            float smooth_prob = smoothing / num_classes;
            float target_prob = 1.0f - smoothing + smooth_prob;
            
            for (int c = 0; c < num_classes; c++) {
                float pred_prob = fmaxf(predictions[batch_idx * num_classes + c], 1e-7f);
                float true_prob = (c == target_class) ? target_prob : smooth_prob;
                loss -= true_prob * logf(pred_prob);
            }
            
            losses[batch_idx] = loss;
        } else {
            losses[batch_idx] = 0.0f;
        }
    }
}

// Cross-entropy loss backward pass
__global__ void cross_entropy_loss_backward(const float* predictions, const int* targets,
                                           float* grad_predictions, int batch_size, int num_classes) {
    
    int batch_idx = blockIdx.y;
    int class_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        int target_class = targets[batch_idx];
        int idx = batch_idx * num_classes + class_idx;
        
        if (target_class >= 0 && target_class < num_classes) {
            float pred_prob = predictions[idx];
            if (class_idx == target_class) {
                grad_predictions[idx] = pred_prob - 1.0f; // derivative of -log(p) w.r.t. p
            } else {
                grad_predictions[idx] = pred_prob;
            }
        } else {
            grad_predictions[idx] = 0.0f; // Ignore invalid targets
        }
    }
}

// Focal loss for imbalanced datasets
__global__ void focal_loss_forward(const float* predictions, const int* targets,
                                  float* losses, int batch_size, int num_classes,
                                  float alpha = 1.0f, float gamma = 2.0f) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int target_class = targets[batch_idx];
        
        if (target_class >= 0 && target_class < num_classes) {
            float pred_prob = predictions[batch_idx * num_classes + target_class];
            pred_prob = fmaxf(pred_prob, 1e-7f);
            
            // Focal loss: -α * (1-p)^γ * log(p)
            float focal_weight = alpha * powf(1.0f - pred_prob, gamma);
            losses[batch_idx] = focal_weight * (-logf(pred_prob));
        } else {
            losses[batch_idx] = 0.0f;
        }
    }
}

// Focal loss backward pass
__global__ void focal_loss_backward(const float* predictions, const int* targets,
                                   float* grad_predictions, int batch_size, int num_classes,
                                   float alpha = 1.0f, float gamma = 2.0f) {
    
    int batch_idx = blockIdx.y;
    int class_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        int target_class = targets[batch_idx];
        int idx = batch_idx * num_classes + class_idx;
        
        if (target_class >= 0 && target_class < num_classes) {
            float pred_prob = fmaxf(predictions[idx], 1e-7f);
            
            if (class_idx == target_class) {
                // Derivative for target class
                float one_minus_p = 1.0f - pred_prob;
                float focal_weight = alpha * powf(one_minus_p, gamma);
                float grad = focal_weight * (gamma * pred_prob * logf(pred_prob) + one_minus_p - 1.0f);
                grad_predictions[idx] = grad;
            } else {
                // Derivative for non-target classes  
                grad_predictions[idx] = alpha * powf(1.0f - pred_prob, gamma - 1.0f) * 
                                       gamma * pred_prob * pred_prob;
            }
        } else {
            grad_predictions[idx] = 0.0f;
        }
    }
}

// Dice loss for segmentation tasks
__global__ void dice_loss_forward(const float* predictions, const int* targets,
                                 float* losses, int batch_size, int num_classes,
                                 float smooth = 1.0f) {
    
    int class_idx = blockIdx.x;
    int batch_idx = threadIdx.x;
    
    if (class_idx >= num_classes || batch_idx >= batch_size) return;
    
    extern __shared__ float sdata[];
    float* s_intersection = sdata;
    float* s_pred_sum = sdata + blockDim.x;
    float* s_target_sum = sdata + 2 * blockDim.x;
    
    // Calculate intersection and sums for this class
    float intersection = 0.0f, pred_sum = 0.0f, target_sum = 0.0f;
    
    for (int i = batch_idx; i < batch_size; i += blockDim.x) {
        float pred = predictions[i * num_classes + class_idx];
        float target = (targets[i] == class_idx) ? 1.0f : 0.0f;
        
        intersection += pred * target;
        pred_sum += pred;
        target_sum += target;
    }
    
    s_intersection[batch_idx] = intersection;
    s_pred_sum[batch_idx] = pred_sum;
    s_target_sum[batch_idx] = target_sum;
    
    __syncthreads();
    
    // Reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (batch_idx < stride) {
            s_intersection[batch_idx] += s_intersection[batch_idx + stride];
            s_pred_sum[batch_idx] += s_pred_sum[batch_idx + stride];
            s_target_sum[batch_idx] += s_target_sum[batch_idx + stride];
        }
        __syncthreads();
    }
    
    if (batch_idx == 0) {
        float dice_coeff = (2.0f * s_intersection[0] + smooth) / 
                          (s_pred_sum[0] + s_target_sum[0] + smooth);
        losses[class_idx] = 1.0f - dice_coeff;
    }
}

// Binary cross-entropy loss
__global__ void binary_cross_entropy_loss(const float* predictions, const float* targets,
                                         float* losses, int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        float pred = fmaxf(fminf(predictions[idx], 1.0f - 1e-7f), 1e-7f); // Clamp to [eps, 1-eps]
        float target = targets[idx];
        
        losses[idx] = -target * logf(pred) - (1.0f - target) * logf(1.0f - pred);
    }
}

// KL divergence loss
__global__ void kl_divergence_loss(const float* predictions, const float* targets,
                                  float* losses, int batch_size, int num_classes) {
    
    int batch_idx = blockIdx.y;
    int class_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ float s_kl_loss[];
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        float pred = fmaxf(predictions[batch_idx * num_classes + class_idx], 1e-7f);
        float target = fmaxf(targets[batch_idx * num_classes + class_idx], 1e-7f);
        
        // KL(target || pred) = target * log(target / pred)
        s_kl_loss[threadIdx.x] = target * logf(target / pred);
    } else {
        s_kl_loss[threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Reduction to sum over classes for this batch
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_kl_loss[threadIdx.x] += s_kl_loss[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        losses[batch_idx] = s_kl_loss[0];
    }
}

// Contrastive loss for metric learning
__global__ void contrastive_loss(const float* embeddings1, const float* embeddings2,
                                const float* labels, float* losses,
                                int batch_size, int embedding_dim,
                                float margin = 1.0f) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Compute Euclidean distance
        float distance_sq = 0.0f;
        for (int d = 0; d < embedding_dim; d++) {
            float diff = embeddings1[batch_idx * embedding_dim + d] - 
                        embeddings2[batch_idx * embedding_dim + d];
            distance_sq += diff * diff;
        }
        float distance = sqrtf(distance_sq);
        
        float label = labels[batch_idx];
        if (label > 0.5f) {
            // Positive pair - minimize distance
            losses[batch_idx] = distance_sq;
        } else {
            // Negative pair - maximize distance up to margin
            losses[batch_idx] = fmaxf(0.0f, margin - distance) * fmaxf(0.0f, margin - distance);
        }
    }
}

// Triplet loss for metric learning
__global__ void triplet_loss(const float* anchor, const float* positive, const float* negative,
                            float* losses, int batch_size, int embedding_dim,
                            float margin = 1.0f) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Compute distances
        float pos_dist_sq = 0.0f, neg_dist_sq = 0.0f;
        
        for (int d = 0; d < embedding_dim; d++) {
            int idx = batch_idx * embedding_dim + d;
            
            float anchor_val = anchor[idx];
            float pos_diff = anchor_val - positive[idx];
            float neg_diff = anchor_val - negative[idx];
            
            pos_dist_sq += pos_diff * pos_diff;
            neg_dist_sq += neg_diff * neg_diff;
        }
        
        // Triplet loss: max(0, d(a,p) - d(a,n) + margin)
        losses[batch_idx] = fmaxf(0.0f, pos_dist_sq - neg_dist_sq + margin);
    }
}

// Compute loss reduction (mean, sum, none)
__global__ void reduce_losses(const float* losses, float* reduced_loss,
                             int batch_size, int reduction_type) {
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < batch_size) {
        sdata[tid] = losses[idx];
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
    
    if (tid == 0) {
        float result = sdata[0];
        
        if (reduction_type == 0) {
            // Mean reduction
            result = result / batch_size;
        }
        // Sum reduction is just the accumulated value
        // No reduction would keep individual losses
        
        atomicAdd(reduced_loss, result);
    }
}

// Loss with class weights for imbalanced datasets
__global__ void weighted_cross_entropy_loss(const float* predictions, const int* targets,
                                           const float* class_weights, float* losses,
                                           int batch_size, int num_classes) {
    
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        int target_class = targets[batch_idx];
        
        if (target_class >= 0 && target_class < num_classes) {
            float pred_prob = fmaxf(predictions[batch_idx * num_classes + target_class], 1e-7f);
            float weight = class_weights[target_class];
            losses[batch_idx] = -weight * logf(pred_prob);
        } else {
            losses[batch_idx] = 0.0f;
        }
    }
}

void initialize_random_float(float* data, int size, float min_val = 0.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
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

void normalize_predictions(float* predictions, int batch_size, int num_classes) {
    // Simple softmax normalization on CPU for initialization
    for (int b = 0; b < batch_size; b++) {
        float* batch_pred = &predictions[b * num_classes];
        
        // Find max for numerical stability
        float max_val = *std::max_element(batch_pred, batch_pred + num_classes);
        
        // Compute softmax
        float sum = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            batch_pred[c] = expf(batch_pred[c] - max_val);
            sum += batch_pred[c];
        }
        
        for (int c = 0; c < num_classes; c++) {
            batch_pred[c] /= sum;
        }
    }
}

int main() {
    // Loss computation parameters
    const int batch_size = 1024;
    const int num_classes = 1000;
    const int embedding_dim = 128;
    const float smoothing = 0.1f;
    const float focal_alpha = 1.0f;
    const float focal_gamma = 2.0f;
    const float margin = 1.0f;
    
    std::cout << "Loss Function Benchmark\n";
    std::cout << "Batch size: " << batch_size << ", Num classes: " << num_classes << std::endl;
    
    // Memory sizes
    int predictions_size = batch_size * num_classes;
    int targets_size = batch_size;
    int embeddings_size = batch_size * embedding_dim;
    
    // Host memory
    std::vector<float> h_predictions(predictions_size);
    std::vector<float> h_soft_targets(predictions_size); // For KL divergence
    std::vector<int> h_targets(targets_size);
    std::vector<float> h_losses(batch_size);
    std::vector<float> h_grad_predictions(predictions_size);
    std::vector<float> h_class_weights(num_classes);
    std::vector<float> h_embeddings1(embeddings_size), h_embeddings2(embeddings_size);
    std::vector<float> h_binary_targets(batch_size);
    
    // Initialize data
    initialize_random_float(h_predictions.data(), predictions_size, -5.0f, 5.0f);
    normalize_predictions(h_predictions.data(), batch_size, num_classes);
    
    initialize_random_float(h_soft_targets.data(), predictions_size);
    normalize_predictions(h_soft_targets.data(), batch_size, num_classes);
    
    initialize_random_int(h_targets.data(), targets_size, 0, num_classes);
    initialize_random_float(h_class_weights.data(), num_classes, 0.5f, 2.0f);
    initialize_random_float(h_embeddings1.data(), embeddings_size, -1.0f, 1.0f);
    initialize_random_float(h_embeddings2.data(), embeddings_size, -1.0f, 1.0f);
    initialize_random_float(h_binary_targets.data(), batch_size, 0.0f, 1.0f);
    
    // Device memory
    float *d_predictions, *d_soft_targets, *d_losses, *d_grad_predictions;
    float *d_class_weights, *d_reduced_loss;
    float *d_embeddings1, *d_embeddings2, *d_binary_targets;
    int *d_targets;
    
    cudaMalloc(&d_predictions, predictions_size * sizeof(float));
    cudaMalloc(&d_soft_targets, predictions_size * sizeof(float));
    cudaMalloc(&d_targets, targets_size * sizeof(int));
    cudaMalloc(&d_losses, batch_size * sizeof(float));
    cudaMalloc(&d_grad_predictions, predictions_size * sizeof(float));
    cudaMalloc(&d_class_weights, num_classes * sizeof(float));
    cudaMalloc(&d_reduced_loss, sizeof(float));
    cudaMalloc(&d_embeddings1, embeddings_size * sizeof(float));
    cudaMalloc(&d_embeddings2, embeddings_size * sizeof(float));
    cudaMalloc(&d_binary_targets, batch_size * sizeof(float));
    
    cudaMemcpy(d_predictions, h_predictions.data(), predictions_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_soft_targets, h_soft_targets.data(), predictions_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_targets, h_targets.data(), targets_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_class_weights, h_class_weights.data(), num_classes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embeddings1, h_embeddings1.data(), embeddings_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_embeddings2, h_embeddings2.data(), embeddings_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_binary_targets, h_binary_targets.data(), batch_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Kernel configurations
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    
    dim3 backward_block(32);
    dim3 backward_grid((num_classes + backward_block.x - 1) / backward_block.x, batch_size);
    
    // Benchmark standard cross-entropy loss
    auto start = std::chrono::high_resolution_clock::now();
    cross_entropy_loss_forward<<<grid, block>>>(d_predictions, d_targets, d_losses, batch_size, num_classes);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto ce_forward_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Cross-entropy backward
    start = std::chrono::high_resolution_clock::now();
    cross_entropy_loss_backward<<<backward_grid, backward_block>>>(
        d_predictions, d_targets, d_grad_predictions, batch_size, num_classes);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto ce_backward_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Label smoothing
    start = std::chrono::high_resolution_clock::now();
    cross_entropy_loss_label_smoothing<<<grid, block>>>(
        d_predictions, d_targets, d_losses, batch_size, num_classes, smoothing);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto label_smooth_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Focal loss
    start = std::chrono::high_resolution_clock::now();
    focal_loss_forward<<<grid, block>>>(
        d_predictions, d_targets, d_losses, batch_size, num_classes, focal_alpha, focal_gamma);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto focal_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Binary cross-entropy
    dim3 binary_grid((predictions_size + block.x - 1) / block.x);
    
    start = std::chrono::high_resolution_clock::now();
    binary_cross_entropy_loss<<<binary_grid, block>>>(
        d_predictions, d_predictions, d_losses, predictions_size); // Use predictions as both pred and target for demo
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto bce_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // KL divergence
    dim3 kl_grid((num_classes + backward_block.x - 1) / backward_block.x, batch_size);
    
    start = std::chrono::high_resolution_clock::now();
    kl_divergence_loss<<<kl_grid, backward_block, backward_block.x * sizeof(float)>>>(
        d_predictions, d_soft_targets, d_losses, batch_size, num_classes);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto kl_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Contrastive loss
    start = std::chrono::high_resolution_clock::now();
    contrastive_loss<<<grid, block>>>(
        d_embeddings1, d_embeddings2, d_binary_targets, d_losses, 
        batch_size, embedding_dim, margin);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto contrastive_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Triplet loss
    start = std::chrono::high_resolution_clock::now();
    triplet_loss<<<grid, block>>>(
        d_embeddings1, d_embeddings2, d_embeddings1, d_losses, // Use embeddings1 as negative for demo
        batch_size, embedding_dim, margin);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto triplet_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Weighted cross-entropy
    start = std::chrono::high_resolution_clock::now();
    weighted_cross_entropy_loss<<<grid, block>>>(
        d_predictions, d_targets, d_class_weights, d_losses, batch_size, num_classes);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto weighted_ce_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Loss reduction test
    cudaMemset(d_reduced_loss, 0, sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    reduce_losses<<<grid, block, block.x * sizeof(float)>>>(
        d_losses, d_reduced_loss, batch_size, 0); // Mean reduction
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    auto reduction_time = std::chrono::duration<float, std::milli>(end - start).count();
    
    // Copy results back for verification
    cudaMemcpy(h_losses.data(), d_losses, batch_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_grad_predictions.data(), d_grad_predictions, predictions_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    float h_reduced_loss;
    cudaMemcpy(&h_reduced_loss, d_reduced_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "\nPerformance Results:\n";
    std::cout << "Cross-Entropy Forward: " << ce_forward_time << " ms\n";
    std::cout << "Cross-Entropy Backward: " << ce_backward_time << " ms\n";
    std::cout << "Label Smoothing CE: " << label_smooth_time << " ms\n";
    std::cout << "Focal Loss: " << focal_time << " ms\n";
    std::cout << "Binary Cross-Entropy: " << bce_time << " ms\n";
    std::cout << "KL Divergence: " << kl_time << " ms\n";
    std::cout << "Contrastive Loss: " << contrastive_time << " ms\n";
    std::cout << "Triplet Loss: " << triplet_time << " ms\n";
    std::cout << "Weighted Cross-Entropy: " << weighted_ce_time << " ms\n";
    std::cout << "Loss Reduction: " << reduction_time << " ms\n";
    
    // Calculate loss statistics
    float loss_mean = 0.0f, loss_max = 0.0f, loss_min = FLT_MAX;
    for (int i = 0; i < batch_size; i++) {
        loss_mean += h_losses[i];
        loss_max = fmaxf(loss_max, h_losses[i]);
        loss_min = fminf(loss_min, h_losses[i]);
    }
    loss_mean /= batch_size;
    
    // Calculate gradient statistics
    float grad_mean = 0.0f, grad_abs_mean = 0.0f;
    for (int i = 0; i < predictions_size; i++) {
        grad_mean += h_grad_predictions[i];
        grad_abs_mean += fabsf(h_grad_predictions[i]);
    }
    grad_mean /= predictions_size;
    grad_abs_mean /= predictions_size;
    
    std::cout << "\nLoss Statistics:\n";
    std::cout << "Loss Mean: " << loss_mean << std::endl;
    std::cout << "Loss Min: " << loss_min << std::endl;
    std::cout << "Loss Max: " << loss_max << std::endl;
    std::cout << "Reduced Loss (mean): " << h_reduced_loss << std::endl;
    
    std::cout << "\nGradient Statistics:\n";
    std::cout << "Gradient Mean: " << grad_mean << std::endl;
    std::cout << "Gradient Absolute Mean: " << grad_abs_mean << std::endl;
    
    std::cout << "\nFirst few losses: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_losses[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "\nFirst few gradients: ";
    for (int i = 0; i < 5; i++) {
        std::cout << h_grad_predictions[i] << " ";
    }
    std::cout << std::endl;
    
    // Throughput calculation
    long long total_elements_processed = (long long)batch_size * num_classes;
    float ce_throughput = total_elements_processed / (ce_forward_time / 1000.0f);
    
    std::cout << "\nThroughput:\n";
    std::cout << "Cross-Entropy Elements/sec: " << ce_throughput << std::endl;
    
    // Cleanup
    cudaFree(d_predictions); cudaFree(d_soft_targets); cudaFree(d_targets);
    cudaFree(d_losses); cudaFree(d_grad_predictions); cudaFree(d_class_weights);
    cudaFree(d_reduced_loss); cudaFree(d_embeddings1); cudaFree(d_embeddings2);
    cudaFree(d_binary_targets);
    
    return 0;
}