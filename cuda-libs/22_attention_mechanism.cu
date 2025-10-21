// Attention Mechanism: Core operation for Transformers and BERT
// Implements Scaled Dot-Product Attention: Attention(Q,K,V) = softmax(QK^T/√d)V

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Softmax kernel with numerically stable computation
__global__ void softmax(float* input, float* output, int seq_len, int d_model) {
    int row = blockIdx.x;  // Each block handles one sequence position

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_val = &shared[blockDim.x];

    // Find max for numerical stability
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = input[row * seq_len + i];
        local_max = fmaxf(local_max, val);
    }

    max_val[threadIdx.x] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    float global_max = max_val[0];
    __syncthreads();

    // Compute exp and sum
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        float val = expf(input[row * seq_len + i] - global_max);
        output[row * seq_len + i] = val;
        local_sum += val;
    }

    sum_val[threadIdx.x] = local_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum_val[threadIdx.x] += sum_val[threadIdx.x + stride];
        }
        __syncthreads();
    }

    float global_sum = sum_val[0];

    // Normalize
    for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
        output[row * seq_len + i] /= global_sum;
    }
}

// Scale attention scores by 1/sqrt(d_k)
__global__ void scaleScores(float* scores, int n, float scale) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        scores[tid] *= scale;
    }
}

int main() {
    // Attention parameters
    const int batch_size = 32;
    const int seq_len = 64;      // Sequence length
    const int d_model = 512;     // Model dimension
    const int d_k = d_model;     // Key dimension (same as model for simplicity)
    const int d_v = d_model;     // Value dimension

    printf("=== Transformer Attention Mechanism ===\n");
    printf("Batch size:     %d\n", batch_size);
    printf("Sequence:       %d tokens\n", seq_len);
    printf("Model dim:      %d\n", d_model);
    printf("Key/Value dim:  %d\n\n", d_k);

    const float scale = 1.0f / sqrtf((float)d_k);

    // Allocate memory
    size_t qkv_bytes = batch_size * seq_len * d_model * sizeof(float);
    size_t scores_bytes = batch_size * seq_len * seq_len * sizeof(float);

    float *h_Q = (float*)malloc(qkv_bytes);
    float *h_K = (float*)malloc(qkv_bytes);
    float *h_V = (float*)malloc(qkv_bytes);
    float *h_output = (float*)malloc(qkv_bytes);
    float *h_attention = (float*)malloc(scores_bytes);

    // Initialize Q, K, V with random values
    for (int i = 0; i < batch_size * seq_len * d_model; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_K[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_V[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }

    // Device memory
    float *d_Q, *d_K, *d_V, *d_scores, *d_attention, *d_output;
    CHECK_CUDA(cudaMalloc(&d_Q, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_K, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_V, qkv_bytes));
    CHECK_CUDA(cudaMalloc(&d_scores, scores_bytes));
    CHECK_CUDA(cudaMalloc(&d_attention, scores_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, qkv_bytes));

    CHECK_CUDA(cudaMemcpy(d_Q, h_Q, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_K, h_K, qkv_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_V, h_V, qkv_bytes, cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Step 1: Compute attention scores: Scores = Q @ K^T
    // For simplicity, processing one batch element at a time
    for (int b = 0; b < batch_size; b++) {
        float alpha = 1.0f, beta = 0.0f;

        float *Q_batch = d_Q + b * seq_len * d_model;
        float *K_batch = d_K + b * seq_len * d_model;
        float *scores_batch = d_scores + b * seq_len * seq_len;

        // Scores = Q @ K^T  (seq_len x seq_len) = (seq_len x d_k) @ (d_k x seq_len)
        CHECK_CUBLAS(cublasSgemm(handle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                seq_len, seq_len, d_k,
                                &alpha,
                                K_batch, d_k,      // K^T: d_k x seq_len
                                Q_batch, d_k,      // Q: d_k x seq_len
                                &beta,
                                scores_batch, seq_len));
    }

    // Step 2: Scale scores by 1/sqrt(d_k)
    int total_scores = batch_size * seq_len * seq_len;
    int blockSize = 256;
    int gridSize = (total_scores + blockSize - 1) / blockSize;
    scaleScores<<<gridSize, blockSize>>>(d_scores, total_scores, scale);

    // Step 3: Apply softmax to get attention weights
    dim3 softmaxBlocks(batch_size * seq_len);
    dim3 softmaxThreads(256);
    size_t sharedMem = 2 * softmaxThreads.x * sizeof(float);
    softmax<<<softmaxBlocks, softmaxThreads, sharedMem>>>(d_scores, d_attention, seq_len, d_model);

    // Step 4: Apply attention to values: Output = Attention @ V
    for (int b = 0; b < batch_size; b++) {
        float alpha = 1.0f, beta = 0.0f;

        float *attention_batch = d_attention + b * seq_len * seq_len;
        float *V_batch = d_V + b * seq_len * d_model;
        float *output_batch = d_output + b * seq_len * d_model;

        // Output = Attention @ V  (seq_len x d_v) = (seq_len x seq_len) @ (seq_len x d_v)
        CHECK_CUBLAS(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                d_v, seq_len, seq_len,
                                &alpha,
                                V_batch, d_v,
                                attention_batch, seq_len,
                                &beta,
                                output_batch, d_v));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results
    CHECK_CUDA(cudaMemcpy(h_attention, d_attention, scores_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_output, d_output, qkv_bytes, cudaMemcpyDeviceToHost));

    // Calculate FLOPS
    long long matmul1_ops = (long long)batch_size * 2LL * seq_len * seq_len * d_k;
    long long matmul2_ops = (long long)batch_size * 2LL * seq_len * seq_len * d_v;
    long long softmax_ops = (long long)batch_size * seq_len * seq_len * 5;  // exp + div
    long long total_ops = matmul1_ops + matmul2_ops + softmax_ops;

    double gflops = (total_ops * 1e-9) / (milliseconds / 1000.0);

    printf("=== Performance ===\n");
    printf("Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("Sequences/sec: %.2f\n\n", batch_size / (milliseconds / 1000.0));

    // Display sample attention weights (first sequence, first query)
    printf("Attention weights (first sequence, first query position):\n");
    printf("(Shows which tokens it attends to)\n");
    for (int i = 0; i < 10; i++) {
        printf("Token %d: %.4f\n", i, h_attention[i]);
    }

    // Verify softmax (should sum to 1)
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += h_attention[i];
    }
    printf("\nAttention weights sum: %.6f (should be 1.0)\n", sum);

    printf("\n=== Output Sample ===\n");
    printf("Output vector (first sequence, first position, first 10 dims):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUBLAS(cublasDestroy(handle));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_K));
    CHECK_CUDA(cudaFree(d_V));
    CHECK_CUDA(cudaFree(d_scores));
    CHECK_CUDA(cudaFree(d_attention));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_Q);
    free(h_K);
    free(h_V);
    free(h_output);
    free(h_attention);

    printf("\nAttention mechanism completed successfully!\n");
    return 0;
}
