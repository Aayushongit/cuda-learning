// cuDNN Pooling & Activation: Max/Average pooling and activation functions
// Essential operations for downsampling and non-linear transformations in CNNs

#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDNN(call) { \
    cudnnStatus_t status = call; \
    if (status != CUDNN_STATUS_SUCCESS) { \
        fprintf(stderr, "cuDNN Error: %s (line %d)\n", cudnnGetErrorString(status), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    const int batch_size = 32;
    const int channels = 128;
    const int in_height = 56;
    const int in_width = 56;

    const int pool_h = 2;
    const int pool_w = 2;
    const int stride_h = 2;
    const int stride_w = 2;

    const int out_height = (in_height - pool_h) / stride_h + 1;
    const int out_width = (in_width - pool_w) / stride_w + 1;

    printf("=== cuDNN Pooling & Activation ===\n");
    printf("Input:  [%d, %d, %d, %d]\n", batch_size, channels, in_height, in_width);
    printf("Pool:   %dx%d, stride %d\n", pool_h, pool_w, stride_h);
    printf("Output: [%d, %d, %d, %d]\n\n", batch_size, channels, out_height, out_width);

    size_t input_bytes = batch_size * channels * in_height * in_width * sizeof(float);
    size_t output_bytes = batch_size * channels * out_height * out_width * sizeof(float);

    // Allocate memory
    float *h_input = (float*)malloc(input_bytes);
    float *h_pooled = (float*)malloc(output_bytes);
    float *h_activated = (float*)malloc(input_bytes);

    // Initialize input
    for (size_t i = 0; i < batch_size * channels * in_height * in_width; i++) {
        h_input[i] = (float)(rand() % 200 - 100) / 10.0f;  // -10 to 10
    }

    float *d_input, *d_pooled, *d_activated;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_pooled, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_activated, input_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnPoolingDescriptor_t pooling_desc;
    cudnnActivationDescriptor_t activation_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&pooling_desc));
    CHECK_CUDNN(cudnnCreateActivationDescriptor(&activation_desc));

    // Set descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, channels, in_height, in_width));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, channels, out_height, out_width));

    // Max pooling
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc,
                                            CUDNN_POOLING_MAX,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            pool_h, pool_w,
                                            0, 0,  // padding
                                            stride_h, stride_w));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    float alpha = 1.0f, beta = 0.0f;

    // 1. MAX POOLING
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnPoolingForward(cudnn, pooling_desc, &alpha,
                                    input_desc, d_input, &beta,
                                    output_desc, d_pooled));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_maxpool = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_maxpool, start, stop));

    // 2. AVERAGE POOLING
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(pooling_desc,
                                            CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING,
                                            CUDNN_NOT_PROPAGATE_NAN,
                                            pool_h, pool_w,
                                            0, 0,
                                            stride_h, stride_w));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnPoolingForward(cudnn, pooling_desc, &alpha,
                                    input_desc, d_input, &beta,
                                    output_desc, d_pooled));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_avgpool = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_avgpool, start, stop));

    // 3. ReLU ACTIVATION
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                             CUDNN_ACTIVATION_RELU,
                                             CUDNN_NOT_PROPAGATE_NAN,
                                             0.0));  // ceiling for clipped ReLU

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc, &alpha,
                                       input_desc, d_input, &beta,
                                       input_desc, d_activated));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_relu = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_relu, start, stop));

    // 4. SIGMOID ACTIVATION
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                             CUDNN_ACTIVATION_SIGMOID,
                                             CUDNN_NOT_PROPAGATE_NAN,
                                             0.0));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc, &alpha,
                                       input_desc, d_input, &beta,
                                       input_desc, d_activated));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_sigmoid = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sigmoid, start, stop));

    // 5. TANH ACTIVATION
    CHECK_CUDNN(cudnnSetActivationDescriptor(activation_desc,
                                             CUDNN_ACTIVATION_TANH,
                                             CUDNN_NOT_PROPAGATE_NAN,
                                             0.0));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUDNN(cudnnActivationForward(cudnn, activation_desc, &alpha,
                                       input_desc, d_input, &beta,
                                       input_desc, d_activated));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_tanh = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_tanh, start, stop));

    // Copy results
    CHECK_CUDA(cudaMemcpy(h_pooled, d_pooled, output_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_activated, d_activated, input_bytes, cudaMemcpyDeviceToHost));

    // Performance metrics
    size_t pool_elements = batch_size * channels * out_height * out_width;
    size_t act_elements = batch_size * channels * in_height * in_width;

    printf("=== Performance ===\n");
    printf("Max Pooling:    %.3f ms (%.2f B elem/sec)\n",
           ms_maxpool, pool_elements / (ms_maxpool * 1e6));
    printf("Avg Pooling:    %.3f ms (%.2f B elem/sec)\n",
           ms_avgpool, pool_elements / (ms_avgpool * 1e6));
    printf("ReLU:           %.3f ms (%.2f B elem/sec)\n",
           ms_relu, act_elements / (ms_relu * 1e6));
    printf("Sigmoid:        %.3f ms (%.2f B elem/sec)\n",
           ms_sigmoid, act_elements / (ms_sigmoid * 1e6));
    printf("Tanh:           %.3f ms (%.2f B elem/sec)\n",
           ms_tanh, act_elements / (ms_tanh * 1e6));

    // Sample outputs
    printf("\nInput sample (4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", h_input[i * in_width + j]);
        }
        printf("\n");
    }

    printf("\nMax pooled output (2x2):\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%6.2f ", h_pooled[i * out_width + j]);
        }
        printf("\n");
    }

    printf("\nReLU output sample (first 10):\n");
    for (int i = 0; i < 10; i++) {
        printf("%6.2f ", h_activated[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyPoolingDescriptor(pooling_desc));
    CHECK_CUDNN(cudnnDestroyActivationDescriptor(activation_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_pooled));
    CHECK_CUDA(cudaFree(d_activated));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_pooled);
    free(h_activated);

    printf("\ncuDNN pooling & activation completed successfully!\n");
    return 0;
}
