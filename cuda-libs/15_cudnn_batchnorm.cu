// cuDNN Batch Normalization: Normalize activations across mini-batch
// Critical for training stability and convergence speed in deep networks

#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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
    const int batch_size = 64;
    const int channels = 256;
    const int height = 28;
    const int width = 28;

    printf("=== cuDNN Batch Normalization ===\n");
    printf("Input shape: [%d, %d, %d, %d] (N, C, H, W)\n\n", batch_size, channels, height, width);

    size_t tensor_bytes = batch_size * channels * height * width * sizeof(float);
    size_t param_bytes = channels * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(tensor_bytes);
    float *h_output = (float*)malloc(tensor_bytes);
    float *h_scale = (float*)malloc(param_bytes);
    float *h_bias = (float*)malloc(param_bytes);
    float *h_running_mean = (float*)malloc(param_bytes);
    float *h_running_var = (float*)malloc(param_bytes);
    float *h_saved_mean = (float*)malloc(param_bytes);
    float *h_saved_inv_var = (float*)malloc(param_bytes);

    // Initialize input with random data
    for (size_t i = 0; i < batch_size * channels * height * width; i++) {
        h_input[i] = (float)(rand() % 1000) / 100.0f;
    }

    // Initialize scale (gamma) and bias (beta)
    for (int i = 0; i < channels; i++) {
        h_scale[i] = 1.0f;
        h_bias[i] = 0.0f;
        h_running_mean[i] = 0.0f;
        h_running_var[i] = 1.0f;
    }

    // Device memory
    float *d_input, *d_output;
    float *d_scale, *d_bias;
    float *d_running_mean, *d_running_var;
    float *d_saved_mean, *d_saved_inv_var;

    CHECK_CUDA(cudaMalloc(&d_input, tensor_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, tensor_bytes));
    CHECK_CUDA(cudaMalloc(&d_scale, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_bias, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_running_mean, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_running_var, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_saved_mean, param_bytes));
    CHECK_CUDA(cudaMalloc(&d_saved_inv_var, param_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, tensor_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_scale, h_scale, param_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, h_bias, param_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_running_mean, h_running_mean, param_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_running_var, h_running_var, param_bytes, cudaMemcpyHostToDevice));

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, param_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&param_desc));

    // Set tensor descriptors
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, channels, height, width));

    // Derive parameters descriptor from input
    cudnnBatchNormMode_t mode = CUDNN_BATCHNORM_SPATIAL;
    CHECK_CUDNN(cudnnDeriveBNTensorDescriptor(param_desc, input_desc, mode));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    const double epsilon = 1e-5;
    const double exponential_avg_factor = 0.1;  // Momentum for running stats
    const float alpha = 1.0f, beta = 0.0f;

    // TRAINING MODE: Compute batch statistics
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDNN(cudnnBatchNormalizationForwardTraining(
        cudnn, mode, &alpha, &beta,
        input_desc, d_input,
        input_desc, d_output,
        param_desc,
        d_scale, d_bias,
        exponential_avg_factor,
        d_running_mean, d_running_var,
        epsilon,
        d_saved_mean, d_saved_inv_var));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_training = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_training, start, stop));

    // INFERENCE MODE: Use running statistics
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDNN(cudnnBatchNormalizationForwardInference(
        cudnn, mode, &alpha, &beta,
        input_desc, d_input,
        input_desc, d_output,
        param_desc,
        d_scale, d_bias,
        d_running_mean, d_running_var,
        epsilon));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_inference = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_inference, start, stop));

    // Copy results
    CHECK_CUDA(cudaMemcpy(h_output, d_output, tensor_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_saved_mean, d_saved_mean, param_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_saved_inv_var, d_saved_inv_var, param_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_running_mean, d_running_mean, param_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_running_var, d_running_var, param_bytes, cudaMemcpyDeviceToHost));

    // Calculate statistics for verification
    float input_mean = 0.0f, output_mean = 0.0f;
    float input_var = 0.0f, output_var = 0.0f;

    size_t total = batch_size * channels * height * width;
    for (size_t i = 0; i < total; i++) {
        input_mean += h_input[i];
        output_mean += h_output[i];
    }
    input_mean /= total;
    output_mean /= total;

    for (size_t i = 0; i < total; i++) {
        input_var += (h_input[i] - input_mean) * (h_input[i] - input_mean);
        output_var += (h_output[i] - output_mean) * (h_output[i] - output_mean);
    }
    input_var /= total;
    output_var /= total;

    // Performance metrics
    size_t elements = batch_size * channels * height * width;

    printf("=== Statistics ===\n");
    printf("Input  - Mean: %.4f, Variance: %.4f\n", input_mean, input_var);
    printf("Output - Mean: %.4f, Variance: %.4f\n", output_mean, output_var);
    printf("\nFirst channel saved mean: %.4f\n", h_saved_mean[0]);
    printf("First channel saved variance: %.4f\n", 1.0f / (h_saved_inv_var[0] * h_saved_inv_var[0]) - epsilon);
    printf("First channel running mean: %.4f\n", h_running_mean[0]);
    printf("First channel running variance: %.4f\n", h_running_var[0]);

    printf("\n=== Performance ===\n");
    printf("Training mode:   %.3f ms (%.2f B elem/sec)\n",
           ms_training, elements / (ms_training * 1e6));
    printf("Inference mode:  %.3f ms (%.2f B elem/sec)\n",
           ms_inference, elements / (ms_inference * 1e6));

    // Sample input/output
    printf("\nInput sample (first 10 elements of first channel):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_input[i]);
    }

    printf("\n\nOutput sample (first 10 elements of first channel):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_output[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(param_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_scale));
    CHECK_CUDA(cudaFree(d_bias));
    CHECK_CUDA(cudaFree(d_running_mean));
    CHECK_CUDA(cudaFree(d_running_var));
    CHECK_CUDA(cudaFree(d_saved_mean));
    CHECK_CUDA(cudaFree(d_saved_inv_var));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_output);
    free(h_scale);
    free(h_bias);
    free(h_running_mean);
    free(h_running_var);
    free(h_saved_mean);
    free(h_saved_inv_var);

    printf("\ncuDNN batch normalization completed successfully!\n");
    return 0;
}
