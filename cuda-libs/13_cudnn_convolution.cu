// cuDNN Convolution: Optimized 2D convolution for CNNs
// Core operation for computer vision and image processing models

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
    // Convolution parameters
    const int batch_size = 64;
    const int in_channels = 3;
    const int in_height = 224;
    const int in_width = 224;

    const int out_channels = 64;
    const int kernel_h = 3;
    const int kernel_w = 3;
    const int pad_h = 1;
    const int pad_w = 1;
    const int stride_h = 1;
    const int stride_w = 1;

    const int out_height = (in_height + 2 * pad_h - kernel_h) / stride_h + 1;
    const int out_width = (in_width + 2 * pad_w - kernel_w) / stride_w + 1;

    printf("=== cuDNN 2D Convolution ===\n");
    printf("Input:  [%d, %d, %d, %d] (N, C, H, W)\n", batch_size, in_channels, in_height, in_width);
    printf("Kernel: [%d, %d, %d, %d] (K, C, H, W)\n", out_channels, in_channels, kernel_h, kernel_w);
    printf("Output: [%d, %d, %d, %d] (N, K, H, W)\n\n", batch_size, out_channels, out_height, out_width);

    // Allocate host memory
    size_t input_bytes = batch_size * in_channels * in_height * in_width * sizeof(float);
    size_t filter_bytes = out_channels * in_channels * kernel_h * kernel_w * sizeof(float);
    size_t output_bytes = batch_size * out_channels * out_height * out_width * sizeof(float);

    float *h_input = (float*)malloc(input_bytes);
    float *h_filter = (float*)malloc(filter_bytes);
    float *h_output = (float*)malloc(output_bytes);

    // Initialize with random values
    for (size_t i = 0; i < batch_size * in_channels * in_height * in_width; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }
    for (size_t i = 0; i < out_channels * in_channels * kernel_h * kernel_w; i++) {
        h_filter[i] = (float)(rand() % 100) / 100.0f - 0.5f;
    }

    // Allocate device memory
    float *d_input, *d_filter, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_filter, filter_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter, filter_bytes, cudaMemcpyHostToDevice));

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));

    // Set tensor descriptors (NCHW format)
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, in_channels, in_height, in_width));

    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                           batch_size, out_channels, out_height, out_width));

    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                           out_channels, in_channels, kernel_h, kernel_w));

    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc,
                                                pad_h, pad_w,
                                                stride_h, stride_w,
                                                1, 1,  // dilation
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    // Enable Tensor Core operations (if available)
    CHECK_CUDNN(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));

    // Find best convolution algorithm
    cudnnConvolutionFwdAlgoPerf_t perf_results[10];
    int returned_algo_count;

    CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
                                                     input_desc, filter_desc, conv_desc, output_desc,
                                                     10, &returned_algo_count, perf_results));

    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;
    printf("Selected algorithm: %d\n", algo);
    printf("Estimated time: %.3f ms\n\n", perf_results[0].time);

    // Allocate workspace
    size_t workspace_bytes = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        input_desc, filter_desc, conv_desc, output_desc,
                                                        algo, &workspace_bytes));

    void* d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_bytes));
    printf("Workspace size: %.2f MB\n\n", workspace_bytes / 1024.0 / 1024.0);

    // Warm-up
    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                        input_desc, d_input,
                                        filter_desc, d_filter,
                                        conv_desc, algo,
                                        d_workspace, workspace_bytes,
                                        &beta, output_desc, d_output));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));

    // Forward convolution
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                        input_desc, d_input,
                                        filter_desc, d_filter,
                                        conv_desc, algo,
                                        d_workspace, workspace_bytes,
                                        &beta, output_desc, d_output));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // Calculate performance
    long long total_ops = 2LL * batch_size * out_channels * out_height * out_width *
                         in_channels * kernel_h * kernel_w;
    double gflops = (total_ops * 1e-9) / (milliseconds / 1000.0);

    printf("=== Performance ===\n");
    printf("Time: %.3f ms\n", milliseconds);
    printf("Throughput: %.2f GFLOPS\n", gflops);
    printf("Images/sec: %.2f\n\n", batch_size / (milliseconds / 1000.0));

    // Sample output
    printf("Output sample (first channel, 4x4 corner):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%.3f ", h_output[i * out_width + j]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(input_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(output_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_input);
    free(h_filter);
    free(h_output);

    printf("\ncuDNN convolution completed successfully!\n");
    return 0;
}
