// cuDNN RNN/LSTM: Recurrent neural networks for sequence processing
// Essential for NLP, time-series, and sequential data processing

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
    // RNN configuration
    const int seq_length = 32;      // Sequence length
    const int batch_size = 64;      // Batch size
    const int input_size = 128;     // Input feature dimension
    const int hidden_size = 256;    // Hidden state size
    const int num_layers = 2;       // Number of stacked RNN layers

    printf("=== cuDNN LSTM Network ===\n");
    printf("Sequence length: %d\n", seq_length);
    printf("Batch size:      %d\n", batch_size);
    printf("Input size:      %d\n", input_size);
    printf("Hidden size:     %d\n", hidden_size);
    printf("Num layers:      %d\n\n", num_layers);

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    // Create RNN descriptor
    cudnnRNNDescriptor_t rnn_desc;
    cudnnDropoutDescriptor_t dropout_desc;
    CHECK_CUDNN(cudnnCreateRNNDescriptor(&rnn_desc));
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&dropout_desc));

    // Setup dropout (disabled for simplicity)
    size_t state_size;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(cudnn, &state_size));
    void *dropout_states;
    CHECK_CUDA(cudaMalloc(&dropout_states, state_size));

    CHECK_CUDNN(cudnnSetDropoutDescriptor(dropout_desc, cudnn,
                                          0.0f,  // dropout probability (0 = disabled)
                                          dropout_states, state_size,
                                          1234ULL));  // seed

    // Set RNN descriptor for LSTM
    CHECK_CUDNN(cudnnSetRNNDescriptor_v6(cudnn, rnn_desc,
                                         hidden_size,
                                         num_layers,
                                         dropout_desc,
                                         CUDNN_LINEAR_INPUT,
                                         CUDNN_UNIDIRECTIONAL,
                                         CUDNN_LSTM,
                                         CUDNN_RNN_ALGO_STANDARD,
                                         CUDNN_DATA_FLOAT));

    // Create tensor descriptors for each time step
    cudnnTensorDescriptor_t *x_desc = (cudnnTensorDescriptor_t*)malloc(seq_length * sizeof(cudnnTensorDescriptor_t));
    cudnnTensorDescriptor_t *y_desc = (cudnnTensorDescriptor_t*)malloc(seq_length * sizeof(cudnnTensorDescriptor_t));

    int dims_x[3] = {batch_size, input_size, 1};
    int strides_x[3] = {dims_x[1] * dims_x[2], dims_x[2], 1};

    int dims_y[3] = {batch_size, hidden_size, 1};
    int strides_y[3] = {dims_y[1] * dims_y[2], dims_y[2], 1};

    for (int i = 0; i < seq_length; i++) {
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&x_desc[i]));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(x_desc[i], CUDNN_DATA_FLOAT, 3, dims_x, strides_x));

        CHECK_CUDNN(cudnnCreateTensorDescriptor(&y_desc[i]));
        CHECK_CUDNN(cudnnSetTensorNdDescriptor(y_desc[i], CUDNN_DATA_FLOAT, 3, dims_y, strides_y));
    }

    // Hidden and cell state descriptors
    cudnnTensorDescriptor_t h_desc, c_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&h_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&c_desc));

    int dims_hc[3] = {num_layers, batch_size, hidden_size};
    int strides_hc[3] = {dims_hc[1] * dims_hc[2], dims_hc[2], 1};

    CHECK_CUDNN(cudnnSetTensorNdDescriptor(h_desc, CUDNN_DATA_FLOAT, 3, dims_hc, strides_hc));
    CHECK_CUDNN(cudnnSetTensorNdDescriptor(c_desc, CUDNN_DATA_FLOAT, 3, dims_hc, strides_hc));

    // Get weight space size
    size_t weight_space_size;
    CHECK_CUDNN(cudnnGetRNNParamsSize(cudnn, rnn_desc, x_desc[0], &weight_space_size, CUDNN_DATA_FLOAT));

    printf("Weight space size: %.2f MB\n\n", weight_space_size / 1024.0 / 1024.0);

    // Allocate memory
    size_t input_bytes = seq_length * batch_size * input_size * sizeof(float);
    size_t output_bytes = seq_length * batch_size * hidden_size * sizeof(float);
    size_t hidden_bytes = num_layers * batch_size * hidden_size * sizeof(float);

    float *h_input = (float*)malloc(input_bytes);
    float *h_output = (float*)malloc(output_bytes);

    // Initialize input with random data
    for (size_t i = 0; i < seq_length * batch_size * input_size; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;
    }

    float *d_input, *d_output;
    float *d_hx, *d_cx, *d_hy, *d_cy;
    void *d_weights;

    CHECK_CUDA(cudaMalloc(&d_input, input_bytes));
    CHECK_CUDA(cudaMalloc(&d_output, output_bytes));
    CHECK_CUDA(cudaMalloc(&d_hx, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_cx, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_hy, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_cy, hidden_bytes));
    CHECK_CUDA(cudaMalloc(&d_weights, weight_space_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hx, 0, hidden_bytes));
    CHECK_CUDA(cudaMemset(d_cx, 0, hidden_bytes));
    CHECK_CUDA(cudaMemset(d_weights, 0, weight_space_size));

    // Get workspace size
    size_t workspace_size;
    CHECK_CUDNN(cudnnGetRNNWorkspaceSize(cudnn, rnn_desc, seq_length, x_desc, &workspace_size));

    void *d_workspace;
    CHECK_CUDA(cudaMalloc(&d_workspace, workspace_size));

    printf("Workspace size: %.2f MB\n\n", workspace_size / 1024.0 / 1024.0);

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Create array of input/output pointers
    float **d_x = (float**)malloc(seq_length * sizeof(float*));
    float **d_y = (float**)malloc(seq_length * sizeof(float*));

    for (int i = 0; i < seq_length; i++) {
        d_x[i] = d_input + i * batch_size * input_size;
        d_y[i] = d_output + i * batch_size * hidden_size;
    }

    // Warm-up
    CHECK_CUDNN(cudnnRNNForwardInference(cudnn, rnn_desc,
                                         seq_length,
                                         x_desc, d_input,
                                         h_desc, d_hx,
                                         c_desc, d_cx,
                                         rnn_desc, d_weights,
                                         y_desc, d_output,
                                         h_desc, d_hy,
                                         c_desc, d_cy,
                                         d_workspace, workspace_size));

    // Forward pass
    CHECK_CUDA(cudaEventRecord(start));

    CHECK_CUDNN(cudnnRNNForwardInference(cudnn, rnn_desc,
                                         seq_length,
                                         x_desc, d_input,
                                         h_desc, d_hx,
                                         c_desc, d_cx,
                                         rnn_desc, d_weights,
                                         y_desc, d_output,
                                         h_desc, d_hy,
                                         c_desc, d_cy,
                                         d_workspace, workspace_size));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy results
    CHECK_CUDA(cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost));

    // Performance metrics
    printf("=== Performance ===\n");
    printf("Forward pass time: %.3f ms\n", milliseconds);
    printf("Sequences/sec:     %.2f\n", batch_size / (milliseconds / 1000.0));
    printf("Throughput:        %.2f M elements/sec\n",
           (seq_length * batch_size * hidden_size) / (milliseconds * 1000.0));

    // Sample output
    printf("\nOutput sample (first timestep, first 10 hidden units):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_output[i]);
    }

    printf("\n\nOutput sample (last timestep, first 10 hidden units):\n");
    int last_timestep = (seq_length - 1) * batch_size * hidden_size;
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_output[last_timestep + i]);
    }
    printf("\n");

    // Cleanup
    for (int i = 0; i < seq_length; i++) {
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(x_desc[i]));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(y_desc[i]));
    }
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(h_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(c_desc));
    CHECK_CUDNN(cudnnDestroyRNNDescriptor(rnn_desc));
    CHECK_CUDNN(cudnnDestroyDropoutDescriptor(dropout_desc));
    CHECK_CUDNN(cudnnDestroy(cudnn));

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_hx));
    CHECK_CUDA(cudaFree(d_cx));
    CHECK_CUDA(cudaFree(d_hy));
    CHECK_CUDA(cudaFree(d_cy));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_workspace));
    CHECK_CUDA(cudaFree(dropout_states));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(x_desc);
    free(y_desc);
    free(d_x);
    free(d_y);
    free(h_input);
    free(h_output);

    printf("\ncuDNN RNN/LSTM completed successfully!\n");
    return 0;
}
