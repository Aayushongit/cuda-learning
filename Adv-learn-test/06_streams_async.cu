#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024*1024
#define BLOCK_SIZE 256
#define NUM_STREAMS 4

__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simulate some computation
        float temp = a[idx] + b[idx];
        for (int i = 0; i < 100; i++) {
            temp = temp * 0.99f + 0.01f;
        }
        c[idx] = temp;
    }
}

__global__ void matrix_vector_multiply(float *matrix, float *vector, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += matrix[idx * n + i] * vector[i];
        }
        result[idx] = sum;
    }
}

void synchronous_execution(float *d_a, float *d_b, float *d_c, int n) {
    printf("=== Synchronous Execution ===\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEventRecord(start);
    
    // All operations execute sequentially
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (n / NUM_STREAMS);
        int chunk_size = n / NUM_STREAMS;
        
        vector_add<<<grid_size/NUM_STREAMS, BLOCK_SIZE>>>(
            d_a + offset, d_b + offset, d_c + offset, chunk_size);
        cudaDeviceSynchronize(); // Wait for completion
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Synchronous time: %.3f ms\n", time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void asynchronous_execution(float *d_a, float *d_b, float *d_c, int n) {
    printf("\n=== Asynchronous Execution with Streams ===\n");
    
    cudaStream_t streams[NUM_STREAMS];
    cudaEvent_t start, stop;
    
    // Create streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int grid_size = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaEventRecord(start);
    
    // Launch kernels asynchronously in different streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * (n / NUM_STREAMS);
        int chunk_size = n / NUM_STREAMS;
        
        vector_add<<<grid_size/NUM_STREAMS, BLOCK_SIZE, 0, streams[i]>>>(
            d_a + offset, d_b + offset, d_c + offset, chunk_size);
    }
    
    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Asynchronous time: %.3f ms\n", time);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void overlapped_memory_compute() {
    printf("\n=== Overlapped Memory and Compute ===\n");
    
    size_t chunk_bytes = (N / NUM_STREAMS) * sizeof(float);
    
    // Host pinned memory for faster transfers
    float *h_a, *h_b, *h_c;
    cudaMallocHost(&h_a, N * sizeof(float));
    cudaMallocHost(&h_b, N * sizeof(float));
    cudaMallocHost(&h_c, N * sizeof(float));
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i + 1);
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(float));
    cudaMalloc(&d_b, N * sizeof(float));
    cudaMalloc(&d_c, N * sizeof(float));
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    int chunk_size = N / NUM_STREAMS;
    int grid_size = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Pipeline: H2D transfer, compute, D2H transfer
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        
        // Asynchronous memory transfers and kernel execution
        cudaMemcpyAsync(d_a + offset, h_a + offset, chunk_bytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(d_b + offset, h_b + offset, chunk_bytes, 
                       cudaMemcpyHostToDevice, streams[i]);
        
        vector_add<<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
            d_a + offset, d_b + offset, d_c + offset, chunk_size);
        
        cudaMemcpyAsync(h_c + offset, d_c + offset, chunk_bytes, 
                       cudaMemcpyDeviceToHost, streams[i]);
    }
    
    // Wait for all operations to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Overlapped execution time: %.3f ms\n", time);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        float expected = h_a[i] + h_b[i];
        if (abs(h_c[i] - expected) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Results %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void stream_priorities_demo() {
    printf("\n=== Stream Priorities Demo ===\n");
    
    cudaStream_t high_priority_stream, low_priority_stream;
    
    // Get priority range
    int lowest, highest;
    cudaDeviceGetStreamPriorityRange(&lowest, &highest);
    printf("Priority range: %d (highest) to %d (lowest)\n", highest, lowest);
    
    // Create streams with different priorities
    cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, highest);
    cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, lowest);
    
    size_t bytes = 1024 * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Launch work on both streams
    vector_add<<<4, 256, 0, high_priority_stream>>>(d_data, d_data, d_data, 1024);
    vector_add<<<4, 256, 0, low_priority_stream>>>(d_data, d_data, d_data, 1024);
    
    printf("High priority work should complete first\n");
    
    cudaStreamSynchronize(high_priority_stream);
    cudaStreamSynchronize(low_priority_stream);
    
    cudaStreamDestroy(high_priority_stream);
    cudaStreamDestroy(low_priority_stream);
    cudaFree(d_data);
}

void stream_callbacks_demo() {
    printf("\n=== Stream Callbacks Demo ===\n");
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    size_t bytes = 1024 * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, bytes);
    
    // Launch kernel
    vector_add<<<4, 256, 0, stream>>>(d_data, d_data, d_data, 1024);
    
    // Add callback (will execute after kernel completes)
    auto callback = [](cudaStream_t stream, cudaError_t status, void *userData) {
        printf("Callback executed! Stream completed with status: %s\n", 
               cudaGetErrorString(status));
    };
    
    cudaStreamAddCallback(stream, callback, nullptr, 0);
    
    cudaStreamSynchronize(stream);
    
    cudaStreamDestroy(stream);
    cudaFree(d_data);
}

int main() {
    printf("=== CUDA Streams and Asynchronous Execution Demo ===\n");
    printf("This example demonstrates various stream optimization techniques.\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Supported" : "Not supported");
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("\n");
    
    size_t bytes = N * sizeof(float);
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i + 1);
    }
    
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);
    
    // Compare synchronous vs asynchronous execution
    synchronous_execution(d_a, d_b, d_c, N);
    asynchronous_execution(d_a, d_b, d_c, N);
    
    // Demonstrate overlapped memory and compute
    overlapped_memory_compute();
    
    // Stream priorities
    stream_priorities_demo();
    
    // Stream callbacks
    stream_callbacks_demo();
    
    printf("\nKey Learnings:\n");
    printf("- Streams enable concurrent kernel execution and memory transfers\n");
    printf("- Pinned memory improves transfer performance\n");
    printf("- Overlapping compute and memory operations improves throughput\n");
    printf("- Stream priorities help manage workload scheduling\n");
    printf("- Callbacks provide asynchronous notification of completion\n");
    printf("- Always check device capabilities for concurrent execution\n");
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    
    return 0;
}