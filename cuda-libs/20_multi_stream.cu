// Multi-Stream Concurrent Execution: Maximize GPU utilization
// Critical for overlapping computation, memory transfers, and pipelining

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

// Simple kernel for demonstration
__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

// Computation-intensive kernel
__global__ void matrixCompute(float* data, int n, int iterations) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float value = data[tid];
        for (int i = 0; i < iterations; i++) {
            value = sinf(value) * cosf(value) + value * 0.5f;
        }
        data[tid] = value;
    }
}

int main() {
    const int NUM_STREAMS = 4;
    const int N = 1 << 24;  // 16M elements per stream
    const int bytes_per_stream = N * sizeof(float);

    printf("=== Multi-Stream Concurrent Execution ===\n");
    printf("Number of streams: %d\n", NUM_STREAMS);
    printf("Elements per stream: %d\n", N);
    printf("Memory per stream: %.2f MB\n\n", bytes_per_stream / 1024.0 / 1024.0);

    // Create streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Allocate pinned host memory for faster transfers
    float *h_a[NUM_STREAMS], *h_b[NUM_STREAMS], *h_c[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMallocHost(&h_a[i], bytes_per_stream));
        CHECK_CUDA(cudaMallocHost(&h_b[i], bytes_per_stream));
        CHECK_CUDA(cudaMallocHost(&h_c[i], bytes_per_stream));

        // Initialize data
        for (int j = 0; j < N; j++) {
            h_a[i][j] = (float)(rand() % 100);
            h_b[i][j] = (float)(rand() % 100);
        }
    }

    // Allocate device memory
    float *d_a[NUM_STREAMS], *d_b[NUM_STREAMS], *d_c[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMalloc(&d_a[i], bytes_per_stream));
        CHECK_CUDA(cudaMalloc(&d_b[i], bytes_per_stream));
        CHECK_CUDA(cudaMalloc(&d_c[i], bytes_per_stream));
    }

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ===== SEQUENTIAL EXECUTION (Baseline) =====
    printf("=== Sequential Execution (No Streams) ===\n");

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMemcpy(d_a[i], h_a[i], bytes_per_stream, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b[i], h_b[i], bytes_per_stream, cudaMemcpyHostToDevice));
        vectorAdd<<<gridSize, blockSize>>>(d_a[i], d_b[i], d_c[i], N);
        CHECK_CUDA(cudaMemcpy(h_c[i], d_c[i], bytes_per_stream, cudaMemcpyDeviceToHost));
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_sequential = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sequential, start, stop));

    printf("Total time: %.3f ms\n\n", ms_sequential);

    // ===== CONCURRENT EXECUTION WITH STREAMS =====
    printf("=== Concurrent Execution (Multi-Stream) ===\n");

    CHECK_CUDA(cudaEventRecord(start));

    // Launch all operations asynchronously
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMemcpyAsync(d_a[i], h_a[i], bytes_per_stream,
                                   cudaMemcpyHostToDevice, streams[i]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[i], h_b[i], bytes_per_stream,
                                   cudaMemcpyHostToDevice, streams[i]));

        vectorAdd<<<gridSize, blockSize, 0, streams[i]>>>(d_a[i], d_b[i], d_c[i], N);

        CHECK_CUDA(cudaMemcpyAsync(h_c[i], d_c[i], bytes_per_stream,
                                   cudaMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams to complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_concurrent = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_concurrent, start, stop));

    printf("Total time: %.3f ms\n", ms_concurrent);
    printf("Speedup: %.2fx\n\n", ms_sequential / ms_concurrent);

    // ===== PIPELINED EXECUTION =====
    printf("=== Pipelined Execution (Staged) ===\n");

    const int NUM_CHUNKS = 8;
    const int chunk_size = N / NUM_CHUNKS;
    const int chunk_bytes = chunk_size * sizeof(float);

    CHECK_CUDA(cudaEventRecord(start));

    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        int stream_id = chunk % NUM_STREAMS;
        int offset = chunk * chunk_size;

        CHECK_CUDA(cudaMemcpyAsync(d_a[stream_id] + offset, h_a[stream_id] + offset,
                                   chunk_bytes, cudaMemcpyHostToDevice, streams[stream_id]));
        CHECK_CUDA(cudaMemcpyAsync(d_b[stream_id] + offset, h_b[stream_id] + offset,
                                   chunk_bytes, cudaMemcpyHostToDevice, streams[stream_id]));

        int chunk_grid = (chunk_size + blockSize - 1) / blockSize;
        vectorAdd<<<chunk_grid, blockSize, 0, streams[stream_id]>>>(
            d_a[stream_id] + offset, d_b[stream_id] + offset, d_c[stream_id] + offset, chunk_size);

        CHECK_CUDA(cudaMemcpyAsync(h_c[stream_id] + offset, d_c[stream_id] + offset,
                                   chunk_bytes, cudaMemcpyDeviceToHost, streams[stream_id]));
    }

    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_pipelined = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_pipelined, start, stop));

    printf("Total time: %.3f ms\n", ms_pipelined);
    printf("Speedup: %.2fx\n\n", ms_sequential / ms_pipelined);

    // ===== STREAM PRIORITIES =====
    printf("=== Stream Priorities ===\n");

    int priority_low, priority_high;
    CHECK_CUDA(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));

    cudaStream_t high_priority_stream, low_priority_stream;
    CHECK_CUDA(cudaStreamCreateWithPriority(&high_priority_stream, cudaStreamNonBlocking, priority_high));
    CHECK_CUDA(cudaStreamCreateWithPriority(&low_priority_stream, cudaStreamNonBlocking, priority_low));

    printf("Priority range: %d (high) to %d (low)\n", priority_high, priority_low);

    // Launch computation-heavy kernel on low priority
    matrixCompute<<<gridSize, blockSize, 0, low_priority_stream>>>(d_a[0], N, 100);

    // Launch quick kernel on high priority
    vectorAdd<<<gridSize, blockSize, 0, high_priority_stream>>>(d_a[1], d_b[1], d_c[1], N);

    CHECK_CUDA(cudaStreamSynchronize(high_priority_stream));
    CHECK_CUDA(cudaStreamSynchronize(low_priority_stream));

    printf("Stream priorities demonstrated successfully!\n\n");

    // ===== STREAM DEPENDENCIES (EVENTS) =====
    printf("=== Stream Dependencies with Events ===\n");

    cudaEvent_t event1, event2;
    CHECK_CUDA(cudaEventCreate(&event1));
    CHECK_CUDA(cudaEventCreate(&event2));

    CHECK_CUDA(cudaEventRecord(start));

    // Stream 0: First operation
    vectorAdd<<<gridSize, blockSize, 0, streams[0]>>>(d_a[0], d_b[0], d_c[0], N);
    CHECK_CUDA(cudaEventRecord(event1, streams[0]));

    // Stream 1: Wait for stream 0 to complete
    CHECK_CUDA(cudaStreamWaitEvent(streams[1], event1, 0));
    vectorAdd<<<gridSize, blockSize, 0, streams[1]>>>(d_c[0], d_b[1], d_c[1], N);
    CHECK_CUDA(cudaEventRecord(event2, streams[1]));

    // Stream 2: Wait for stream 1 to complete
    CHECK_CUDA(cudaStreamWaitEvent(streams[2], event2, 0));
    vectorAdd<<<gridSize, blockSize, 0, streams[2]>>>(d_c[1], d_b[2], d_c[2], N);

    CHECK_CUDA(cudaStreamSynchronize(streams[2]));

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_dependent = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_dependent, start, stop));

    printf("Dependent stream execution: %.3f ms\n\n", ms_dependent);

    // Performance summary
    printf("=== Performance Summary ===\n");
    printf("Sequential:   %.3f ms\n", ms_sequential);
    printf("Concurrent:   %.3f ms (%.2fx speedup)\n", ms_concurrent, ms_sequential / ms_concurrent);
    printf("Pipelined:    %.3f ms (%.2fx speedup)\n", ms_pipelined, ms_sequential / ms_pipelined);
    printf("Dependent:    %.3f ms\n", ms_dependent);

    printf("\nTotal data processed: %.2f GB\n",
           (NUM_STREAMS * N * 3 * sizeof(float)) / 1024.0 / 1024.0 / 1024.0);

    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        CHECK_CUDA(cudaFreeHost(h_a[i]));
        CHECK_CUDA(cudaFreeHost(h_b[i]));
        CHECK_CUDA(cudaFreeHost(h_c[i]));
        CHECK_CUDA(cudaFree(d_a[i]));
        CHECK_CUDA(cudaFree(d_b[i]));
        CHECK_CUDA(cudaFree(d_c[i]));
    }

    CHECK_CUDA(cudaStreamDestroy(high_priority_stream));
    CHECK_CUDA(cudaStreamDestroy(low_priority_stream));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaEventDestroy(event1));
    CHECK_CUDA(cudaEventDestroy(event2));

    printf("\nMulti-stream operations completed successfully!\n");
    return 0;
}
