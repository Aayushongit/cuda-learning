// CUDA Graphs: Reduce kernel launch overhead by recording execution patterns
// Can reduce CPU overhead from ~10μs to <1μs per kernel launch

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

__global__ void vectorAdd(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void vectorMul(float* a, float* b, float* c, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        c[tid] = a[tid] * b[tid];
    }
}

__global__ void vectorScale(float* a, float scale, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        a[tid] *= scale;
    }
}

int main() {
    const int N = 1 << 20;  // 1M elements
    const int iterations = 1000;
    const size_t bytes = N * sizeof(float);

    printf("=== CUDA Graphs Optimization ===\n");
    printf("Array size: %d elements\n", N);
    printf("Iterations: %d\n\n", iterations);

    // Allocate memory
    float *d_a, *d_b, *d_c, *d_temp;
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));
    CHECK_CUDA(cudaMalloc(&d_temp, bytes));

    // Initialize
    CHECK_CUDA(cudaMemset(d_a, 1, bytes));
    CHECK_CUDA(cudaMemset(d_b, 2, bytes));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ===== Method 1: Traditional kernel launches =====
    printf("=== Traditional Kernel Launches ===\n");

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_temp, N);
        vectorMul<<<gridSize, blockSize>>>(d_temp, d_b, d_c, N);
        vectorScale<<<gridSize, blockSize>>>(d_c, 0.5f, N);
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_traditional = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_traditional, start, stop));

    printf("Total time: %.3f ms\n", ms_traditional);
    printf("Avg per iteration: %.3f ms\n", ms_traditional / iterations);
    printf("Launch overhead: ~%.2f μs per kernel\n\n",
           (ms_traditional / iterations / 3.0) * 1000.0);

    // ===== Method 2: CUDA Graphs with Stream Capture =====
    printf("=== CUDA Graphs (Stream Capture) ===\n");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // Capture the sequence of operations
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    vectorAdd<<<gridSize, blockSize, 0, stream>>>(d_a, d_b, d_temp, N);
    vectorMul<<<gridSize, blockSize, 0, stream>>>(d_temp, d_b, d_c, N);
    vectorScale<<<gridSize, blockSize, 0, stream>>>(d_c, 0.5f, N);

    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Benchmark graph execution
    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_graph = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_graph, start, stop));

    printf("Total time: %.3f ms\n", ms_graph);
    printf("Avg per iteration: %.3f ms\n", ms_graph / iterations);
    printf("Launch overhead: ~%.2f μs per graph\n", (ms_graph / iterations) * 1000.0);
    printf("Speedup: %.2fx\n\n", ms_traditional / ms_graph);

    // ===== Method 3: Manual Graph Construction =====
    printf("=== CUDA Graphs (Manual Construction) ===\n");

    cudaGraph_t manual_graph;
    cudaGraphExec_t manual_graphExec;
    CHECK_CUDA(cudaGraphCreate(&manual_graph, 0));

    // Create nodes manually
    cudaGraphNode_t addNode, mulNode, scaleNode;
    cudaKernelNodeParams addParams = {0};
    cudaKernelNodeParams mulParams = {0};
    cudaKernelNodeParams scaleParams = {0};

    // Configure add kernel
    void* addArgs[] = {&d_a, &d_b, &d_temp, &N};
    addParams.func = (void*)vectorAdd;
    addParams.gridDim = dim3(gridSize);
    addParams.blockDim = dim3(blockSize);
    addParams.kernelParams = addArgs;

    // Configure mul kernel
    void* mulArgs[] = {&d_temp, &d_b, &d_c, &N};
    mulParams.func = (void*)vectorMul;
    mulParams.gridDim = dim3(gridSize);
    mulParams.blockDim = dim3(blockSize);
    mulParams.kernelParams = mulArgs;

    // Configure scale kernel
    float scale = 0.5f;
    void* scaleArgs[] = {&d_c, &scale, &N};
    scaleParams.func = (void*)vectorScale;
    scaleParams.gridDim = dim3(gridSize);
    scaleParams.blockDim = dim3(blockSize);
    scaleParams.kernelParams = scaleArgs;

    // Add nodes to graph
    CHECK_CUDA(cudaGraphAddKernelNode(&addNode, manual_graph, NULL, 0, &addParams));
    CHECK_CUDA(cudaGraphAddKernelNode(&mulNode, manual_graph, &addNode, 1, &mulParams));
    CHECK_CUDA(cudaGraphAddKernelNode(&scaleNode, manual_graph, &mulNode, 1, &scaleParams));

    // Instantiate and execute
    CHECK_CUDA(cudaGraphInstantiate(&manual_graphExec, manual_graph, NULL, NULL, 0));

    CHECK_CUDA(cudaEventRecord(start));

    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(manual_graphExec, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaStreamSynchronize(stream));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_manual = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_manual, start, stop));

    printf("Total time: %.3f ms\n", ms_manual);
    printf("Avg per iteration: %.3f ms\n", ms_manual / iterations);
    printf("Speedup: %.2fx\n\n", ms_traditional / ms_manual);

    // ===== Method 4: Graph with Memory Operations =====
    printf("=== CUDA Graphs with Memory Operations ===\n");

    float *h_result = (float*)malloc(10 * sizeof(float));

    cudaGraph_t mem_graph;
    cudaGraphExec_t mem_graphExec;
    cudaStream_t mem_stream;
    CHECK_CUDA(cudaStreamCreate(&mem_stream));

    CHECK_CUDA(cudaStreamBeginCapture(mem_stream, cudaStreamCaptureModeGlobal));

    // Compute
    vectorAdd<<<gridSize, blockSize, 0, mem_stream>>>(d_a, d_b, d_c, N);

    // Copy result back (small sample)
    CHECK_CUDA(cudaMemcpyAsync(h_result, d_c, 10 * sizeof(float),
                               cudaMemcpyDeviceToHost, mem_stream));

    CHECK_CUDA(cudaStreamEndCapture(mem_stream, &mem_graph));
    CHECK_CUDA(cudaGraphInstantiate(&mem_graphExec, mem_graph, NULL, NULL, 0));

    CHECK_CUDA(cudaGraphLaunch(mem_graphExec, mem_stream));
    CHECK_CUDA(cudaStreamSynchronize(mem_stream));

    printf("Graph with memory operations executed successfully\n");
    printf("Sample results: %.2f, %.2f, %.2f\n\n", h_result[0], h_result[1], h_result[2]);

    // ===== Graph Update (Parameter Change) =====
    printf("=== Graph Update (Changing Parameters) ===\n");

    // Update scale parameter in existing graph
    float new_scale = 2.0f;
    void* new_scaleArgs[] = {&d_c, &new_scale, &N};
    scaleParams.kernelParams = new_scaleArgs;

    CHECK_CUDA(cudaGraphExecKernelNodeSetParams(graphExec, scaleNode, &scaleParams));

    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    printf("Graph parameters updated successfully\n\n");

    // ===== Performance Summary =====
    printf("=== Performance Summary ===\n");
    printf("Traditional:      %.3f ms (1.00x)\n", ms_traditional);
    printf("Graph (capture):  %.3f ms (%.2fx faster)\n",
           ms_graph, ms_traditional / ms_graph);
    printf("Graph (manual):   %.3f ms (%.2fx faster)\n",
           ms_manual, ms_traditional / ms_manual);

    printf("\n=== Benefits of CUDA Graphs ===\n");
    printf("✓ Reduced CPU overhead (10μs → <1μs)\n");
    printf("✓ Better optimization opportunities\n");
    printf("✓ Predictable execution patterns\n");
    printf("✓ Ideal for inference pipelines\n");
    printf("✓ Can update parameters without reconstruction\n");

    printf("\n=== Use Cases ===\n");
    printf("• Fixed topology inference pipelines\n");
    printf("• Repeated training iterations\n");
    printf("• Real-time applications\n");
    printf("• Minimizing launch latency\n");

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphExecDestroy(manual_graphExec));
    CHECK_CUDA(cudaGraphExecDestroy(mem_graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaGraphDestroy(manual_graph));
    CHECK_CUDA(cudaGraphDestroy(mem_graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaStreamDestroy(mem_stream));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    free(h_result);

    printf("\nCUDA Graphs completed successfully!\n");
    return 0;
}
