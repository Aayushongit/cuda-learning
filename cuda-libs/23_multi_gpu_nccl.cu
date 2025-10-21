// Multi-GPU Programming with NCCL: Efficient collective communication
// Essential for distributed training and multi-GPU inference

#include <cuda_runtime.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_NCCL(call) { \
    ncclResult_t res = call; \
    if (res != ncclSuccess) { \
        fprintf(stderr, "NCCL Error: %s (line %d)\n", ncclGetErrorString(res), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void initKernel(float* data, int n, int offset) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = offset + tid;
    }
}

__global__ void gradientKernel(float* data, int n, float lr) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        data[tid] = data[tid] * lr + 0.01f;
    }
}

int main() {
    int num_gpus = 0;
    CHECK_CUDA(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        printf("This example requires at least 2 GPUs\n");
        printf("Found: %d GPU(s)\n", num_gpus);
        printf("Running in single-GPU demo mode...\n\n");
        num_gpus = 1;
    } else {
        printf("=== Multi-GPU Programming with NCCL ===\n");
        printf("Number of GPUs: %d\n\n", num_gpus);
    }

    // Print GPU information
    for (int i = 0; i < num_gpus; i++) {
        cudaDeviceProp prop;
        CHECK_CUDA(cudaGetDeviceProperties(&prop, i));
        printf("GPU %d: %s (%.2f GB)\n", i, prop.name,
               prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    }
    printf("\n");

    const int N = 1 << 24;  // 16M elements per GPU
    const size_t bytes = N * sizeof(float);

    if (num_gpus == 1) {
        printf("Single GPU demonstration - skipping NCCL operations\n");
        return 0;
    }

    // Allocate arrays for each GPU
    float **d_data = (float**)malloc(num_gpus * sizeof(float*));
    float **d_recv = (float**)malloc(num_gpus * sizeof(float*));
    cudaStream_t *streams = (cudaStream_t*)malloc(num_gpus * sizeof(cudaStream_t));

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_data[i], bytes));
        CHECK_CUDA(cudaMalloc(&d_recv[i], bytes));
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Initialize data on each GPU
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        initKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_data[i], N, i * 1000);
    }

    // Initialize NCCL
    ncclComm_t *comms = (ncclComm_t*)malloc(num_gpus * sizeof(ncclComm_t));
    CHECK_NCCL(ncclCommInitAll(comms, num_gpus, NULL));

    printf("=== NCCL Collective Operations ===\n\n");

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // ===== 1. ALL-REDUCE (sum across all GPUs) =====
    printf("1. AllReduce (sum gradients across GPUs)\n");

    // Simulate gradients on each GPU
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        gradientKernel<<<gridSize, blockSize, 0, streams[i]>>>(d_data[i], N, 0.01f);
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    // Start NCCL group call
    CHECK_NCCL(ncclGroupStart());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclAllReduce(d_data[i], d_recv[i], N, ncclFloat, ncclSum, comms[i], streams[i]));
    }

    CHECK_NCCL(ncclGroupEnd());

    // Wait for completion
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_allreduce = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_allreduce, start, stop));

    double bandwidth_allreduce = (2.0 * bytes * (num_gpus - 1) / num_gpus) / (ms_allreduce / 1000.0) / 1e9;

    printf("   Time: %.3f ms\n", ms_allreduce);
    printf("   Bandwidth: %.2f GB/s\n\n", bandwidth_allreduce);

    // ===== 2. BROADCAST (GPU 0 to all others) =====
    printf("2. Broadcast (GPU 0 to all others)\n");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    CHECK_NCCL(ncclGroupStart());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclBroadcast(d_data[0], d_recv[i], N, ncclFloat, 0, comms[i], streams[i]));
    }

    CHECK_NCCL(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_broadcast = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_broadcast, start, stop));

    double bandwidth_broadcast = (bytes * (num_gpus - 1)) / (ms_broadcast / 1000.0) / 1e9;

    printf("   Time: %.3f ms\n", ms_broadcast);
    printf("   Bandwidth: %.2f GB/s\n\n", bandwidth_broadcast);

    // ===== 3. REDUCE (all to GPU 0) =====
    printf("3. Reduce (sum to GPU 0)\n");

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    CHECK_NCCL(ncclGroupStart());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclReduce(d_data[i], d_recv[0], N, ncclFloat, ncclSum, 0, comms[i], streams[i]));
    }

    CHECK_NCCL(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_reduce = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_reduce, start, stop));

    printf("   Time: %.3f ms\n\n", ms_reduce);

    // ===== 4. ALL-GATHER (gather from all GPUs) =====
    printf("4. AllGather (collect data from all GPUs)\n");

    int chunk_size = N / num_gpus;
    float **d_gathered = (float**)malloc(num_gpus * sizeof(float*));

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaMalloc(&d_gathered[i], bytes));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(start, streams[0]));

    CHECK_NCCL(ncclGroupStart());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_NCCL(ncclAllGather(d_data[i] + i * chunk_size,
                                 d_gathered[i],
                                 chunk_size,
                                 ncclFloat,
                                 comms[i],
                                 streams[i]));
    }

    CHECK_NCCL(ncclGroupEnd());

    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaEventRecord(stop, streams[0]));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_allgather = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_allgather, start, stop));

    printf("   Time: %.3f ms\n\n", ms_allgather);

    // ===== Summary =====
    printf("=== Performance Summary ===\n");
    printf("AllReduce:  %.3f ms (%.2f GB/s) - Data parallel training\n",
           ms_allreduce, bandwidth_allreduce);
    printf("Broadcast:  %.3f ms (%.2f GB/s) - Model distribution\n",
           ms_broadcast, bandwidth_broadcast);
    printf("Reduce:     %.3f ms - Gradient aggregation\n", ms_reduce);
    printf("AllGather:  %.3f ms - Tensor parallel\n", ms_allgather);

    printf("\n=== Use Cases ===\n");
    printf("AllReduce:  Synchronous SGD, gradient averaging\n");
    printf("Broadcast:  Model initialization, parameter sync\n");
    printf("Reduce:     Loss aggregation, metric collection\n");
    printf("AllGather:  Tensor parallelism, large model training\n");

    // Cleanup
    for (int i = 0; i < num_gpus; i++) {
        CHECK_CUDA(cudaSetDevice(i));
        CHECK_CUDA(cudaFree(d_data[i]));
        CHECK_CUDA(cudaFree(d_recv[i]));
        CHECK_CUDA(cudaFree(d_gathered[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }

    free(d_data);
    free(d_recv);
    free(d_gathered);
    free(streams);
    free(comms);
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nMulti-GPU NCCL operations completed successfully!\n");
    return 0;
}
