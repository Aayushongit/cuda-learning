#include <cuda_runtime.h>
#include <stdio.h>
#include <omp.h>

#define N 1024*1024
#define BLOCK_SIZE 256

// Simple kernel for multi-GPU demonstrations
__global__ void vector_add(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void matrix_multiply_chunk(float *A, float *B, float *C, 
                                     int n, int chunk_start, int chunk_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + chunk_start;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < chunk_start + chunk_size && row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

__global__ void reduction_kernel(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

void query_gpu_devices() {
    printf("=== GPU Device Query ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    printf("Number of CUDA devices: %d\n\n", device_count);
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        printf("Device %d: %s\n", i, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Memory Clock Rate: %.1f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Can Access Peer: ");
        
        for (int j = 0; j < device_count; j++) {
            if (i != j) {
                int can_access;
                cudaDeviceCanAccessPeer(&can_access, i, j);
                printf("GPU%d:%s ", j, can_access ? "Yes" : "No");
            }
        }
        printf("\n\n");
    }
}

void demonstrate_basic_multi_gpu() {
    printf("=== Basic Multi-GPU Vector Addition ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count < 2) {
        printf("This demo requires at least 2 GPUs. Found %d GPU(s).\n", device_count);
        printf("Simulating multi-GPU behavior on single GPU...\n\n");
        device_count = 1;
    }
    
    size_t bytes = N * sizeof(float);
    int chunk_size = N / device_count;
    
    // Host data
    float *h_a = (float*)malloc(bytes);
    float *h_b = (float*)malloc(bytes);
    float *h_c = (float*)malloc(bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)i;
        h_b[i] = (float)(i + 1);
    }
    
    // Device data arrays
    float **d_a = (float**)malloc(device_count * sizeof(float*));
    float **d_b = (float**)malloc(device_count * sizeof(float*));
    float **d_c = (float**)malloc(device_count * sizeof(float*));
    
    cudaStream_t *streams = (cudaStream_t*)malloc(device_count * sizeof(cudaStream_t));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate memory on each GPU
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaMalloc(&d_a[gpu], chunk_size * sizeof(float));
        cudaMalloc(&d_b[gpu], chunk_size * sizeof(float));
        cudaMalloc(&d_c[gpu], chunk_size * sizeof(float));
        cudaStreamCreate(&streams[gpu]);
    }
    
    cudaEventRecord(start);
    
    // Launch work on each GPU
    #pragma omp parallel for
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        
        int offset = gpu * chunk_size;
        int current_chunk = (gpu == device_count - 1) ? N - offset : chunk_size;
        
        // Copy data to GPU
        cudaMemcpyAsync(d_a[gpu], h_a + offset, current_chunk * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[gpu]);
        cudaMemcpyAsync(d_b[gpu], h_b + offset, current_chunk * sizeof(float), 
                       cudaMemcpyHostToDevice, streams[gpu]);
        
        // Launch kernel
        int grid_size = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<grid_size, BLOCK_SIZE, 0, streams[gpu]>>>(
            d_a[gpu], d_b[gpu], d_c[gpu], current_chunk);
        
        // Copy result back
        cudaMemcpyAsync(h_c + offset, d_c[gpu], current_chunk * sizeof(float), 
                       cudaMemcpyDeviceToHost, streams[gpu]);
    }
    
    // Synchronize all GPUs
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamSynchronize(streams[gpu]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    printf("Multi-GPU vector addition time: %.3f ms\n", time);
    
    // Verify results
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    // Cleanup
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_a[gpu]);
        cudaFree(d_b[gpu]);
        cudaFree(d_c[gpu]);
        cudaStreamDestroy(streams[gpu]);
    }
    
    free(h_a);
    free(h_b);
    free(h_c);
    free(d_a);
    free(d_b);
    free(d_c);
    free(streams);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_peer_to_peer() {
    printf("\n=== Peer-to-Peer Memory Access ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count < 2) {
        printf("P2P demo requires at least 2 GPUs\n");
        return;
    }
    
    // Check P2P access capability
    int can_access_peer;
    cudaDeviceCanAccessPeer(&can_access_peer, 0, 1);
    
    if (!can_access_peer) {
        printf("P2P access not supported between GPU 0 and GPU 1\n");
        return;
    }
    
    printf("Enabling P2P access between GPU 0 and GPU 1\n");
    
    // Enable P2P access
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    
    size_t bytes = 1024 * 1024 * sizeof(float);
    float *d_data0, *d_data1;
    
    // Allocate memory on both GPUs
    cudaSetDevice(0);
    cudaMalloc(&d_data0, bytes);
    
    cudaSetDevice(1);
    cudaMalloc(&d_data1, bytes);
    
    // Initialize data on GPU 0
    cudaSetDevice(0);
    cudaMemset(d_data0, 0, bytes);
    
    // Copy from GPU 0 to GPU 1 using P2P
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    cudaMemcpyPeer(d_data1, 1, d_data0, 0, bytes);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float p2p_time;
    cudaEventElapsedTime(&p2p_time, start, stop);
    printf("P2P copy time: %.3f ms\n", p2p_time);
    printf("P2P bandwidth: %.2f GB/s\n", (bytes / (1024.0*1024.0*1024.0)) / (p2p_time / 1000.0));
    
    // Compare with regular host-device copy
    float *h_temp = (float*)malloc(bytes);
    
    cudaEventRecord(start);
    cudaSetDevice(0);
    cudaMemcpy(h_temp, d_data0, bytes, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(d_data1, h_temp, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float host_time;
    cudaEventElapsedTime(&host_time, start, stop);
    printf("Host-mediated copy time: %.3f ms\n", host_time);
    printf("P2P speedup: %.2fx\n", host_time / p2p_time);
    
    // Cleanup
    cudaSetDevice(0);
    cudaFree(d_data0);
    cudaSetDevice(1);
    cudaFree(d_data1);
    free(h_temp);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_multi_gpu_reduction() {
    printf("\n=== Multi-GPU Parallel Reduction ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    device_count = min(device_count, 4); // Limit for demo
    
    size_t bytes = N * sizeof(float);
    int chunk_size = N / device_count;
    
    float *h_data = (float*)malloc(bytes);
    
    // Initialize with ones for easy verification
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    float **d_data = (float**)malloc(device_count * sizeof(float*));
    float **d_partial = (float**)malloc(device_count * sizeof(float*));
    float *h_partial = (float*)malloc(device_count * sizeof(float));
    
    // Allocate and initialize data on each GPU
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        
        int current_chunk = (gpu == device_count - 1) ? N - gpu * chunk_size : chunk_size;
        
        cudaMalloc(&d_data[gpu], current_chunk * sizeof(float));
        cudaMalloc(&d_partial[gpu], sizeof(float));
        
        cudaMemcpy(d_data[gpu], h_data + gpu * chunk_size, 
                  current_chunk * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Perform reduction on each GPU
    #pragma omp parallel for
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        
        int current_chunk = (gpu == device_count - 1) ? N - gpu * chunk_size : chunk_size;
        int grid_size = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        // Multi-level reduction
        float *d_temp;
        cudaMalloc(&d_temp, grid_size * sizeof(float));
        
        reduction_kernel<<<grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
            d_data[gpu], d_temp, current_chunk);
        
        // Final reduction for this GPU
        while (grid_size > 1) {
            int new_grid_size = (grid_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
            reduction_kernel<<<new_grid_size, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(
                d_temp, d_temp, grid_size);
            grid_size = new_grid_size;
        }
        
        cudaMemcpy(&h_partial[gpu], d_temp, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_temp);
    }
    
    // Final reduction on CPU
    float final_result = 0.0f;
    for (int gpu = 0; gpu < device_count; gpu++) {
        final_result += h_partial[gpu];
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Multi-GPU reduction result: %.0f (expected: %d)\n", final_result, N);
    printf("Multi-GPU reduction time: %.3f ms\n", time);
    printf("Using %d GPU(s)\n", device_count);
    
    // Cleanup
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        cudaFree(d_data[gpu]);
        cudaFree(d_partial[gpu]);
    }
    
    free(h_data);
    free(d_data);
    free(d_partial);
    free(h_partial);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_unified_memory_multi_gpu() {
    printf("\n=== Unified Memory with Multi-GPU ===\n");
    
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    // Check unified memory support
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    if (!prop.managedMemory) {
        printf("Unified Memory not supported on this device\n");
        return;
    }
    
    size_t bytes = N * sizeof(float);
    float *data, *result;
    
    // Allocate unified memory
    cudaMallocManaged(&data, bytes);
    cudaMallocManaged(&result, bytes);
    
    // Initialize data
    for (int i = 0; i < N; i++) {
        data[i] = (float)i;
        result[i] = 0.0f;
    }
    
    int chunk_size = N / device_count;
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    
    // Process data on multiple GPUs
    #pragma omp parallel for
    for (int gpu = 0; gpu < device_count; gpu++) {
        cudaSetDevice(gpu);
        
        int offset = gpu * chunk_size;
        int current_chunk = (gpu == device_count - 1) ? N - offset : chunk_size;
        
        // Prefetch data to this GPU
        cudaMemPrefetchAsync(data + offset, current_chunk * sizeof(float), gpu);
        cudaMemPrefetchAsync(result + offset, current_chunk * sizeof(float), gpu);
        
        int grid_size = (current_chunk + BLOCK_SIZE - 1) / BLOCK_SIZE;
        vector_add<<<grid_size, BLOCK_SIZE>>>(
            data + offset, data + offset, result + offset, current_chunk);
        
        cudaDeviceSynchronize();
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    
    printf("Unified Memory multi-GPU processing time: %.3f ms\n", time);
    
    // Verify results (first few elements)
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        if (abs(result[i] - (2.0f * data[i])) > 0.001f) {
            correct = false;
            break;
        }
    }
    printf("Results: %s\n", correct ? "CORRECT" : "INCORRECT");
    
    cudaFree(data);
    cudaFree(result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== CUDA Multi-GPU Programming Demo ===\n");
    printf("This example demonstrates various multi-GPU programming techniques.\n\n");
    
    // Query available GPUs
    query_gpu_devices();
    
    // Basic multi-GPU operations
    demonstrate_basic_multi_gpu();
    
    // Peer-to-peer memory access
    demonstrate_peer_to_peer();
    
    // Multi-GPU reduction
    demonstrate_multi_gpu_reduction();
    
    // Unified memory with multiple GPUs
    demonstrate_unified_memory_multi_gpu();
    
    printf("\nKey Learnings:\n");
    printf("- Multi-GPU programming enables scaling beyond single GPU limits\n");
    printf("- Each GPU has its own context and memory space\n");
    printf("- Peer-to-peer access can eliminate host memory bottlenecks\n");
    printf("- Work distribution requires careful load balancing\n");
    printf("- Unified Memory simplifies multi-GPU memory management\n");
    printf("- OpenMP can help manage CPU threads for GPU coordination\n");
    printf("- Always check device capabilities before using advanced features\n");
    printf("- Consider NUMA topology for optimal performance\n");
    
    return 0;
}