#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024*1024
#define BLOCK_SIZE 256
#define NUM_BINS 256

// Histogram using global memory atomics
__global__ void histogram_global_atomics(int *data, int *histogram, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % NUM_BINS;
        atomicAdd(&histogram[bin], 1);
    }
}

// Histogram using shared memory atomics
__global__ void histogram_shared_atomics(int *data, int *histogram, int n) {
    extern __shared__ int shared_histogram[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Initialize shared histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    __syncthreads();
    
    // Compute histogram in shared memory
    for (int i = idx; i < n; i += stride) {
        int bin = data[i] % NUM_BINS;
        atomicAdd(&shared_histogram[bin], 1);
    }
    __syncthreads();
    
    // Add shared histogram to global histogram
    for (int i = tid; i < NUM_BINS; i += blockDim.x) {
        atomicAdd(&histogram[i], shared_histogram[i]);
    }
}

// Atomic operations showcase
__global__ void atomic_operations_demo() {
    __shared__ int shared_counter;
    __shared__ float shared_sum;
    __shared__ int shared_max;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        shared_counter = 0;
        shared_sum = 0.0f;
        shared_max = 0;
    }
    __syncthreads();
    
    // Atomic addition
    atomicAdd(&shared_counter, 1);
    
    // Atomic float addition (compute capability 2.0+)
    atomicAdd(&shared_sum, (float)tid);
    
    // Atomic maximum
    atomicMax(&shared_max, tid);
    
    // Atomic exchange
    int old_val = atomicExch(&shared_counter, tid);
    
    // Atomic compare-and-swap
    int expected = tid;
    int desired = tid * 2;
    atomicCAS(&shared_counter, expected, desired);
    
    __syncthreads();
    
    if (tid == 0) {
        printf("Block %d results:\n", blockIdx.x);
        printf("  Final counter: %d\n", shared_counter);
        printf("  Sum: %.1f\n", shared_sum);
        printf("  Max: %d\n", shared_max);
    }
}

// Lock-free data structure example
__global__ void lock_free_stack_demo(int *stack, int *top, int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Push operation using atomic compare-and-swap
    int value = data[idx];
    int old_top, new_top;
    
    do {
        old_top = *top;
        new_top = old_top + 1;
        stack[new_top] = value;
        // Atomically update top if it hasn't changed
    } while (atomicCAS(top, old_top, new_top) != old_top);
    
    if (idx == 0) {
        printf("Pushed %d elements to lock-free stack\n", n);
    }
}

// Producer-consumer pattern with atomics
__global__ void producer_consumer_demo() {
    __shared__ int buffer[32];
    __shared__ int head, tail, count;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        head = 0;
        tail = 0;
        count = 0;
    }
    __syncthreads();
    
    // Producer threads (even thread IDs)
    if (tid % 2 == 0 && tid < 16) {
        int data = tid;
        
        // Wait for space in buffer
        while (atomicAdd(&count, 0) >= 32) {
            __threadfence_block();
        }
        
        // Produce data
        int pos = atomicAdd(&tail, 1) % 32;
        buffer[pos] = data;
        atomicAdd(&count, 1);
        __threadfence_block();
        
        if (tid == 0) printf("Produced: %d\n", data);
    }
    
    __syncthreads();
    
    // Consumer threads (odd thread IDs)  
    if (tid % 2 == 1 && tid < 16) {
        // Wait for data in buffer
        while (atomicAdd(&count, 0) <= 0) {
            __threadfence_block();
        }
        
        // Consume data
        int pos = atomicAdd(&head, 1) % 32;
        int data = buffer[pos];
        atomicAdd(&count, -1);
        __threadfence_block();
        
        if (tid == 1) printf("Consumed: %d\n", data);
    }
}

// Cooperative groups synchronization (requires compute capability 6.0+)
__global__ void coop_groups_sync_demo() {
    // Manual warp synchronization without cooperative groups
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    __shared__ int warp_data[32];
    
    // Initialize data
    if (lane_id == 0) {
        warp_data[warp_id] = warp_id * 100;
    }
    
    // Warp-level synchronization
    __syncwarp();
    
    // All threads in warp can now read the data
    int data = warp_data[warp_id];
    
    if (tid < 4) {
        printf("Thread %d in warp %d read: %d\n", tid, warp_id, data);
    }
}

// Memory fence demonstration
__global__ void memory_fence_demo() {
    __shared__ int flag;
    __shared__ int data;
    
    int tid = threadIdx.x;
    
    if (tid == 0) {
        flag = 0;
        data = 0;
    }
    __syncthreads();
    
    if (tid == 0) {
        // Writer thread
        data = 42;
        __threadfence_block(); // Ensure data write is visible
        flag = 1; // Signal that data is ready
    } else if (tid == 1) {
        // Reader thread
        while (flag == 0) {
            __threadfence_block(); // Ensure we see the latest flag value
        }
        printf("Reader saw data: %d\n", data);
    }
}

float benchmark_histogram(const char *name, void (*kernel_launcher)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        kernel_launcher();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    time /= 10.0f;
    
    printf("%s: %.3f ms\n", name, time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time;
}

int main() {
    printf("=== CUDA Atomic Operations and Synchronization Demo ===\n");
    printf("This example demonstrates various atomic operations and synchronization primitives.\n\n");
    
    // Check device atomic capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory atomic support: %s\n", prop.major >= 2 ? "Yes" : "Limited");
    printf("\n");
    
    // Histogram benchmark
    printf("=== Histogram Performance Comparison ===\n");
    
    size_t data_bytes = N * sizeof(int);
    size_t hist_bytes = NUM_BINS * sizeof(int);
    
    int *h_data = (int*)malloc(data_bytes);
    int *h_histogram = (int*)malloc(hist_bytes);
    
    // Initialize random data
    for (int i = 0; i < N; i++) {
        h_data[i] = rand() % NUM_BINS;
    }
    
    int *d_data, *d_histogram;
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_histogram, hist_bytes);
    
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    
    int grid_size = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = min(grid_size, 128); // Limit grid size for this demo
    
    // Global atomics
    auto global_test = [&]() {
        cudaMemset(d_histogram, 0, hist_bytes);
        histogram_global_atomics<<<grid_size, BLOCK_SIZE>>>(d_data, d_histogram, N);
        cudaDeviceSynchronize();
    };
    float global_time = benchmark_histogram("Global Memory Atomics", global_test);
    
    // Shared memory atomics
    auto shared_test = [&]() {
        cudaMemset(d_histogram, 0, hist_bytes);
        histogram_shared_atomics<<<grid_size, BLOCK_SIZE, NUM_BINS * sizeof(int)>>>(d_data, d_histogram, N);
        cudaDeviceSynchronize();
    };
    float shared_time = benchmark_histogram("Shared Memory Atomics", shared_test);
    
    printf("Shared memory speedup: %.2fx\n", global_time / shared_time);
    
    // Verify histogram correctness
    cudaMemcpy(h_histogram, d_histogram, hist_bytes, cudaMemcpyDeviceToHost);
    int total = 0;
    for (int i = 0; i < NUM_BINS; i++) {
        total += h_histogram[i];
    }
    printf("Histogram verification: %s (total = %d)\n", total == N ? "PASSED" : "FAILED", total);
    
    printf("\n=== Atomic Operations Demo ===\n");
    atomic_operations_demo<<<2, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Lock-Free Stack Demo ===\n");
    int *d_stack, *d_top;
    cudaMalloc(&d_stack, 1024 * sizeof(int));
    cudaMalloc(&d_top, sizeof(int));
    cudaMemset(d_top, 0, sizeof(int));
    
    lock_free_stack_demo<<<4, 64>>>(d_stack, d_top, d_data, 256);
    cudaDeviceSynchronize();
    
    printf("\n=== Producer-Consumer Demo ===\n");
    producer_consumer_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Cooperative Groups Sync Demo ===\n");
    coop_groups_sync_demo<<<1, 64>>>();
    cudaDeviceSynchronize();
    
    printf("\n=== Memory Fence Demo ===\n");
    memory_fence_demo<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\nKey Learnings:\n");
    printf("- Atomic operations enable lock-free programming\n");
    printf("- Shared memory atomics are faster than global memory atomics\n");
    printf("- Use atomic operations sparingly to avoid serialization\n");
    printf("- Memory fences ensure proper ordering of memory operations\n");
    printf("- Warp-level synchronization can be more efficient than block-level\n");
    printf("- Lock-free data structures can improve performance and avoid deadlocks\n");
    printf("- Cooperative groups provide more flexible synchronization patterns\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_histogram);
    cudaFree(d_stack);
    cudaFree(d_top);
    free(h_data);
    free(h_histogram);
    
    return 0;
}