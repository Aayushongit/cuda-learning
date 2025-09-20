#include <cuda_runtime.h>
#include <stdio.h>

#define WARP_SIZE 32
#define N 1024

__global__ void warp_shuffle_demo() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    float value = (float)(lane_id + 1); // Values 1-32 within each warp
    
    printf("Warp %d, Lane %d: Initial value = %.0f\n", warp_id, lane_id, value);
    
    // Warp shuffle down - each thread gets value from thread (lane + offset)
    float shuffled_down = __shfl_down_sync(0xFFFFFFFF, value, 4);
    if (lane_id == 0) {
        printf("Warp %d: Shuffle down by 4: %.0f\n", warp_id, shuffled_down);
    }
    
    // Warp shuffle up - each thread gets value from thread (lane - offset)  
    float shuffled_up = __shfl_up_sync(0xFFFFFFFF, value, 4);
    if (lane_id == 4) {
        printf("Warp %d: Shuffle up by 4: %.0f\n", warp_id, shuffled_up);
    }
    
    // Warp shuffle xor - each thread gets value from thread (lane ^ offset)
    float shuffled_xor = __shfl_xor_sync(0xFFFFFFFF, value, 1);
    if (lane_id < 2) {
        printf("Warp %d, Lane %d: Shuffle XOR by 1: %.0f\n", warp_id, lane_id, shuffled_xor);
    }
    
    // Broadcast from lane 0 to all threads in warp
    float broadcast = __shfl_sync(0xFFFFFFFF, value, 0);
    if (lane_id == 31) {
        printf("Warp %d: Broadcast from lane 0: %.0f\n", warp_id, broadcast);
    }
}

__global__ void warp_vote_demo() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Create a predicate - true for even lane IDs
    bool predicate = (lane_id % 2 == 0);
    
    // Vote functions
    bool all_true = __all_sync(0xFFFFFFFF, predicate);
    bool any_true = __any_sync(0xFFFFFFFF, predicate);
    unsigned int ballot = __ballot_sync(0xFFFFFFFF, predicate);
    
    if (lane_id == 0) {
        printf("Warp %d Vote Results:\n", warp_id);
        printf("  All threads have even lane ID: %s\n", all_true ? "true" : "false");
        printf("  Any thread has even lane ID: %s\n", any_true ? "true" : "false");
        printf("  Ballot (even lanes): 0x%08x\n", ballot);
        printf("  Population count: %d\n", __popc(ballot));
    }
}

__global__ void warp_scan_demo() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    int value = lane_id + 1; // Values 1-32
    
    // Inclusive prefix sum using shuffle
    int sum = value;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1) {
        int temp = __shfl_up_sync(0xFFFFFFFF, sum, offset);
        if (lane_id >= offset) {
            sum += temp;
        }
    }
    
    if (lane_id < 8) { // Print first 8 results
        printf("Warp %d, Lane %d: Value=%d, Prefix Sum=%d\n", 
               warp_id, lane_id, value, sum);
    }
}

__global__ void warp_matrix_transpose() {
    __shared__ float tile[WARP_SIZE][WARP_SIZE + 1]; // +1 to avoid bank conflicts
    
    int tid = threadIdx.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (warp_id < 1) { // Only first warp
        // Initialize with lane ID
        float value = (float)lane_id;
        
        // Store in row-major order
        tile[lane_id][0] = value;
        tile[lane_id][1] = value + 32;
        
        __syncwarp();
        
        // Read in column-major order (transpose)
        float transposed1 = tile[0][lane_id];
        float transposed2 = tile[1][lane_id];
        
        if (lane_id < 4) {
            printf("Lane %d: Original=(%.0f, %.0f), Transposed=(%.0f, %.0f)\n",
                   lane_id, value, value + 32, transposed1, transposed2);
        }
    }
}

__global__ void warp_cooperative_groups_demo() {
    #if __CUDA_ARCH__ >= 700  // Cooperative groups requires compute capability 7.0+
    // This would use cooperative_groups.h
    printf("Cooperative groups demo requires compute capability 7.0+\n");
    #endif
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    // Simulate sub-warp group operations
    if (lane_id < 16) { // First half-warp
        float value = (float)(lane_id + 1);
        
        // Manual half-warp reduction
        for (int offset = 8; offset >= 1; offset /= 2) {
            value += __shfl_down_sync(0x0000FFFF, value, offset); // Half-warp mask
        }
        
        if (lane_id == 0) {
            printf("Warp %d, Half-warp 0 sum: %.0f\n", warp_id, value);
        }
    }
}

int main() {
    printf("=== Warp-Level Primitives Demo ===\n");
    printf("This example demonstrates advanced warp-level operations in CUDA.\n\n");
    
    // Launch with single block to keep output manageable
    dim3 blockDim(64);  // 2 warps
    dim3 gridDim(1);
    
    printf("1. Warp Shuffle Operations:\n");
    warp_shuffle_demo<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    printf("\n2. Warp Vote Operations:\n");
    warp_vote_demo<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    printf("\n3. Warp-Level Scan (Prefix Sum):\n");
    warp_scan_demo<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    printf("\n4. Warp Matrix Transpose:\n");
    warp_matrix_transpose<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    printf("\n5. Sub-Warp Groups:\n");
    warp_cooperative_groups_demo<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
    
    printf("\nKey Learnings:\n");
    printf("- Warp primitives eliminate need for shared memory in many cases\n");
    printf("- __syncwarp() synchronizes threads within a warp\n");
    printf("- Shuffle operations enable efficient data exchange within warps\n");
    printf("- Vote operations provide collective decision making\n");
    printf("- Warp-level algorithms can be much faster than block-level ones\n");
    printf("- Always use appropriate sync masks for partial warp operations\n");
    
    return 0;
}