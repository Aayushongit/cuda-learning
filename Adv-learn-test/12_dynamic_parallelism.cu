#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define THRESHOLD 64

// Recursive parallel reduction using dynamic parallelism
__global__ void parallel_reduction_recursive(float *data, int size, float *result) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (idx < size) {
        sdata[tid] = data[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (idx + s) < size) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // First thread of each block writes partial result
    if (tid == 0) {
        data[blockIdx.x] = sdata[0];
    }
    
    // If we have more than one block, recursively launch another kernel
    if (tid == 0 && blockIdx.x == 0) {
        int num_blocks = (size + blockDim.x - 1) / blockDim.x;
        if (num_blocks > 1) {
            // Launch child kernel to reduce partial results
            int child_blocks = (num_blocks + blockDim.x - 1) / blockDim.x;
            parallel_reduction_recursive<<<child_blocks, blockDim.x, blockDim.x * sizeof(float)>>>(
                data, num_blocks, result);
        } else {
            // Base case - store final result
            *result = data[0];
        }
    }
}

// Adaptive quicksort using dynamic parallelism
__global__ void quicksort_dynamic(int *data, int left, int right, int depth) {
    if (left >= right) return;
    
    // Use simple bubble sort for small arrays
    if (right - left + 1 <= THRESHOLD) {
        for (int i = left; i <= right; i++) {
            for (int j = left; j < right; j++) {
                if (data[j] > data[j + 1]) {
                    int temp = data[j];
                    data[j] = data[j + 1];
                    data[j + 1] = temp;
                }
            }
        }
        return;
    }
    
    // Partition
    int pivot = data[right];
    int i = left - 1;
    
    for (int j = left; j < right; j++) {
        if (data[j] <= pivot) {
            i++;
            int temp = data[i];
            data[i] = data[j];
            data[j] = temp;
        }
    }
    
    i++;
    int temp = data[i];
    data[i] = data[right];
    data[right] = temp;
    
    int partition_pos = i;
    
    // Launch child kernels for recursive sorting if depth allows
    if (depth > 0) {
        // Sort left partition
        if (partition_pos - 1 > left) {
            quicksort_dynamic<<<1, 1>>>(data, left, partition_pos - 1, depth - 1);
        }
        
        // Sort right partition
        if (partition_pos + 1 < right) {
            quicksort_dynamic<<<1, 1>>>(data, partition_pos + 1, right, depth - 1);
        }
        
        // Wait for child kernels to complete
        cudaDeviceSynchronize();
    } else {
        // Sequential sort for remaining parts
        if (partition_pos - 1 > left) {
            quicksort_dynamic<<<1, 1>>>(data, left, partition_pos - 1, 0);
        }
        if (partition_pos + 1 < right) {
            quicksort_dynamic<<<1, 1>>>(data, partition_pos + 1, right, 0);
        }
    }
}

// Adaptive parallel tree traversal
struct TreeNode {
    int value;
    int left_child;
    int right_child;
};

__global__ void tree_search_dynamic(TreeNode *tree, int *results, int target, 
                                   int node_index, int max_depth) {
    if (node_index == -1) return;
    
    TreeNode node = tree[node_index];
    
    // Check if current node matches target
    if (node.value == target) {
        results[node_index] = 1;
    }
    
    // Launch child kernels if we haven't reached max depth
    if (max_depth > 0) {
        if (node.left_child != -1) {
            tree_search_dynamic<<<1, 1>>>(tree, results, target, node.left_child, max_depth - 1);
        }
        
        if (node.right_child != -1) {
            tree_search_dynamic<<<1, 1>>>(tree, results, target, node.right_child, max_depth - 1);
        }
        
        cudaDeviceSynchronize();
    }
}

// Matrix multiplication with dynamic block sizing
__global__ void matmul_adaptive(float *A, float *B, float *C, int n, int block_start_x, int block_start_y) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int global_x = block_start_x + tid_x;
    int global_y = block_start_y + tid_y;
    
    if (global_x >= n || global_y >= n) return;
    
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += A[global_y * n + k] * B[k * n + global_x];
    }
    C[global_y * n + global_x] = sum;
    
    // If this is the first thread and the block size is large enough,
    // subdivide the work using child kernels
    if (tid_x == 0 && tid_y == 0 && blockDim.x > 16 && blockDim.y > 16) {
        int new_block_size = min(blockDim.x / 2, blockDim.y / 2);
        dim3 child_block(new_block_size, new_block_size);
        
        // Launch 4 child kernels for quadrants
        matmul_adaptive<<<1, child_block>>>(A, B, C, n, block_start_x, block_start_y);
        matmul_adaptive<<<1, child_block>>>(A, B, C, n, block_start_x + new_block_size, block_start_y);
        matmul_adaptive<<<1, child_block>>>(A, B, C, n, block_start_x, block_start_y + new_block_size);
        matmul_adaptive<<<1, child_block>>>(A, B, C, n, block_start_x + new_block_size, block_start_y + new_block_size);
        
        cudaDeviceSynchronize();
    }
}

// Mandelbrot set with adaptive subdivision
__global__ void mandelbrot_adaptive(float *output, float x_min, float x_max, 
                                   float y_min, float y_max, int width, int height,
                                   int start_x, int start_y, int region_width, int region_height,
                                   int max_depth) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    
    if (tid_x >= region_width || tid_y >= region_height) return;
    
    int pixel_x = start_x + tid_x;
    int pixel_y = start_y + tid_y;
    
    if (pixel_x >= width || pixel_y >= height) return;
    
    // Calculate Mandelbrot value for this pixel
    float dx = (x_max - x_min) / width;
    float dy = (y_max - y_min) / height;
    
    float cx = x_min + pixel_x * dx;
    float cy = y_min + pixel_y * dy;
    
    float zx = 0.0f, zy = 0.0f;
    int iterations = 0;
    
    while (zx*zx + zy*zy < 4.0f && iterations < 1000) {
        float temp = zx*zx - zy*zy + cx;
        zy = 2*zx*zy + cy;
        zx = temp;
        iterations++;
    }
    
    output[pixel_y * width + pixel_x] = (float)iterations / 1000.0f;
    
    // Adaptive subdivision: if region is large and we have depth left, subdivide
    if (tid_x == 0 && tid_y == 0 && max_depth > 0 && 
        region_width > 32 && region_height > 32) {
        
        int sub_width = region_width / 2;
        int sub_height = region_height / 2;
        
        dim3 child_block(min(sub_width, 16), min(sub_height, 16));
        
        // Launch 4 child kernels for quadrants
        mandelbrot_adaptive<<<1, child_block>>>(output, x_min, x_max, y_min, y_max, 
                                               width, height, start_x, start_y, 
                                               sub_width, sub_height, max_depth - 1);
        mandelbrot_adaptive<<<1, child_block>>>(output, x_min, x_max, y_min, y_max, 
                                               width, height, start_x + sub_width, start_y, 
                                               sub_width, sub_height, max_depth - 1);
        mandelbrot_adaptive<<<1, child_block>>>(output, x_min, x_max, y_min, y_max, 
                                               width, height, start_x, start_y + sub_height, 
                                               sub_width, sub_height, max_depth - 1);
        mandelbrot_adaptive<<<1, child_block>>>(output, x_min, x_max, y_min, y_max, 
                                               width, height, start_x + sub_width, start_y + sub_height, 
                                               sub_width, sub_height, max_depth - 1);
        
        cudaDeviceSynchronize();
    }
}

int main() {
    printf("=== CUDA Dynamic Parallelism Demo ===\n");
    printf("This example demonstrates kernels launching other kernels on GPU.\n\n");
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    
    if (prop.major < 3 || (prop.major == 3 && prop.minor < 5)) {
        printf("Dynamic Parallelism requires compute capability 3.5 or higher\n");
        printf("This device does not support dynamic parallelism\n");
        return 1;
    }
    
    printf("Dynamic Parallelism: Supported\n\n");
    
    // 1. Parallel Reduction Demo
    printf("=== Recursive Parallel Reduction ===\n");
    
    size_t bytes = N * sizeof(float);
    float *h_data = (float*)malloc(bytes);
    float *d_data, *d_result;
    
    // Initialize with ones for easy verification
    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }
    
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_result, sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);
    
    // Launch recursive reduction
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    
    parallel_reduction_recursive<<<grid_size, block_size, block_size * sizeof(float)>>>(
        d_data, N, d_result);
    cudaDeviceSynchronize();
    
    float result;
    cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Reduction result: %.0f (expected: %d)\n", result, N);
    
    // 2. Dynamic Quicksort Demo
    printf("\n=== Adaptive Quicksort ===\n");
    
    int sort_size = 512;
    int *h_sort_data = (int*)malloc(sort_size * sizeof(int));
    int *d_sort_data;
    
    // Initialize with random data
    for (int i = 0; i < sort_size; i++) {
        h_sort_data[i] = rand() % 1000;
    }
    
    cudaMalloc(&d_sort_data, sort_size * sizeof(int));
    cudaMemcpy(d_sort_data, h_sort_data, sort_size * sizeof(int), cudaMemcpyHostToDevice);
    
    printf("Original: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_sort_data[i]);
    }
    printf("...\n");
    
    // Launch dynamic quicksort
    quicksort_dynamic<<<1, 1>>>(d_sort_data, 0, sort_size - 1, 3);
    cudaDeviceSynchronize();
    
    int *h_sorted = (int*)malloc(sort_size * sizeof(int));
    cudaMemcpy(h_sorted, d_sort_data, sort_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Sorted:   ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_sorted[i]);
    }
    printf("...\n");
    
    // Verify sorting
    bool is_sorted = true;
    for (int i = 1; i < sort_size; i++) {
        if (h_sorted[i] < h_sorted[i-1]) {
            is_sorted = false;
            break;
        }
    }
    printf("Sort verification: %s\n", is_sorted ? "PASSED" : "FAILED");
    
    // 3. Tree Search Demo
    printf("\n=== Dynamic Tree Search ===\n");
    
    int tree_size = 15;
    TreeNode *h_tree = (TreeNode*)malloc(tree_size * sizeof(TreeNode));
    TreeNode *d_tree;
    int *d_search_results;
    
    // Build a simple binary tree
    for (int i = 0; i < tree_size; i++) {
        h_tree[i].value = i + 1;
        h_tree[i].left_child = (2*i + 1 < tree_size) ? 2*i + 1 : -1;
        h_tree[i].right_child = (2*i + 2 < tree_size) ? 2*i + 2 : -1;
    }
    
    cudaMalloc(&d_tree, tree_size * sizeof(TreeNode));
    cudaMalloc(&d_search_results, tree_size * sizeof(int));
    cudaMemcpy(d_tree, h_tree, tree_size * sizeof(TreeNode), cudaMemcpyHostToDevice);
    cudaMemset(d_search_results, 0, tree_size * sizeof(int));
    
    int search_target = 7;
    tree_search_dynamic<<<1, 1>>>(d_tree, d_search_results, search_target, 0, 4);
    cudaDeviceSynchronize();
    
    int *h_search_results = (int*)malloc(tree_size * sizeof(int));
    cudaMemcpy(h_search_results, d_search_results, tree_size * sizeof(int), cudaMemcpyDeviceToHost);
    
    printf("Searching for value %d in tree:\n", search_target);
    for (int i = 0; i < tree_size; i++) {
        if (h_search_results[i]) {
            printf("Found at node %d (value %d)\n", i, h_tree[i].value);
        }
    }
    
    // 4. Mandelbrot with adaptive subdivision
    printf("\n=== Adaptive Mandelbrot Set ===\n");
    
    int mandel_width = 512, mandel_height = 512;
    float *d_mandelbrot;
    cudaMalloc(&d_mandelbrot, mandel_width * mandel_height * sizeof(float));
    
    dim3 initial_block(32, 32);
    mandelbrot_adaptive<<<1, initial_block>>>(d_mandelbrot, -2.0f, 1.0f, -1.5f, 1.5f,
                                             mandel_width, mandel_height, 0, 0,
                                             mandel_width, mandel_height, 2);
    cudaDeviceSynchronize();
    
    printf("Mandelbrot set computed with adaptive subdivision\n");
    
    printf("\nKey Learnings:\n");
    printf("- Dynamic parallelism allows kernels to launch child kernels\n");
    printf("- Useful for irregular or adaptive algorithms\n");
    printf("- Requires compute capability 3.5 or higher\n");
    printf("- Child kernel launches have overhead - use judiciously\n");
    printf("- Excellent for divide-and-conquer algorithms\n");
    printf("- Enables more natural expression of recursive algorithms\n");
    printf("- Can improve load balancing for irregular workloads\n");
    printf("- Consider depth limits to prevent excessive recursion\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_result);
    cudaFree(d_sort_data);
    cudaFree(d_tree);
    cudaFree(d_search_results);
    cudaFree(d_mandelbrot);
    free(h_data);
    free(h_sort_data);
    free(h_sorted);
    free(h_tree);
    free(h_search_results);
    
    return 0;
}