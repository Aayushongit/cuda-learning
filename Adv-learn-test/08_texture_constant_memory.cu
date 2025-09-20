#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024
#define RADIUS 3

// Constant memory declaration (limited to 64KB)
__constant__ float const_kernel[2*RADIUS+1][2*RADIUS+1];

// Texture object for 2D data
texture<float, 2D, cudaReadModeElementType> tex_2d;

// Global memory convolution (baseline)
__global__ void convolution_global(float *input, float *output, float *kernel, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                int y = row + dy;
                int x = col + dx;
                
                // Handle boundaries
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    sum += input[y * width + x] * kernel[(dy + RADIUS) * (2*RADIUS+1) + (dx + RADIUS)];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Constant memory convolution
__global__ void convolution_constant(float *input, float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                int y = row + dy;
                int x = col + dx;
                
                // Handle boundaries
                if (x >= 0 && x < width && y >= 0 && y < height) {
                    // Access kernel from constant memory
                    sum += input[y * width + x] * const_kernel[dy + RADIUS][dx + RADIUS];
                }
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Texture memory convolution
__global__ void convolution_texture(float *output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                // Texture memory automatically handles boundary conditions
                // and provides hardware interpolation
                float pixel = tex2D(tex_2d, col + dx + 0.5f, row + dy + 0.5f);
                sum += pixel * const_kernel[dy + RADIUS][dx + RADIUS];
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Shared memory + constant memory convolution
__global__ void convolution_shared_constant(float *input, float *output, int width, int height) {
    extern __shared__ float shared_data[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;
    
    int shared_width = blockDim.x + 2 * RADIUS;
    int shared_height = blockDim.y + 2 * RADIUS;
    
    // Load data into shared memory (including halo)
    for (int dy = 0; dy < shared_height; dy += blockDim.y) {
        for (int dx = 0; dx < shared_width; dx += blockDim.x) {
            int shared_x = tx + dx;
            int shared_y = ty + dy;
            
            if (shared_x < shared_width && shared_y < shared_height) {
                int global_x = col + dx - RADIUS;
                int global_y = row + dy - RADIUS;
                
                int shared_idx = shared_y * shared_width + shared_x;
                
                if (global_x >= 0 && global_x < width && global_y >= 0 && global_y < height) {
                    shared_data[shared_idx] = input[global_y * width + global_x];
                } else {
                    shared_data[shared_idx] = 0.0f; // Zero padding
                }
            }
        }
    }
    
    __syncthreads();
    
    // Perform convolution using shared memory
    if (col < width && row < height) {
        float sum = 0.0f;
        
        for (int dy = -RADIUS; dy <= RADIUS; dy++) {
            for (int dx = -RADIUS; dx <= RADIUS; dx++) {
                int shared_x = tx + RADIUS + dx;
                int shared_y = ty + RADIUS + dy;
                int shared_idx = shared_y * shared_width + shared_x;
                
                sum += shared_data[shared_idx] * const_kernel[dy + RADIUS][dx + RADIUS];
            }
        }
        
        output[row * width + col] = sum;
    }
}

// Demonstrate constant memory access patterns
__global__ void constant_memory_demo() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    // All threads in a warp access the same constant memory location
    // This is the optimal access pattern for constant memory
    float value = const_kernel[RADIUS][RADIUS]; // Center of kernel
    
    if (tid == 0) {
        printf("Constant memory center value: %f\n", value);
        printf("Constant memory is cached and optimized for broadcast access\n");
    }
    
    // Divergent access to constant memory (less efficient)
    int offset = tid % (2*RADIUS+1);
    float divergent_value = const_kernel[RADIUS][offset];
    
    if (tid < 7) {
        printf("Thread %d: divergent constant access = %f\n", tid, divergent_value);
    }
}

float benchmark_convolution(const char *name, void (*kernel_launcher)()) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warm up
    kernel_launcher();
    cudaDeviceSynchronize();
    
    cudaEventRecord(start);
    for (int i = 0; i < 50; i++) {
        kernel_launcher();
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float time;
    cudaEventElapsedTime(&time, start, stop);
    time /= 50.0f;
    
    printf("%s: %.3f ms\n", name, time);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return time;
}

int main() {
    printf("=== Texture and Constant Memory Optimization Demo ===\n");
    printf("This example demonstrates the performance benefits of texture and constant memory.\n\n");
    
    size_t bytes = N * N * sizeof(float);
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float *h_kernel = (float*)malloc((2*RADIUS+1) * (2*RADIUS+1) * sizeof(float));
    
    // Initialize input image with random data
    for (int i = 0; i < N * N; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Initialize Gaussian blur kernel
    float sigma = 1.0f;
    float sum = 0.0f;
    for (int y = -RADIUS; y <= RADIUS; y++) {
        for (int x = -RADIUS; x <= RADIUS; x++) {
            float value = expf(-(x*x + y*y) / (2*sigma*sigma));
            h_kernel[(y + RADIUS) * (2*RADIUS+1) + (x + RADIUS)] = value;
            sum += value;
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < (2*RADIUS+1) * (2*RADIUS+1); i++) {
        h_kernel[i] /= sum;
    }
    
    // Copy kernel to constant memory
    cudaMemcpyToSymbol(const_kernel, h_kernel, (2*RADIUS+1) * (2*RADIUS+1) * sizeof(float));
    
    float *d_input, *d_output, *d_kernel;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_kernel, (2*RADIUS+1) * (2*RADIUS+1) * sizeof(float));
    
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, (2*RADIUS+1) * (2*RADIUS+1) * sizeof(float), cudaMemcpyHostToDevice);
    
    // Setup texture memory
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray *cuda_array;
    cudaMallocArray(&cuda_array, &channelDesc, N, N);
    cudaMemcpyToArray(cuda_array, 0, 0, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Bind texture
    tex_2d.addressMode[0] = cudaAddressModeClamp;
    tex_2d.addressMode[1] = cudaAddressModeClamp;
    tex_2d.filterMode = cudaFilterModeLinear;
    tex_2d.normalized = false;
    cudaBindTextureToArray(tex_2d, cuda_array, channelDesc);
    
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    printf("=== Performance Comparison ===\n");
    
    // Global memory convolution
    auto global_test = [&]() {
        convolution_global<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, N, N);
    };
    float global_time = benchmark_convolution("Global Memory Convolution", global_test);
    
    // Constant memory convolution
    auto constant_test = [&]() {
        convolution_constant<<<gridDim, blockDim>>>(d_input, d_output, N, N);
    };
    float constant_time = benchmark_convolution("Constant Memory Convolution", constant_test);
    
    // Texture memory convolution
    auto texture_test = [&]() {
        convolution_texture<<<gridDim, blockDim>>>(d_output, N, N);
    };
    float texture_time = benchmark_convolution("Texture Memory Convolution", texture_test);
    
    // Shared + constant memory convolution
    int shared_size = (blockDim.x + 2*RADIUS) * (blockDim.y + 2*RADIUS) * sizeof(float);
    auto shared_test = [&]() {
        convolution_shared_constant<<<gridDim, blockDim, shared_size>>>(d_input, d_output, N, N);
    };
    float shared_time = benchmark_convolution("Shared + Constant Memory", shared_test);
    
    printf("\nSpeedup Analysis:\n");
    printf("Constant vs Global: %.2fx\n", global_time / constant_time);
    printf("Texture vs Global: %.2fx\n", global_time / texture_time);
    printf("Shared+Constant vs Global: %.2fx\n", global_time / shared_time);
    
    printf("\n=== Constant Memory Access Pattern Demo ===\n");
    constant_memory_demo<<<2, 32>>>();
    cudaDeviceSynchronize();
    
    printf("\nKey Learnings:\n");
    printf("- Constant memory is cached and optimized for broadcast reads\n");
    printf("- Texture memory provides hardware interpolation and boundary handling\n");
    printf("- Texture cache is optimized for 2D spatial locality\n");
    printf("- Constant memory is best when all threads read the same data\n");
    printf("- Combining shared memory with constant memory often gives best performance\n");
    printf("- Texture memory handles boundary conditions automatically\n");
    printf("- Both constant and texture memory have limited size (64KB for constant)\n");
    
    // Cleanup
    cudaUnbindTexture(tex_2d);
    cudaFreeArray(cuda_array);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_input);
    free(h_output);
    free(h_kernel);
    
    return 0;
}