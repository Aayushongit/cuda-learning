#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernel_work(float *data, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 1000; i++) {
            data[idx] = data[idx] * multiplier + 0.001f;
        }
    }
}

int main() {
    const int n = 1024 * 1024;
    const int num_streams = 4;
    const int stream_size = n / num_streams;
    const int stream_bytes = stream_size * sizeof(float);
    
    float *h_data;
    cudaMallocHost(&h_data, n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        h_data[i] = i % 1000 + 1.0f;
    }
    
    float *d_data;
    cudaMalloc(&d_data, n * sizeof(float));
    
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int blocksPerStream = (stream_size + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("CUDA Streams example:\n");
    printf("Processing %d elements using %d streams\n", n, num_streams);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < num_streams; i++) {
        int offset = i * stream_size;
        
        cudaMemcpyAsync(d_data + offset, h_data + offset, stream_bytes, 
                        cudaMemcpyHostToDevice, streams[i]);
        
        kernel_work<<<blocksPerStream, threadsPerBlock, 0, streams[i]>>>(
            d_data + offset, stream_size, 1.01f);
        
        cudaMemcpyAsync(h_data + offset, d_data + offset, stream_bytes, 
                        cudaMemcpyDeviceToHost, streams[i]);
    }
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);
    
    printf("Processing completed in %.3f ms\n", elapsed_time);
    printf("First 10 processed values:\n");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", h_data[i]);
    }
    printf("\n");
    
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFreeHost(h_data);
    
    return 0;
}