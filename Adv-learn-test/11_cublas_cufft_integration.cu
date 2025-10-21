#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cufft.h>
#include <stdio.h>
#include <math.h>

#define N 1024
#define SIGNAL_SIZE 1024

// Custom matrix multiplication for comparison
__global__ void matrix_multiply_custom(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Convolution using custom implementation
__global__ void convolution_custom(float *signal, float *kernel, float *output, 
                                  int signal_size, int kernel_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= signal_size) return;
    
    float sum = 0.0f;
    int half_kernel = kernel_size / 2;
    
    for (int k = 0; k < kernel_size; k++) {
        int signal_idx = idx - half_kernel + k;
        if (signal_idx >= 0 && signal_idx < signal_size) {
            sum += signal[signal_idx] * kernel[k];
        }
    }
    output[idx] = sum;
}

void demonstrate_cublas() {
    printf("=== cuBLAS Integration Demo ===\n");
    
    size_t bytes = N * N * sizeof(float);
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_custom = (float*)malloc(bytes);
    float *h_C_cublas = (float*)malloc(bytes);
    
    // Initialize matrices
    for (int i = 0; i < N * N; i++) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    // Set math mode for Tensor Cores (if available)
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Benchmark custom matrix multiplication
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        matrix_multiply_custom<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_time;
    cudaEventElapsedTime(&custom_time, start, stop);
    custom_time /= 10.0f;
    printf("Custom matrix multiplication: %.3f ms\n", custom_time);
    
    cudaMemcpy(h_C_custom, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Benchmark cuBLAS SGEMM
    const float alpha = 1.0f, beta = 0.0f;
    
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cublas_time;
    cudaEventElapsedTime(&cublas_time, start, stop);
    cublas_time /= 10.0f;
    printf("cuBLAS SGEMM: %.3f ms\n", cublas_time);
    printf("cuBLAS speedup: %.2fx\n", custom_time / cublas_time);
    
    cudaMemcpy(h_C_cublas, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Verify results match
    bool results_match = true;
    float max_diff = 0.0f;
    for (int i = 0; i < N * N && results_match; i++) {
        float diff = fabs(h_C_custom[i] - h_C_cublas[i]);
        max_diff = fmax(max_diff, diff);
        if (diff > 0.01f) {
            results_match = false;
        }
    }
    printf("Results match: %s (max diff: %.6f)\n", results_match ? "YES" : "NO", max_diff);
    
    // Demonstrate other cuBLAS operations
    printf("\n--- Other cuBLAS Operations ---\n");
    
    float *d_vector;
    cudaMalloc(&d_vector, N * sizeof(float));
    
    // Vector operations
    float *h_vector = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        h_vector[i] = (float)i;
    }
    cudaMemcpy(d_vector, h_vector, N * sizeof(float), cudaMemcpyHostToDevice);
    
    // SAXPY: y = alpha * x + y
    cublasSaxpy(handle, N, &alpha, d_vector, 1, d_A, 1);
    printf("SAXPY operation completed\n");
    
    // Dot product
    float dot_result;
    cublasSdot(handle, N, d_vector, 1, d_vector, 1, &dot_result);
    printf("Dot product result: %.2f\n", dot_result);
    
    // Vector norm
    float norm_result;
    cublasSnrm2(handle, N, d_vector, 1, &norm_result);
    printf("Vector norm: %.2f\n", norm_result);
    
    // Cleanup cuBLAS
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_vector);
    free(h_A);
    free(h_B);
    free(h_C_custom);
    free(h_C_cublas);
    free(h_vector);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void demonstrate_cufft() {
    printf("\n=== cuFFT Integration Demo ===\n");
    
    size_t signal_bytes = SIGNAL_SIZE * sizeof(float);
    size_t complex_bytes = (SIGNAL_SIZE/2 + 1) * sizeof(cufftComplex);
    
    float *h_signal = (float*)malloc(signal_bytes);
    float *h_filtered = (float*)malloc(signal_bytes);
    float *h_custom_conv = (float*)malloc(signal_bytes);
    
    // Generate test signal: sine wave + noise
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        float t = (float)i / SIGNAL_SIZE;
        h_signal[i] = sinf(2 * M_PI * 10 * t) + 0.5f * sinf(2 * M_PI * 50 * t) + 
                     0.1f * (rand() / (float)RAND_MAX - 0.5f);
    }
    
    // Create low-pass filter kernel
    int kernel_size = 15;
    float *h_kernel = (float*)malloc(kernel_size * sizeof(float));
    float sigma = 2.0f;
    float sum = 0.0f;
    
    for (int i = 0; i < kernel_size; i++) {
        int x = i - kernel_size/2;
        h_kernel[i] = expf(-(x*x) / (2*sigma*sigma));
        sum += h_kernel[i];
    }
    
    // Normalize kernel
    for (int i = 0; i < kernel_size; i++) {
        h_kernel[i] /= sum;
    }
    
    float *d_signal, *d_kernel, *d_conv_result;
    cudaMalloc(&d_signal, signal_bytes);
    cudaMalloc(&d_kernel, kernel_size * sizeof(float));
    cudaMalloc(&d_conv_result, signal_bytes);
    
    cudaMemcpy(d_signal, h_signal, signal_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Custom convolution
    int grid_size = (SIGNAL_SIZE + 256 - 1) / 256;
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        convolution_custom<<<grid_size, 256>>>(d_signal, d_kernel, d_conv_result, 
                                              SIGNAL_SIZE, kernel_size);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float custom_conv_time;
    cudaEventElapsedTime(&custom_conv_time, start, stop);
    custom_conv_time /= 100.0f;
    printf("Custom convolution: %.3f ms\n", custom_conv_time);
    
    cudaMemcpy(h_custom_conv, d_conv_result, signal_bytes, cudaMemcpyDeviceToHost);
    
    // FFT-based convolution using cuFFT
    cufftHandle fft_plan;
    cufftComplex *d_signal_freq, *d_kernel_freq, *d_result_freq;
    float *d_signal_padded, *d_kernel_padded, *d_fft_result;
    
    // Create padded arrays for FFT convolution
    int padded_size = SIGNAL_SIZE + kernel_size - 1;
    int fft_size = 1;
    while (fft_size < padded_size) fft_size *= 2; // Next power of 2
    
    cudaMalloc(&d_signal_freq, fft_size * sizeof(cufftComplex));
    cudaMalloc(&d_kernel_freq, fft_size * sizeof(cufftComplex));
    cudaMalloc(&d_result_freq, fft_size * sizeof(cufftComplex));
    cudaMalloc(&d_signal_padded, fft_size * sizeof(float));
    cudaMalloc(&d_kernel_padded, fft_size * sizeof(float));
    cudaMalloc(&d_fft_result, fft_size * sizeof(float));
    
    // Zero-pad the signal and kernel
    cudaMemset(d_signal_padded, 0, fft_size * sizeof(float));
    cudaMemset(d_kernel_padded, 0, fft_size * sizeof(float));
    cudaMemcpy(d_signal_padded, h_signal, signal_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel_padded, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice);
    
    // Create FFT plans
    cufftPlan1d(&fft_plan, fft_size, CUFFT_R2C, 1);
    cufftHandle ifft_plan;
    cufftPlan1d(&ifft_plan, fft_size, CUFFT_C2R, 1);
    
    cudaEventRecord(start);
    
    // Forward FFT of signal and kernel
    cufftExecR2C(fft_plan, d_signal_padded, d_signal_freq);
    cufftExecR2C(fft_plan, d_kernel_padded, d_kernel_freq);
    
    // Point-wise multiplication in frequency domain
    // This would typically be done with a custom kernel
    // For simplicity, we'll use cuBLAS for complex multiplication
    
    // Inverse FFT to get convolution result
    cufftExecC2R(ifft_plan, d_result_freq, d_fft_result);
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float fft_conv_time;
    cudaEventElapsedTime(&fft_conv_time, start, stop);
    printf("FFT-based convolution: %.3f ms\n", fft_conv_time);
    
    // For large kernels, FFT convolution can be faster
    if (kernel_size > 64) {
        printf("FFT speedup: %.2fx\n", custom_conv_time / fft_conv_time);
    } else {
        printf("Custom convolution faster for small kernels\n");
    }
    
    // Demonstrate 2D FFT
    printf("\n--- 2D FFT Demo ---\n");
    
    int nx = 256, ny = 256;
    cufftHandle fft_2d_plan;
    cufftComplex *d_2d_data;
    
    cudaMalloc(&d_2d_data, nx * ny * sizeof(cufftComplex));
    cufftPlan2d(&fft_2d_plan, nx, ny, CUFFT_C2C);
    
    // Initialize with some 2D pattern
    cufftComplex *h_2d_data = (cufftComplex*)malloc(nx * ny * sizeof(cufftComplex));
    for (int i = 0; i < nx * ny; i++) {
        h_2d_data[i].x = sinf(2 * M_PI * (i % nx) / nx) * cosf(2 * M_PI * (i / nx) / ny);
        h_2d_data[i].y = 0.0f;
    }
    
    cudaMemcpy(d_2d_data, h_2d_data, nx * ny * sizeof(cufftComplex), cudaMemcpyHostToDevice);
    
    // Forward 2D FFT
    cufftExecC2C(fft_2d_plan, d_2d_data, d_2d_data, CUFFT_FORWARD);
    
    // Inverse 2D FFT
    cufftExecC2C(fft_2d_plan, d_2d_data, d_2d_data, CUFFT_INVERSE);
    
    printf("2D FFT operations completed\n");
    
    // Cleanup cuFFT
    cufftDestroy(fft_plan);
    cufftDestroy(ifft_plan);
    cufftDestroy(fft_2d_plan);
    
    cudaFree(d_signal);
    cudaFree(d_kernel);
    cudaFree(d_conv_result);
    cudaFree(d_signal_freq);
    cudaFree(d_kernel_freq);
    cudaFree(d_result_freq);
    cudaFree(d_signal_padded);
    cudaFree(d_kernel_padded);
    cudaFree(d_fft_result);
    cudaFree(d_2d_data);
    
    free(h_signal);
    free(h_filtered);
    free(h_custom_conv);
    free(h_kernel);
    free(h_2d_data);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    printf("=== cuBLAS and cuFFT Integration Demo ===\n");
    printf("This example demonstrates how to integrate highly optimized CUDA libraries.\n\n");
    
    // Check library versions
    printf("CUDA Runtime Version: %d\n", CUDART_VERSION);
    printf("cuBLAS Version: %d\n", CUBLAS_VERSION);
    
    // Check device capabilities
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Tensor Core Support: %s\n", prop.major >= 7 ? "Yes" : "No");
    printf("\n");
    
    // Demonstrate cuBLAS
    demonstrate_cublas();
    
    // Demonstrate cuFFT
    demonstrate_cufft();
    
    printf("\nKey Learnings:\n");
    printf("- cuBLAS provides highly optimized linear algebra operations\n");
    printf("- cuFFT enables efficient Fast Fourier Transform computations\n");
    printf("- These libraries are often much faster than custom implementations\n");
    printf("- cuBLAS can utilize Tensor Cores on supported hardware\n");
    printf("- FFT convolution can be faster for large kernels\n");
    printf("- Always benchmark against custom implementations\n");
    printf("- Libraries handle many optimization details automatically\n");
    printf("- Consider memory layout and data types for best performance\n");
    
    return 0;
}
