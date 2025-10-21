// cuFFT 2D: Image frequency analysis and filtering
// Used in computer vision, image compression, and convolution operations

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUFFT(call) { \
    cufftResult err = call; \
    if (err != CUFFT_SUCCESS) { \
        fprintf(stderr, "cuFFT Error: %d (line %d)\n", err, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

#define PI 3.14159265358979323846

// Generate synthetic 2D image with patterns
void generateImage(cufftReal* image, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float fx = (float)x / width;
            float fy = (float)y / height;

            // Create pattern with multiple frequency components
            float value = sin(10.0f * PI * fx) * cos(10.0f * PI * fy) +
                         0.5f * sin(20.0f * PI * fx) +
                         0.3f * cos(15.0f * PI * fy);

            image[y * width + x] = value;
        }
    }
}

// Apply low-pass filter in frequency domain
__global__ void applyLowPassFilter(cufftComplex* spectrum, int width, int height, float cutoff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int centerX = width / 2;
        int centerY = height / 2;

        // Shift coordinates to center
        int shiftX = (x < centerX) ? x : x - width;
        int shiftY = (y < centerY) ? y : y - height;

        float distance = sqrt((float)(shiftX * shiftX + shiftY * shiftY));

        // Apply Gaussian filter
        if (distance > cutoff) {
            int idx = y * width + x;
            float attenuation = exp(-(distance - cutoff) * (distance - cutoff) / (cutoff * cutoff));
            spectrum[idx].x *= attenuation;
            spectrum[idx].y *= attenuation;
        }
    }
}

void printImageStats(const char* name, cufftReal* image, int width, int height) {
    float min_val = image[0], max_val = image[0], sum = 0.0f;

    for (int i = 0; i < width * height; i++) {
        if (image[i] < min_val) min_val = image[i];
        if (image[i] > max_val) max_val = image[i];
        sum += image[i];
    }

    float mean = sum / (width * height);

    printf("%s stats:\n", name);
    printf("  Min: %.4f, Max: %.4f, Mean: %.4f\n", min_val, max_val, mean);
}

int main() {
    const int width = 512;
    const int height = 512;
    const int N = width * height;

    size_t bytes_image = N * sizeof(cufftReal);
    size_t bytes_spectrum = N * sizeof(cufftComplex);

    // Host memory
    cufftReal *h_image = (cufftReal*)malloc(bytes_image);
    cufftReal *h_filtered = (cufftReal*)malloc(bytes_image);
    cufftComplex *h_spectrum = (cufftComplex*)malloc(bytes_spectrum);

    // Generate test image
    generateImage(h_image, width, height);

    // Device memory
    cufftReal *d_image;
    cufftComplex *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_image, bytes_image));
    CHECK_CUDA(cudaMalloc(&d_spectrum, bytes_spectrum));

    CHECK_CUDA(cudaMemcpy(d_image, h_image, bytes_image, cudaMemcpyHostToDevice));

    // Create cuFFT plans for 2D transforms
    cufftHandle plan_forward, plan_inverse;
    CHECK_CUFFT(cufftPlan2d(&plan_forward, height, width, CUFFT_R2C));
    CHECK_CUFFT(cufftPlan2d(&plan_inverse, height, width, CUFFT_C2R));

    // Timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Forward 2D FFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecR2C(plan_forward, d_image, d_spectrum));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_forward = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_forward, start, stop));

    // Apply frequency domain filter
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    float cutoff_freq = 50.0f;  // Low-pass filter cutoff
    applyLowPassFilter<<<gridSize, blockSize>>>(d_spectrum, width, height, cutoff_freq);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy spectrum for analysis
    CHECK_CUDA(cudaMemcpy(h_spectrum, d_spectrum, bytes_spectrum, cudaMemcpyDeviceToHost));

    // Inverse 2D FFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecC2R(plan_inverse, d_spectrum, d_image));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_inverse = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_inverse, start, stop));

    // Copy filtered image
    CHECK_CUDA(cudaMemcpy(h_filtered, d_image, bytes_image, cudaMemcpyDeviceToHost));

    // Normalize
    for (int i = 0; i < N; i++) {
        h_filtered[i] /= N;
    }

    // Results
    printf("=== 2D FFT Image Processing ===\n");
    printf("Image dimensions: %dx%d\n", width, height);
    printf("Forward FFT time: %.3f ms\n", ms_forward);
    printf("Inverse FFT time: %.3f ms\n\n", ms_inverse);

    printImageStats("Original image", h_image, width, height);
    printImageStats("Filtered image", h_filtered, width, height);

    // Calculate spectrum magnitude statistics
    float max_magnitude = 0.0f;
    int max_x = 0, max_y = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            float mag = sqrt(h_spectrum[idx].x * h_spectrum[idx].x +
                           h_spectrum[idx].y * h_spectrum[idx].y);

            if (mag > max_magnitude && x > 5 && y > 5) {  // Ignore DC component
                max_magnitude = mag;
                max_x = x;
                max_y = y;
            }
        }
    }

    printf("\nFrequency domain:\n");
    printf("  Max magnitude: %.2f at position (%d, %d)\n", max_magnitude, max_x, max_y);
    printf("  Filter cutoff: %.1f\n", cutoff_freq);

    // Sample comparison
    printf("\nSample values (corner 4x4):\n");
    printf("Original:\n");
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            printf("%6.2f ", h_image[y * width + x]);
        }
        printf("\n");
    }

    printf("\nFiltered:\n");
    for (int y = 0; y < 4; y++) {
        for (int x = 0; x < 4; x++) {
            printf("%6.2f ", h_filtered[y * width + x]);
        }
        printf("\n");
    }

    // Cleanup
    CHECK_CUFFT(cufftDestroy(plan_forward));
    CHECK_CUFFT(cufftDestroy(plan_inverse));
    CHECK_CUDA(cudaFree(d_image));
    CHECK_CUDA(cudaFree(d_spectrum));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_image);
    free(h_filtered);
    free(h_spectrum);

    printf("\ncuFFT 2D operations completed successfully!\n");
    return 0;
}
