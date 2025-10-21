// NPP Image Processing: GPU-accelerated image filtering and transformations
// Optimized for computer vision preprocessing pipelines

#include <npp.h>
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

#define CHECK_NPP(call) { \
    NppStatus status = call; \
    if (status != NPP_SUCCESS) { \
        fprintf(stderr, "NPP Error: %d (line %d)\n", status, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

void generateTestImage(Npp8u* image, int width, int height, int step) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int value = ((x / 10) + (y / 10)) % 2 ? 255 : 50;
            value += (rand() % 40) - 20;  // Add noise
            if (value < 0) value = 0;
            if (value > 255) value = 255;
            image[y * step + x] = (Npp8u)value;
        }
    }
}

void printImageStats(const char* name, Npp8u* image, int width, int height, int step) {
    int sum = 0;
    int min_val = 255, max_val = 0;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int val = image[y * step + x];
            sum += val;
            if (val < min_val) min_val = val;
            if (val > max_val) max_val = val;
        }
    }

    printf("%s - Min: %d, Max: %d, Mean: %.1f\n",
           name, min_val, max_val, (float)sum / (width * height));
}

int main() {
    const int width = 1920;
    const int height = 1080;

    printf("=== NPP Image Processing ===\n");
    printf("Image size: %dx%d\n\n", width, height);

    NppiSize roi_size = {width, height};
    NppiPoint roi_offset = {0, 0};

    // Allocate host memory with pitch
    int host_step = width * sizeof(Npp8u);
    Npp8u *h_src = (Npp8u*)malloc(height * host_step);
    Npp8u *h_dst = (Npp8u*)malloc(height * host_step);

    // Generate test image
    generateTestImage(h_src, width, height, width);

    // Allocate device memory with pitch
    Npp8u *d_src, *d_dst, *d_temp;
    int device_step;

    CHECK_CUDA(cudaMallocPitch(&d_src, (size_t*)&device_step, width * sizeof(Npp8u), height));
    CHECK_CUDA(cudaMallocPitch(&d_dst, (size_t*)&device_step, width * sizeof(Npp8u), height));
    CHECK_CUDA(cudaMallocPitch(&d_temp, (size_t*)&device_step, width * sizeof(Npp8u), height));

    CHECK_CUDA(cudaMemcpy2D(d_src, device_step, h_src, host_step,
                           width * sizeof(Npp8u), height, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. GAUSSIAN BLUR (3x3 kernel)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiFilterGauss_8u_C1R(d_src, device_step, d_dst, device_step,
                                     roi_size, NPP_MASK_SIZE_3_X_3));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_gauss = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_gauss, start, stop));

    // 2. BOX FILTER (smoothing)
    NppiSize mask_size = {5, 5};
    NppiPoint anchor = {2, 2};

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiFilterBox_8u_C1R(d_src, device_step, d_dst, device_step,
                                   roi_size, mask_size, anchor));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_box = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_box, start, stop));

    // 3. SOBEL FILTER (edge detection)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiFilterSobel_8u_C1R(d_src, device_step, d_dst, device_step,
                                     roi_size, NPP_MASK_SIZE_3_X_3));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_sobel = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sobel, start, stop));

    // 4. MEDIAN FILTER (noise reduction)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiFilterMedian_8u_C1R(d_src, device_step, d_dst, device_step,
                                      roi_size, mask_size, anchor));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_median = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_median, start, stop));

    // 5. THRESHOLD (binary threshold at 128)
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiThreshold_GTVal_8u_C1R(d_src, device_step, d_dst, device_step,
                                         roi_size, 128, 255));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_threshold = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_threshold, start, stop));

    // 6. HISTOGRAM EQUALIZATION
    int levels[257];  // 256 bins + 1
    for (int i = 0; i <= 256; i++) levels[i] = i;

    int* d_histogram;
    CHECK_CUDA(cudaMalloc(&d_histogram, 256 * sizeof(int)));

    int buffer_size;
    CHECK_NPP(nppiHistogramEvenGetBufferSize_8u_C1R(roi_size, 256, &buffer_size));

    Npp8u* d_buffer;
    CHECK_CUDA(cudaMalloc(&d_buffer, buffer_size));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiHistogramEven_8u_C1R(d_src, device_step, roi_size,
                                       d_histogram, 256, 0, 256, d_buffer));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_histogram = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_histogram, start, stop));

    // 7. RESIZE (downscale by 2x)
    NppiSize dst_size = {width / 2, height / 2};
    NppiRect src_rect = {0, 0, width, height};
    NppiRect dst_rect = {0, 0, width / 2, height / 2};

    Npp8u *d_resized;
    int resized_step;
    CHECK_CUDA(cudaMallocPitch(&d_resized, (size_t*)&resized_step,
                              (width / 2) * sizeof(Npp8u), height / 2));

    CHECK_CUDA(cudaEventRecord(start));
    CHECK_NPP(nppiResize_8u_C1R(d_src, device_step, roi_size, src_rect,
                                d_resized, resized_step, dst_size, dst_rect,
                                NPPI_INTER_LINEAR));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_resize = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_resize, start, stop));

    // Copy results for verification
    CHECK_CUDA(cudaMemcpy2D(h_dst, host_step, d_dst, device_step,
                           width * sizeof(Npp8u), height, cudaMemcpyDeviceToHost));

    // Performance metrics
    size_t pixels = width * height;

    printf("=== Performance ===\n");
    printf("Gaussian Blur:    %.3f ms (%.2f B pixels/sec)\n",
           ms_gauss, pixels / (ms_gauss * 1e6));
    printf("Box Filter:       %.3f ms (%.2f B pixels/sec)\n",
           ms_box, pixels / (ms_box * 1e6));
    printf("Sobel Filter:     %.3f ms (%.2f B pixels/sec)\n",
           ms_sobel, pixels / (ms_sobel * 1e6));
    printf("Median Filter:    %.3f ms (%.2f B pixels/sec)\n",
           ms_median, pixels / (ms_median * 1e6));
    printf("Threshold:        %.3f ms (%.2f B pixels/sec)\n",
           ms_threshold, pixels / (ms_threshold * 1e6));
    printf("Histogram:        %.3f ms (%.2f B pixels/sec)\n",
           ms_histogram, pixels / (ms_histogram * 1e6));
    printf("Resize (2x down): %.3f ms (%.2f B pixels/sec)\n",
           ms_resize, (pixels / 4) / (ms_resize * 1e6));

    printf("\n=== Image Statistics ===\n");
    printImageStats("Original", h_src, width, height, width);
    printImageStats("Processed", h_dst, width, height, width);

    // Cleanup
    CHECK_CUDA(cudaFree(d_src));
    CHECK_CUDA(cudaFree(d_dst));
    CHECK_CUDA(cudaFree(d_temp));
    CHECK_CUDA(cudaFree(d_resized));
    CHECK_CUDA(cudaFree(d_histogram));
    CHECK_CUDA(cudaFree(d_buffer));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_src);
    free(h_dst);

    printf("\nNPP image processing completed successfully!\n");
    return 0;
}
