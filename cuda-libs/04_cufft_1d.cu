// cuFFT 1D: Fast Fourier Transform for signal processing and audio analysis
// Essential for feature extraction in speech recognition and time-series analysis

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

// Generate composite signal: sum of sine waves
void generateSignal(cufftReal* signal, int N) {
    float freq1 = 5.0f;    // 5 Hz component
    float freq2 = 15.0f;   // 15 Hz component
    float freq3 = 30.0f;   // 30 Hz component

    for (int i = 0; i < N; i++) {
        float t = (float)i / N;
        signal[i] = sin(2.0f * PI * freq1 * t) +
                   0.5f * sin(2.0f * PI * freq2 * t) +
                   0.3f * sin(2.0f * PI * freq3 * t);
    }
}

void printSpectrum(cufftComplex* spectrum, int N, int maxDisplay = 50) {
    printf("\nFrequency Spectrum (magnitude):\n");
    printf("Bin\tFreq\tMagnitude\n");

    int display = (N/2 < maxDisplay) ? N/2 : maxDisplay;

    for (int i = 0; i < display; i++) {
        float magnitude = sqrt(spectrum[i].x * spectrum[i].x +
                             spectrum[i].y * spectrum[i].y);
        float frequency = (float)i;

        if (magnitude > 10.0f) {  // Only show significant peaks
            printf("%d\t%.1f\t%.2f\n", i, frequency, magnitude);
        }
    }
}

int main() {
    const int N = 1024;  // Signal length (power of 2 for efficiency)
    const int batchSize = 1;

    size_t bytes_signal = N * sizeof(cufftReal);
    size_t bytes_spectrum = (N/2 + 1) * sizeof(cufftComplex);

    // Host memory
    cufftReal *h_signal = (cufftReal*)malloc(bytes_signal);
    cufftComplex *h_spectrum = (cufftComplex*)malloc(bytes_spectrum);
    cufftReal *h_reconstructed = (cufftReal*)malloc(bytes_signal);

    // Generate test signal
    generateSignal(h_signal, N);

    // Device memory
    cufftReal *d_signal;
    cufftComplex *d_spectrum;
    CHECK_CUDA(cudaMalloc(&d_signal, bytes_signal));
    CHECK_CUDA(cudaMalloc(&d_spectrum, bytes_spectrum));

    CHECK_CUDA(cudaMemcpy(d_signal, h_signal, bytes_signal, cudaMemcpyHostToDevice));

    // Create cuFFT plans
    cufftHandle plan_forward, plan_inverse;

    // Forward plan: Real to Complex
    CHECK_CUFFT(cufftPlan1d(&plan_forward, N, CUFFT_R2C, batchSize));

    // Inverse plan: Complex to Real
    CHECK_CUFFT(cufftPlan1d(&plan_inverse, N, CUFFT_C2R, batchSize));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Forward FFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecR2C(plan_forward, d_signal, d_spectrum));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_forward = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_forward, start, stop));

    // Copy spectrum to host
    CHECK_CUDA(cudaMemcpy(h_spectrum, d_spectrum, bytes_spectrum, cudaMemcpyDeviceToHost));

    // Inverse FFT
    CHECK_CUDA(cudaEventRecord(start));
    CHECK_CUFFT(cufftExecC2R(plan_inverse, d_spectrum, d_signal));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_inverse = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_inverse, start, stop));

    // Copy reconstructed signal
    CHECK_CUDA(cudaMemcpy(h_reconstructed, d_signal, bytes_signal, cudaMemcpyDeviceToHost));

    // Normalize inverse FFT result
    for (int i = 0; i < N; i++) {
        h_reconstructed[i] /= N;
    }

    // Display results
    printf("=== 1D FFT Signal Processing ===\n");
    printf("Signal length: %d samples\n", N);
    printf("Forward FFT time: %.3f ms\n", ms_forward);
    printf("Inverse FFT time: %.3f ms\n", ms_inverse);

    // Show original signal sample
    printf("\nOriginal signal (first 10 samples):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_signal[i]);
    }
    printf("\n");

    // Show frequency spectrum
    printSpectrum(h_spectrum, N);

    // Verify reconstruction accuracy
    float max_error = 0.0f;
    for (int i = 0; i < N; i++) {
        float error = fabs(h_signal[i] - h_reconstructed[i]);
        if (error > max_error) max_error = error;
    }

    printf("\nReconstruction max error: %.6f\n", max_error);

    // Show reconstructed signal sample
    printf("Reconstructed signal (first 10 samples):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.4f ", h_reconstructed[i]);
    }
    printf("\n");

    // Cleanup
    CHECK_CUFFT(cufftDestroy(plan_forward));
    CHECK_CUFFT(cufftDestroy(plan_inverse));
    CHECK_CUDA(cudaFree(d_signal));
    CHECK_CUDA(cudaFree(d_spectrum));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    free(h_signal);
    free(h_spectrum);
    free(h_reconstructed);

    printf("\ncuFFT 1D operations completed successfully!\n");
    return 0;
}
