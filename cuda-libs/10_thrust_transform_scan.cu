// Thrust: Transformations, scans, and functional programming
// Essential for data preprocessing, normalization, and feature engineering

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}

// Custom functors for transformations
struct relu_functor {
    __host__ __device__
    float operator()(float x) const {
        return (x > 0.0f) ? x : 0.0f;
    }
};

struct sigmoid_functor {
    __host__ __device__
    float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct normalize_functor {
    float mean, stddev;

    normalize_functor(float m, float s) : mean(m), stddev(s) {}

    __host__ __device__
    float operator()(float x) const {
        return (x - mean) / stddev;
    }
};

struct saxpy_functor {
    float a;
    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
    float operator()(const thrust::tuple<float, float>& t) const {
        return a * thrust::get<0>(t) + thrust::get<1>(t);
    }
};

int main() {
    const int N = 1000000;

    printf("=== Thrust Transform & Scan Operations ===\n");
    printf("Data size: %d elements\n\n", N);

    // Initialize data
    thrust::host_vector<float> h_data(N);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 2000 - 1000) / 100.0f;  // Range: -10 to 10
    }

    thrust::device_vector<float> d_input = h_data;
    thrust::device_vector<float> d_output(N);

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // 1. TRANSFORM: Apply ReLU activation
    CHECK_CUDA(cudaEventRecord(start));
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), relu_functor());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_relu = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_relu, start, stop));

    // 2. TRANSFORM: Apply Sigmoid activation
    CHECK_CUDA(cudaEventRecord(start));
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), sigmoid_functor());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_sigmoid = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_sigmoid, start, stop));

    // 3. NORMALIZE: Z-score normalization
    float sum = thrust::reduce(d_input.begin(), d_input.end(), 0.0f);
    float mean = sum / N;

    thrust::device_vector<float> d_squared(N);
    thrust::transform(d_input.begin(), d_input.end(), d_squared.begin(),
                     [=] __device__ (float x) { return (x - mean) * (x - mean); });
    float variance = thrust::reduce(d_squared.begin(), d_squared.end(), 0.0f) / N;
    float stddev = sqrtf(variance);

    CHECK_CUDA(cudaEventRecord(start));
    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(),
                     normalize_functor(mean, stddev));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_normalize = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_normalize, start, stop));

    // 4. SCAN: Cumulative sum (prefix sum)
    CHECK_CUDA(cudaEventRecord(start));
    thrust::inclusive_scan(d_input.begin(), d_input.end(), d_output.begin());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_scan = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_scan, start, stop));

    // 5. EXCLUSIVE SCAN: Useful for parallel algorithms
    thrust::device_vector<int> d_indices(N);
    for (int i = 0; i < N; i++) d_indices[i] = 1;

    thrust::device_vector<int> d_offsets(N);
    CHECK_CUDA(cudaEventRecord(start));
    thrust::exclusive_scan(d_indices.begin(), d_indices.end(), d_offsets.begin());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_exclusive_scan = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_exclusive_scan, start, stop));

    // 6. SAXPY: a*x + y (using zip iterators)
    thrust::device_vector<float> d_y(N);
    for (int i = 0; i < N; i++) d_y[i] = (float)(rand() % 100) / 10.0f;

    float alpha = 2.5f;
    CHECK_CUDA(cudaEventRecord(start));
    thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(d_input.begin(), d_y.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(d_input.end(), d_y.end())),
                     d_output.begin(),
                     saxpy_functor(alpha));
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_saxpy = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_saxpy, start, stop));

    // 7. GENERATE: Fill with sequence
    CHECK_CUDA(cudaEventRecord(start));
    thrust::copy(thrust::counting_iterator<int>(0),
                thrust::counting_iterator<int>(N),
                d_offsets.begin());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms_generate = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms_generate, start, stop));

    // Results
    printf("=== Performance ===\n");
    printf("ReLU transform:        %.3f ms (%.2f B elem/sec)\n",
           ms_relu, N / (ms_relu * 1e6));
    printf("Sigmoid transform:     %.3f ms (%.2f B elem/sec)\n",
           ms_sigmoid, N / (ms_sigmoid * 1e6));
    printf("Normalize:             %.3f ms (%.2f B elem/sec)\n",
           ms_normalize, N / (ms_normalize * 1e6));
    printf("Inclusive scan:        %.3f ms (%.2f B elem/sec)\n",
           ms_scan, N / (ms_scan * 1e6));
    printf("Exclusive scan:        %.3f ms (%.2f B elem/sec)\n",
           ms_exclusive_scan, N / (ms_exclusive_scan * 1e6));
    printf("SAXPY:                 %.3f ms (%.2f B elem/sec)\n",
           ms_saxpy, N / (ms_saxpy * 1e6));
    printf("Generate sequence:     %.3f ms (%.2f B elem/sec)\n",
           ms_generate, N / (ms_generate * 1e6));

    // Sample outputs
    thrust::host_vector<float> h_result = d_output;

    printf("\n=== Normalization Stats ===\n");
    printf("Original mean:    %.4f\n", mean);
    printf("Original stddev:  %.4f\n", stddev);

    // Verify normalization
    thrust::device_vector<float> d_normalized(N);
    thrust::transform(d_input.begin(), d_input.end(), d_normalized.begin(),
                     normalize_functor(mean, stddev));
    float new_mean = thrust::reduce(d_normalized.begin(), d_normalized.end(), 0.0f) / N;
    printf("Normalized mean:  %.4f (should be ~0)\n", new_mean);

    printf("\n=== Sample Transformations (first 10) ===\n");
    thrust::host_vector<float> h_input = d_input;

    printf("Original:    ");
    for (int i = 0; i < 10; i++) printf("%6.2f ", h_input[i]);
    printf("\n");

    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), relu_functor());
    h_result = d_output;
    printf("ReLU:        ");
    for (int i = 0; i < 10; i++) printf("%6.2f ", h_result[i]);
    printf("\n");

    thrust::transform(d_input.begin(), d_input.end(), d_output.begin(), sigmoid_functor());
    h_result = d_output;
    printf("Sigmoid:     ");
    for (int i = 0; i < 10; i++) printf("%6.2f ", h_result[i]);
    printf("\n");

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("\nThrust transform & scan operations completed successfully!\n");
    return 0;
}
