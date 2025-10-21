# CUDA Libraries Collection
## Comprehensive Examples for AI and Parallel Computing

A production-ready collection of 24+ CUDA library examples optimized for deep learning and high-performance computing.

---

## 📚 Library Coverage

### **Linear Algebra (cuBLAS)**
- `01_cublas_gemm.cu` - Dense matrix multiplication (GEMM)
- `02_cublas_batched.cu` - Batched operations for transformers
- `03_cublas_tensor_core.cu` - FP16 mixed precision with Tensor Cores
- `03_cublas_tensor_core_compat.cu` - Colab-compatible version

### **Signal Processing (cuFFT)**
- `04_cufft_1d.cu` - 1D FFT for audio/time-series
- `05_cufft_2d.cu` - 2D FFT for images

### **Random Generation (cuRAND)**
- `06_curand.cu` - Uniform, normal, log-normal distributions

### **Sparse Operations (cuSPARSE)**
- `07_cusparse_spmv.cu` - Sparse matrix-vector (GNNs)
- `08_cusparse_spmm.cu` - Sparse matrix-matrix

### **High-Level Algorithms (Thrust)**
- `09_thrust_sort_reduce.cu` - Sort, reduce, statistics
- `10_thrust_transform_scan.cu` - Transformations, activations

### **Low-Level Primitives (CUB)**
- `11_cub_block_primitives.cu` - Block-level operations
- `12_cub_device_reduce.cu` - Device-wide primitives

### **Deep Learning (cuDNN)**
- `13_cudnn_convolution.cu` - 2D convolution
- `14_cudnn_pooling_activation.cu` - Pooling & activations
- `15_cudnn_batchnorm.cu` - Batch normalization
- `16_cudnn_rnn.cu` - LSTM/RNN layers

### **Image Processing (NPP)**
- `17_npp_image_filter.cu` - Gaussian, Sobel, median filters

### **Linear Solvers (cuSOLVER)**
- `18_cusolver_linear.cu` - LU, Cholesky, QR, SVD

### **Advanced Patterns**
- `19_cooperative_groups.cu` - Advanced synchronization
- `20_multi_stream.cu` - Concurrent execution
- `21_unified_memory.cu` - Automatic memory management
- `22_attention_mechanism.cu` - Transformer attention
- `23_multi_gpu_nccl.cu` - Multi-GPU communication
- `24_cuda_graphs.cu` - Reduce launch overhead

---

## 🚀 Quick Start

### Compilation Examples

```bash
# cuBLAS
nvcc -o gemm 01_cublas_gemm.cu -lcublas

# cuDNN
nvcc -o conv 13_cudnn_convolution.cu -lcudnn

# Thrust (header-only)
nvcc -o sort 09_thrust_sort_reduce.cu

# Multiple libraries
nvcc -o solver 18_cusolver_linear.cu -lcusolver -lcublas

# Multi-GPU with NCCL
nvcc -o multi_gpu 23_multi_gpu_nccl.cu -lnccl

# Attention mechanism
nvcc -o attention 22_attention_mechanism.cu -lcublas
```

### Architecture-Specific Compilation

```bash
# For specific GPU architecture
nvcc -arch=sm_75 -o program file.cu -lcublas  # Turing (RTX 20xx)
nvcc -arch=sm_80 -o program file.cu -lcublas  # Ampere (A100, RTX 30xx)
nvcc -arch=sm_86 -o program file.cu -lcublas  # Ampere (RTX 30xx)
nvcc -arch=sm_89 -o program file.cu -lcublas  # Ada (RTX 40xx)
nvcc -arch=sm_90 -o program file.cu -lcublas  # Hopper (H100)
```

### Optimization Flags

```bash
# Fast math optimizations
nvcc -O3 -use_fast_math -o program file.cu -lcublas

# Maximum performance
nvcc -O3 -use_fast_math --maxrregcount=64 -o program file.cu -lcublas

# Debug mode
nvcc -g -G -o program file.cu -lcublas
```

---

## 🔧 Google Colab Setup

```python
# Check GPU
!nvidia-smi
!nvcc --version

# Upload files to /content
# Then compile
!nvcc -o gemm 01_cublas_gemm.cu -lcublas

# Run
!./gemm
```

### Fix Driver/Runtime Mismatch
```python
# Use compatible version for older GPUs
!nvcc -o tensor_core 03_cublas_tensor_core_compat.cu -lcublas

# Or specify architecture
!nvcc -arch=sm_75 -o tensor_core 03_cublas_tensor_core.cu -lcublas
```

---

## 📦 Dependencies

### Required Libraries

| Library | Purpose | Link Flag |
|---------|---------|-----------|
| cuBLAS | Linear algebra | `-lcublas` |
| cuFFT | Fourier transforms | `-lcufft` |
| cuRAND | Random numbers | `-lcurand` |
| cuSPARSE | Sparse operations | `-lcusparse` |
| cuDNN | Deep learning | `-lcudnn` |
| NPP | Image processing | `-lnppc -lnppi -lnppig` |
| cuSOLVER | Linear solvers | `-lcusolver` |
| NCCL | Multi-GPU | `-lnccl` |

### Installation

**Ubuntu/Debian:**
```bash
# CUDA Toolkit (includes cuBLAS, cuFFT, cuRAND, cuSPARSE, cuSOLVER, Thrust, CUB)
sudo apt-get install nvidia-cuda-toolkit

# cuDNN
# Download from NVIDIA Developer: https://developer.nvidia.com/cudnn

# NCCL
sudo apt-get install libnccl2 libnccl-dev

# NPP (included in CUDA Toolkit)
```

**Check Installed Libraries:**
```bash
ldconfig -p | grep cublas
ldconfig -p | grep cudnn
ldconfig -p | grep nccl
```

---

## 💡 Usage Patterns

### Error Checking
All examples include comprehensive error checking:
```cpp
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
```

### Performance Timing
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel execution ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
```

### Memory Management
```cpp
// Traditional
float *d_data;
cudaMalloc(&d_data, bytes);
cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

// Unified Memory (simpler)
float *data;
cudaMallocManaged(&data, bytes);
// Use directly on CPU and GPU
```

---

## 🎯 Performance Tips

1. **Use the right tool:**
   - cuBLAS for dense linear algebra
   - cuSPARSE for sparse operations (>95% zeros)
   - Thrust for quick prototyping
   - CUB for optimized custom kernels

2. **Memory optimization:**
   - Use pinned memory for faster transfers
   - Prefetch with Unified Memory
   - Minimize host-device transfers

3. **Compute optimization:**
   - Enable Tensor Cores with FP16 (7x faster)
   - Use CUDA Graphs for repeated patterns
   - Multi-stream for concurrency

4. **Multi-GPU:**
   - NCCL for efficient communication
   - AllReduce for gradient synchronization
   - Model parallelism for large models

---

## 🔍 Debugging

```bash
# Compute sanitizer
compute-sanitizer ./program

# Memory checker
cuda-memcheck ./program

# Profiling
nsys profile --stats=true ./program
ncu --set full ./program
```

---

## 📊 Benchmarking

Each example includes:
- ✓ Performance metrics (GFLOPS, GB/s)
- ✓ Timing information
- ✓ Throughput calculations
- ✓ Sample outputs for verification

---

## 🎓 Learning Path

**Beginners:**
1. Start with `09_thrust_sort_reduce.cu` (high-level)
2. Try `01_cublas_gemm.cu` (linear algebra)
3. Explore `20_multi_stream.cu` (concurrency)

**Intermediate:**
1. `13_cudnn_convolution.cu` (deep learning)
2. `11_cub_block_primitives.cu` (optimizations)
3. `22_attention_mechanism.cu` (transformers)

**Advanced:**
1. `23_multi_gpu_nccl.cu` (distributed)
2. `24_cuda_graphs.cu` (optimization)
3. `19_cooperative_groups.cu` (advanced patterns)

---

## 📝 Example Output

```
=== cuBLAS GEMM Matrix Multiplication ===
Matrix: 1024x1024x1024
Time: 2.456 ms
Performance: 876.54 GFLOPS
```

---

## 🤝 Contributing

These examples are designed to be:
- **Clear:** Minimal but important comments
- **Production-ready:** Full error checking
- **Educational:** Performance metrics included
- **Modern:** Latest CUDA best practices

---

## 📄 License

Free to use for learning and commercial applications.

---

## 🔗 Resources

- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/)
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

---

## ⚡ Performance Reference

**Typical Numbers (RTX 3090):**
- GEMM: 800-1000 GFLOPS (FP32), 3000+ GFLOPS (FP16 Tensor Core)
- FFT: 200-300 GB/s
- Convolution: 400-600 GFLOPS
- Memory Bandwidth: 800-900 GB/s

---

**Created for AI/ML practitioners and HPC developers**
