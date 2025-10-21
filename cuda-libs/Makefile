# Makefile for CUDA Libraries Examples
# Usage: make all, make cublas, make clean, etc.

NVCC = nvcc
NVCC_FLAGS = -O3 -use_fast_math

# Detect GPU architecture (or specify manually)
ARCH ?= sm_75
NVCC_FLAGS += -arch=$(ARCH)

# Library flags
CUBLAS_FLAGS = -lcublas
CUFFT_FLAGS = -lcufft
CURAND_FLAGS = -lcurand
CUSPARSE_FLAGS = -lcusparse
CUDNN_FLAGS = -lcudnn
NPP_FLAGS = -lnppc -lnppi -lnppig
CUSOLVER_FLAGS = -lcusolver -lcublas
NCCL_FLAGS = -lnccl

# Output directory
OUT_DIR = bin
$(shell mkdir -p $(OUT_DIR))

# All targets
ALL_TARGETS = cublas_gemm cublas_batched cublas_tensor_core \
              cufft_1d cufft_2d curand cusparse_spmv cusparse_spmm \
              thrust_sort thrust_transform cub_block cub_device \
              cudnn_conv cudnn_pool cudnn_batchnorm cudnn_rnn \
              npp_filter cusolver cooperative_groups multi_stream \
              unified_memory attention multi_gpu cuda_graphs

# Default target
all: $(ALL_TARGETS)

# cuBLAS examples
cublas: cublas_gemm cublas_batched cublas_tensor_core

cublas_gemm: 01_cublas_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUBLAS_FLAGS)

cublas_batched: 02_cublas_batched.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUBLAS_FLAGS)

cublas_tensor_core: 03_cublas_tensor_core.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUBLAS_FLAGS)

cublas_tensor_compat: 03_cublas_tensor_core_compat.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUBLAS_FLAGS)

# cuFFT examples
cufft: cufft_1d cufft_2d

cufft_1d: 04_cufft_1d.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUFFT_FLAGS)

cufft_2d: 05_cufft_2d.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUFFT_FLAGS)

# cuRAND example
curand: 06_curand.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CURAND_FLAGS)

# cuSPARSE examples
cusparse: cusparse_spmv cusparse_spmm

cusparse_spmv: 07_cusparse_spmv.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUSPARSE_FLAGS)

cusparse_spmm: 08_cusparse_spmm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUSPARSE_FLAGS)

# Thrust examples
thrust: thrust_sort thrust_transform

thrust_sort: 09_thrust_sort_reduce.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

thrust_transform: 10_thrust_transform_scan.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

# CUB examples
cub: cub_block cub_device

cub_block: 11_cub_block_primitives.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

cub_device: 12_cub_device_reduce.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

# cuDNN examples
cudnn: cudnn_conv cudnn_pool cudnn_batchnorm cudnn_rnn

cudnn_conv: 13_cudnn_convolution.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUDNN_FLAGS)

cudnn_pool: 14_cudnn_pooling_activation.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUDNN_FLAGS)

cudnn_batchnorm: 15_cudnn_batchnorm.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUDNN_FLAGS)

cudnn_rnn: 16_cudnn_rnn.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUDNN_FLAGS)

# NPP example
npp_filter: 17_npp_image_filter.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(NPP_FLAGS)

# cuSOLVER example
cusolver: 18_cusolver_linear.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUSOLVER_FLAGS)

# Advanced examples
cooperative_groups: 19_cooperative_groups.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

multi_stream: 20_multi_stream.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

unified_memory: 21_unified_memory.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

attention: 22_attention_mechanism.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(CUBLAS_FLAGS)

multi_gpu: 23_multi_gpu_nccl.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@ $(NCCL_FLAGS)

cuda_graphs: 24_cuda_graphs.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $(OUT_DIR)/$@

# Group targets
basics: cublas cufft curand cusparse
algorithms: thrust cub
deeplearning: cudnn
advanced: cooperative_groups multi_stream unified_memory attention multi_gpu cuda_graphs

# Test targets (compile and run)
test-%: $(OUT_DIR)/%
	@echo "Running $*..."
	@$(OUT_DIR)/$*

# Run all tests
test-all: all
	@for target in $(ALL_TARGETS); do \
		echo ""; \
		echo "=== Testing $$target ==="; \
		$(OUT_DIR)/$$target || exit 1; \
	done

# Clean
clean:
	rm -rf $(OUT_DIR)

# Help
help:
	@echo "CUDA Libraries Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all examples"
	@echo "  cublas           - Build cuBLAS examples"
	@echo "  cufft            - Build cuFFT examples"
	@echo "  cusparse         - Build cuSPARSE examples"
	@echo "  thrust           - Build Thrust examples"
	@echo "  cub              - Build CUB examples"
	@echo "  cudnn            - Build cuDNN examples"
	@echo "  basics           - Build basic library examples"
	@echo "  algorithms       - Build algorithm examples"
	@echo "  deeplearning     - Build deep learning examples"
	@echo "  advanced         - Build advanced examples"
	@echo "  clean            - Remove compiled binaries"
	@echo "  test-<name>      - Compile and run specific example"
	@echo "  test-all         - Run all examples"
	@echo ""
	@echo "Variables:"
	@echo "  ARCH=sm_XX       - Set GPU architecture (default: sm_75)"
	@echo "                     sm_75: Turing (RTX 20xx, T4)"
	@echo "                     sm_80: Ampere (A100)"
	@echo "                     sm_86: Ampere (RTX 30xx)"
	@echo "                     sm_89: Ada (RTX 40xx)"
	@echo "                     sm_90: Hopper (H100)"
	@echo ""
	@echo "Examples:"
	@echo "  make all                    # Build everything"
	@echo "  make cublas                 # Build only cuBLAS examples"
	@echo "  make ARCH=sm_86 all         # Build for RTX 30xx"
	@echo "  make test-cublas_gemm       # Build and run GEMM"
	@echo "  make clean                  # Clean up"

# Info target
info:
	@echo "CUDA Compiler: $(NVCC)"
	@echo "Architecture: $(ARCH)"
	@echo "Flags: $(NVCC_FLAGS)"
	@echo "Output Directory: $(OUT_DIR)"
	@echo ""
	@echo "Available Libraries:"
	@which nvcc > /dev/null && echo "  ✓ CUDA Toolkit" || echo "  ✗ CUDA Toolkit"
	@ldconfig -p | grep -q libcublas && echo "  ✓ cuBLAS" || echo "  ✗ cuBLAS"
	@ldconfig -p | grep -q libcufft && echo "  ✓ cuFFT" || echo "  ✗ cuFFT"
	@ldconfig -p | grep -q libcudnn && echo "  ✓ cuDNN" || echo "  ✗ cuDNN"
	@ldconfig -p | grep -q libnccl && echo "  ✓ NCCL" || echo "  ✗ NCCL"

.PHONY: all cublas cufft cusparse thrust cub cudnn clean help info test-all \
        basics algorithms deeplearning advanced $(ALL_TARGETS)
