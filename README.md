# Assignment 8: GPU Accelerated Machine Learning

CUDA C implementations for all 5 exercises covering GPU kernel profiling, memory hierarchy, ML primitives, CNN layers, and end-to-end MNIST training.

## Files

| File | Topic | Compile |
|------|-------|---------|
| `ex01_cuda_basics.cu` | GPU architecture, bandwidth, launch config, warp divergence | `nvcc -O2 -arch=sm_86 ex01_cuda_basics.cu -o ex01` |
| `ex02_memory_hierarchy.cu` | Shared memory, tree/warp reduction, bank conflicts, histogram | `nvcc -O2 -arch=sm_86 ex02_memory_hierarchy.cu -o ex02` |
| `ex03_ml_primitives.cu` | Activations, BCE/CE loss, CE gradient, fused Adam | `nvcc -O2 -arch=sm_86 ex03_ml_primitives.cu -o ex03 -lm` |
| `ex04_cnn_layers.cu` | Tiled GEMM, cuBLAS, Conv2D, BatchNorm, MaxPool, im2col | `nvcc -O2 -arch=sm_86 ex04_cnn_layers.cu -o ex04 -lcublas` |
| `ex05_mnist_cnn.cu` | Full cuDNN+cuBLAS MNIST CNN pipeline, ablation study | `nvcc -O2 -arch=sm_86 ex05_mnist_cnn.cu -o ex05 -lcudnn -lcublas` |

## Build

```bash
make all      # builds all 5 executables
make clean    # removes binaries
```

## Requirements

- NVIDIA GPU with Compute Capability 6.0+ (GTX 1060 or better)
- CUDA Toolkit 12.x
- cuBLAS and cuDNN libraries
- 6 GB+ GPU VRAM recommended for ex05

## Key Observations

- **Bandwidth**: H2D/D2H transfers dominate for small N. GPU advantage appears around N = 2^18 where arithmetic intensity amortises PCIe overhead.
- **Warp Divergence**: Branching within a warp serialises both paths, reducing throughput. Branch-free arithmetic predicates eliminate the penalty.
- **Shared Memory**: Stride-1 access avoids bank conflicts; stride-32 causes full serialisation. Padding (`tile[16][17]`) resolves column-major conflict patterns.
- **cuBLAS vs Custom GEMM**: Tiled shared-memory matmul improves over naive but cuBLAS leverages Tensor Cores, vectorised loads, and auto-tuned kernels to approach peak FLOPS.
- **MNIST CNN**: BatchNorm + Adam converges fastest and achieves highest accuracy (~97.5%). Data augmentation provides marginal gains on MNIST.
