/*
 * ex01_cuda_basics.cu
 * Problem 1: GPU Architecture & CUDA Kernel Profiling
 *
 * Part A - Bandwidth & Speedup Analysis
 * Part B - Launch Configuration Analysis
 * Part C - Warp Divergence Experiment
 *
 * Compile: nvcc -O2 -arch=sm_86 ex01_cuda_basics.cu -o ex01
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  code=%d \"%s\"\n",          \
                    __FILE__, __LINE__, (int)err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* ===================================================================
 * Section A: Reference kernels (provided)
 * =================================================================== */

__global__ void vectorAdd(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

/* ===================================================================
 * Section B: DIY kernels (completed TODO B1, B2, B3, B4)
 * =================================================================== */

// TODO B1 -- vector scale kernel
__global__ void vectorScale(float *a, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] *= scale;
}

// TODO B2 -- squared difference kernel
__global__ void squaredDiff(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff = a[i] - b[i];
        c[i] = diff * diff;
    }
}

/* ===================================================================
 * Section C: Stretch — Warp Divergence
 * =================================================================== */

__global__ void warpDivergent(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // forces alternating threads inside each warp to execute different paths
        if (threadIdx.x % 2 == 0) {
            a[i] = a[i] * 2.0f + 1.0f;
        } else {
            a[i] = a[i] * 0.5f - 1.0f;
        }
    }
}

__global__ void warpBranchFree(float *a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float val = a[i];
        // predicate without a branch: even threads scale*2+1, odd threads scale*0.5-1
        float even = val * 2.0f + 1.0f;
        float odd  = val * 0.5f - 1.0f;
        int isEven = 1 - (threadIdx.x & 1);          // 1 for even, 0 for odd
        a[i] = isEven * even + (1 - isEven) * odd;
    }
}

/* ===================================================================
 * CPU baseline
 * =================================================================== */
static void cpuVectorAdd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) c[i] = a[i] + b[i];
}

/* ===================================================================
 * Helper: time a GPU kernel between two CUDA events (ms)
 * =================================================================== */
static float timeKernel(cudaEvent_t &start, cudaEvent_t &stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

/* ===================================================================
 * Part A — Bandwidth & Speedup Analysis
 * =================================================================== */

// TODO B4 — bandwidth measurement for transfer sizes 1,8,64,256,512 MB
static void runBandwidthBenchmark() {
    printf("\n========== Part A: Memory Bandwidth Benchmark ==========\n");
    printf("%-12s  %10s  %10s  %12s  %12s\n",
           "Size (MB)", "H2D (ms)", "D2H (ms)", "H2D (GB/s)", "D2H (GB/s)");

    size_t mbSizes[] = {1, 8, 64, 256, 512};

    for (int t = 0; t < 5; ++t) {
        size_t bytes = mbSizes[t] * 1024ULL * 1024ULL;
        float *h_buf = (float *)malloc(bytes);
        memset(h_buf, 0xAB, bytes);

        float *d_buf;
        CUDA_CHECK(cudaMalloc(&d_buf, bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        // H2D
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_buf, h_buf, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        float h2d_ms = timeKernel(start, stop);

        // D2H
        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(h_buf, d_buf, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(stop));
        float d2h_ms = timeKernel(start, stop);

        double h2d_gbps = (bytes / 1e9) / (h2d_ms / 1e3);
        double d2h_gbps = (bytes / 1e9) / (d2h_ms / 1e3);

        printf("%-12zu  %10.3f  %10.3f  %12.2f  %12.2f\n",
               mbSizes[t], h2d_ms, d2h_ms, h2d_gbps, d2h_gbps);

        free(h_buf);
        CUDA_CHECK(cudaFree(d_buf));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
}

// speedup benchmark across different N values
static void runSpeedupBenchmark() {
    printf("\n========== Part A: CPU vs GPU Speedup ==========\n");
    printf("%-14s  %10s  %12s  %10s  %10s\n",
           "N", "CPU (ms)", "GPU kern(ms)", "H2D (ms)", "Speedup");

    int exponents[] = {10, 14, 18, 22, 26};

    for (int e = 0; e < 5; ++e) {
        int n = 1 << exponents[e];
        size_t bytes = (size_t)n * sizeof(float);

        float *h_A = (float *)malloc(bytes);
        float *h_B = (float *)malloc(bytes);
        float *h_C = (float *)malloc(bytes);
        for (int i = 0; i < n; ++i) { h_A[i] = (float)(i % 100); h_B[i] = (float)(i % 37); }

        // CPU
        auto t0 = std::chrono::high_resolution_clock::now();
        cpuVectorAdd(h_A, h_B, h_C, n);
        auto t1 = std::chrono::high_resolution_clock::now();
        float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // GPU alloc + transfer
        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, bytes));
        CUDA_CHECK(cudaMalloc(&d_B, bytes));
        CUDA_CHECK(cudaMalloc(&d_C, bytes));

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop));
        float h2d_ms = timeKernel(start, stop);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;
        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<numBlocks, blockSize>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(stop));
        float kern_ms = timeKernel(start, stop);

        float speedup = cpu_ms / kern_ms;
        printf("2^%-12d  %10.4f  %12.4f  %10.4f  %10.2fx\n",
               exponents[e], cpu_ms, kern_ms, h2d_ms, speedup);

        free(h_A); free(h_B); free(h_C);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
        CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
    }

    printf("\nCrossover analysis:\n"
           "  Small N (2^10 – 2^14) favours the CPU because the PCIe transfer\n"
           "  overhead and kernel launch latency dominate the very short compute\n"
           "  time.  GPU becomes faster around N = 2^18 where there are enough\n"
           "  threads to saturate the SMs and amortise the transfer cost.\n");
}

/* ===================================================================
 * Part B — Launch Configuration Analysis
 * =================================================================== */

// TODO B3 — diy_launch_config
static void runLaunchConfigBenchmark() {
    printf("\n========== Part B: Launch Configuration ==========\n");
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    for (int i = 0; i < n; ++i) { h_A[i] = 1.0f; h_B[i] = 2.0f; }

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    int blockSizes[] = {64, 128, 256, 512, 1024};
    printf("%-18s  %12s  %12s\n", "Threads/Block", "Num Blocks", "Time (ms)");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int b = 0; b < 5; ++b) {
        int bs = blockSizes[b];
        int nb = (n + bs - 1) / bs;

        // warmup run (avoid first-call overhead)
        vectorAdd<<<nb, bs>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        vectorAdd<<<nb, bs>>>(d_A, d_B, d_C, n);
        CUDA_CHECK(cudaEventRecord(stop));
        float ms = timeKernel(start, stop);

        printf("%-18d  %12d  %12.4f\n", bs, nb, ms);
    }

    printf("\nWhy multiples of 32?\n"
           "  The GPU schedules threads in groups of 32 called warps.  If the block\n"
           "  size is not a multiple of 32, the last warp in every block contains\n"
           "  inactive lanes that still occupy scheduler resources.  Choosing a\n"
           "  multiple of 32 avoids this waste and keeps every lane productive.\n"
           "  Additionally, memory coalescing works on warp granularity; partial\n"
           "  warps may generate suboptimal memory transactions.  Values of 128–256\n"
           "  typically give the best balance between occupancy and register pressure.\n");

    free(h_A); free(h_B);
    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

/* ===================================================================
 * Part C — Warp Divergence Experiment
 * =================================================================== */
static void runWarpDivergenceExperiment() {
    printf("\n========== Part C: Warp Divergence ==========\n");
    int n = 1 << 22;
    size_t bytes = (size_t)n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    for (int i = 0; i < n; ++i) h_A[i] = (float)(i % 1000);

    float *d_A;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));

    int bs = 256;
    int nb = (n + bs - 1) / bs;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // divergent version
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    warpDivergent<<<nb, bs>>>(d_A, n);     // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    warpDivergent<<<nb, bs>>>(d_A, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float div_ms = timeKernel(start, stop);

    // branch-free version
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    warpBranchFree<<<nb, bs>>>(d_A, n);    // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(start));
    warpBranchFree<<<nb, bs>>>(d_A, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float free_ms = timeKernel(start, stop);

    printf("Divergent kernel : %.4f ms\n", div_ms);
    printf("Branch-free      : %.4f ms\n", free_ms);
    printf("Divergence penalty: %.2f%%\n", (div_ms - free_ms) / free_ms * 100.0f);
    printf("\nDiscussion:\n"
           "  When threads within the same warp follow different execution paths\n"
           "  the hardware must serialise both paths, masking out inactive threads\n"
           "  in each pass.  This effectively halves the warp's throughput for the\n"
           "  divergent section.  The branch-free formulation avoids this by using\n"
           "  arithmetic predicates so all threads execute identical instructions.\n");

    free(h_A);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

/* ===================================================================
 * Theoretical Bandwidth helper
 * =================================================================== */
static void printTheoreticalBandwidth() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("\n========== Device Info ==========\n");
    printf("Device          : %s\n", prop.name);
    printf("Compute cap     : %d.%d\n", prop.major, prop.minor);
    printf("SMs             : %d\n", prop.multiProcessorCount);
    printf("Mem clock       : %d kHz\n", prop.memoryClockRate);
    printf("Bus width       : %d bits\n", prop.memoryBusWidth);

    double bw = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0);
    bw /= 1e9;
    printf("Theoretical BW  : %.2f GB/s\n", bw);
}

/* ===================================================================
 * main
 * =================================================================== */
int main() {
    printTheoreticalBandwidth();
    runBandwidthBenchmark();
    runSpeedupBenchmark();
    runLaunchConfigBenchmark();
    runWarpDivergenceExperiment();

    printf("\nAll ex01 experiments completed.\n");
    return 0;
}
