/*
 * ex04_cnn_layers.cu
 * Problem 4: Tiled GEMM vs cuBLAS & CNN Layer Benchmarking
 *
 * Part A - Naive matmul, tiled matmul, cuBLAS comparison across 5 sizes
 * Part B - CNN layer benchmarks: Conv2D, BatchNorm, MaxPool
 * Part C - im2col convolution kernel
 *
 * Compile: nvcc -O2 -arch=sm_86 ex04_cnn_layers.cu -o ex04 -lcublas
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cublas_v2.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  code=%d \"%s\"\n",          \
                    __FILE__, __LINE__, (int)err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define CUBLAS_CHECK(call)                                                    \
    do {                                                                      \
        cublasStatus_t stat = call;                                           \
        if (stat != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS error at %s:%d  status=%d\n",             \
                    __FILE__, __LINE__, (int)stat);                            \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define TILE_SIZE 16

/* ===================================================================
 * Naive matmul (no shared memory)
 * =================================================================== */
__global__ void naiveMatMul(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k)
            sum += A[row * K + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

/* ===================================================================
 * Tiled matmul using shared memory  (TODO B1 — all 5 steps completed)
 * =================================================================== */
__global__ void tiledMatMul(const float *A, const float *B, float *C,
                            int M, int N, int K) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x,  by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    float pValue = 0.0f;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        // Step 1: load tile of A into shared memory
        int aCol = m * TILE_SIZE + tx;
        sA[ty][tx] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;

        // Step 2: load tile of B into shared memory
        int bRow = m * TILE_SIZE + ty;
        sB[ty][tx] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        // Step 3: synchronise before computation
        __syncthreads();

        // Step 4: multiply the two tiles
        for (int k = 0; k < TILE_SIZE; ++k)
            pValue += sA[ty][k] * sB[k][tx];

        // Step 5: synchronise before loading next tile
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = pValue;
}

/* ===================================================================
 * CNN layers — MaxPool2D  (TODO C1 — completed)
 * =================================================================== */
__global__ void maxPool2D(const float *input, float *output,
                          int N, int C, int H, int W, int KH, int KW) {
    int H_out = H / KH;
    int W_out = W / KW;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c     = (idx / (W_out * H_out)) % C;
    int n     = idx / (W_out * H_out * C);

    float mx = -1e20f;
    for (int i = 0; i < KH; ++i) {
        for (int j = 0; j < KW; ++j) {
            int h_in = h_out * KH + i;
            int w_in = w_out * KW + j;
            float val = input[n * C * H * W + c * H * W + h_in * W + w_in];
            mx = fmaxf(mx, val);
        }
    }
    output[idx] = mx;
}

/* ===================================================================
 * CNN layers — BatchNorm inference  (TODO C2 — completed)
 * =================================================================== */
__global__ void batchNormInference(const float *input, float *output,
                                   const float *gamma, const float *beta,
                                   const float *running_mean, const float *running_var,
                                   float eps, int N, int C, int H, int W) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    int c = (idx / (H * W)) % C;
    float norm = (input[idx] - running_mean[c]) / sqrtf(running_var[c] + eps);
    output[idx] = gamma[c] * norm + beta[c];
}

/* ===================================================================
 * CNN layers — Naive Conv2D  (3x3, same padding, single output channel)
 * =================================================================== */
__global__ void conv2d_naive(const float *input, const float *filter, float *output,
                             int N, int C_in, int H, int W, int C_out,
                             int kH, int kW) {
    int H_out = H, W_out = W;      // same padding
    int padH = kH / 2, padW = kW / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int co    = (idx / (W_out * H_out)) % C_out;
    int n     = idx / (W_out * H_out * C_out);

    float sum = 0.0f;
    for (int ci = 0; ci < C_in; ++ci) {
        for (int kh = 0; kh < kH; ++kh) {
            for (int kw = 0; kw < kW; ++kw) {
                int h_in = h_out + kh - padH;
                int w_in = w_out + kw - padW;
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float iv = input[n * C_in * H * W + ci * H * W + h_in * W + w_in];
                    float fv = filter[co * C_in * kH * kW + ci * kH * kW + kh * kW + kw];
                    sum += iv * fv;
                }
            }
        }
    }
    output[idx] = sum;
}

/* ===================================================================
 * Part C — im2col convolution  (stretch)
 * =================================================================== */
__global__ void im2col_kernel(const float *input, float *col,
                              int N, int C, int H, int W,
                              int kH, int kW, int padH, int padW,
                              int H_out, int W_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * kH * kW * H_out * W_out;
    if (idx >= total) return;

    int w_col = idx % W_out;
    int h_col = (idx / W_out) % H_out;
    int k_idx = (idx / (W_out * H_out)) % (C * kH * kW);
    int n     = idx / (W_out * H_out * C * kH * kW);

    int c  = k_idx / (kH * kW);
    int kh = (k_idx / kW) % kH;
    int kw = k_idx % kW;

    int h_in = h_col + kh - padH;
    int w_in = w_col + kw - padW;

    float val = 0.0f;
    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
        val = input[n * C * H * W + c * H * W + h_in * W + w_in];

    // col layout: [N, C*kH*kW, H_out*W_out]
    col[n * (C * kH * kW * H_out * W_out) + k_idx * (H_out * W_out) + h_col * W_out + w_col] = val;
}

/* ===================================================================
 * Benchmark helpers
 * =================================================================== */

static float timeKernel(cudaEvent_t &start, cudaEvent_t &stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void fillRandom(float *arr, int n) {
    for (int i = 0; i < n; ++i)
        arr[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
}

/* ===================================================================
 * Part A — GEMM Benchmark: naive, tiled, cuBLAS  (TODO B2 — completed)
 * =================================================================== */
static void runGemmBenchmark() {
    printf("========== Part A: GEMM Benchmark ==========\n");
    printf("%-8s  %10s %10s  %10s %10s  %10s %10s\n",
           "Size", "Naive(ms)", "GFLOPS", "Tiled(ms)", "GFLOPS", "cuBLAS(ms)", "GFLOPS");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    int sizes[] = {128, 256, 512, 1024, 2048};
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int s = 0; s < 5; ++s) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        size_t sA = M * K * sizeof(float);
        size_t sB = K * N * sizeof(float);
        size_t sC = M * N * sizeof(float);

        float *h_A = (float *)malloc(sA);
        float *h_B = (float *)malloc(sB);
        fillRandom(h_A, M * K);
        fillRandom(h_B, K * N);

        float *d_A, *d_B, *d_C;
        CUDA_CHECK(cudaMalloc(&d_A, sA));
        CUDA_CHECK(cudaMalloc(&d_B, sB));
        CUDA_CHECK(cudaMalloc(&d_C, sC));
        CUDA_CHECK(cudaMemcpy(d_A, h_A, sA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, sB, cudaMemcpyHostToDevice));

        dim3 dimBlock(TILE_SIZE, TILE_SIZE);
        dim3 dimGrid((N + TILE_SIZE - 1) / TILE_SIZE,
                     (M + TILE_SIZE - 1) / TILE_SIZE);

        double flops = 2.0 * M * N * K;

        // warmup
        naiveMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // naive
        CUDA_CHECK(cudaEventRecord(start));
        naiveMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        float naive_ms = timeKernel(start, stop);
        double naive_gf = flops / (naive_ms / 1e3) / 1e9;

        // tiled
        tiledMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        tiledMatMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        float tiled_ms = timeKernel(start, stop);
        double tiled_gf = flops / (tiled_ms / 1e3) / 1e9;

        // cuBLAS (column-major, so we compute C^T = B^T * A^T)
        float alpha = 1.0f, beta_blas = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta_blas, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(start));
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                    &alpha, d_B, N, d_A, K, &beta_blas, d_C, N);
        CUDA_CHECK(cudaEventRecord(stop));
        float cublas_ms = timeKernel(start, stop);
        double cublas_gf = flops / (cublas_ms / 1e3) / 1e9;

        printf("%-8d  %10.3f %10.1f  %10.3f %10.1f  %10.3f %10.1f\n",
               sizes[s], naive_ms, naive_gf, tiled_ms, tiled_gf, cublas_ms, cublas_gf);

        free(h_A); free(h_B);
        CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nWhy tiled underperforms cuBLAS:\n"
           "  cuBLAS exploits hardware-specific features unavailable to hand-written\n"
           "  kernels: Tensor Cores (mixed-precision fused multiply-add on 4x4\n"
           "  matrices), vectorised 128-bit global memory loads via LDG.128,\n"
           "  register-level tiling with much larger effective tile sizes, double\n"
           "  buffering of shared memory tiles to overlap compute and loads, and\n"
           "  auto-tuned kernel selection per GPU architecture.  These optimisations\n"
           "  together allow cuBLAS to approach the theoretical peak FLOPS of the\n"
           "  device, while a simple TILE=16 shared-memory kernel is bottlenecked by\n"
           "  low arithmetic intensity per shared-memory load and idle cycles during\n"
           "  synchronisation barriers.\n");
}

/* ===================================================================
 * Part B — CNN Layer Benchmarks  ([32, 64, 14, 14] tensors)
 * =================================================================== */
static void runCnnLayerBenchmark() {
    printf("\n========== Part B: CNN Layer Benchmarks ==========\n");

    int batchN = 32, C = 64, H = 14, W = 14;
    int total = batchN * C * H * W;
    size_t bytes = total * sizeof(float);

    float *h_in = (float *)malloc(bytes);
    fillRandom(h_in, total);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // large enough for all outputs
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int bs = 256;

    // Conv2D (3x3, same padding, C_out = C_in = 64)
    int C_out = 64, kH = 3, kW = 3;
    int conv_out_total = batchN * C_out * H * W;
    int filter_size = C_out * C * kH * kW;
    float *h_filter = (float *)malloc(filter_size * sizeof(float));
    fillRandom(h_filter, filter_size);
    float *d_filter, *d_conv_out;
    CUDA_CHECK(cudaMalloc(&d_filter, filter_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_conv_out, conv_out_total * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filter, h_filter, filter_size * sizeof(float), cudaMemcpyHostToDevice));

    int nb_conv = (conv_out_total + bs - 1) / bs;
    conv2d_naive<<<nb_conv, bs>>>(d_in, d_filter, d_conv_out,
                                  batchN, C, H, W, C_out, kH, kW);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    conv2d_naive<<<nb_conv, bs>>>(d_in, d_filter, d_conv_out,
                                  batchN, C, H, W, C_out, kH, kW);
    CUDA_CHECK(cudaEventRecord(stop));
    float conv_ms = timeKernel(start, stop);

    // BatchNorm inference
    float *h_gamma = (float *)malloc(C * sizeof(float));
    float *h_beta  = (float *)malloc(C * sizeof(float));
    float *h_mean  = (float *)malloc(C * sizeof(float));
    float *h_var   = (float *)malloc(C * sizeof(float));
    for (int c = 0; c < C; ++c) {
        h_gamma[c] = 1.0f; h_beta[c] = 0.0f;
        h_mean[c] = 0.0f;  h_var[c] = 1.0f;
    }
    float *d_gamma, *d_beta, *d_mean, *d_var;
    CUDA_CHECK(cudaMalloc(&d_gamma, C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mean,  C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_var,   C * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma, C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta,  h_beta,  C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean,  h_mean,  C * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_var,   h_var,   C * sizeof(float), cudaMemcpyHostToDevice));

    int nb_bn = (total + bs - 1) / bs;
    batchNormInference<<<nb_bn, bs>>>(d_in, d_out, d_gamma, d_beta,
                                      d_mean, d_var, 1e-5f,
                                      batchN, C, H, W);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    batchNormInference<<<nb_bn, bs>>>(d_in, d_out, d_gamma, d_beta,
                                      d_mean, d_var, 1e-5f,
                                      batchN, C, H, W);
    CUDA_CHECK(cudaEventRecord(stop));
    float bn_ms = timeKernel(start, stop);

    // MaxPool 2x2
    int H_out_pool = H / 2, W_out_pool = W / 2;
    int pool_total = batchN * C * H_out_pool * W_out_pool;
    float *d_pool_out;
    CUDA_CHECK(cudaMalloc(&d_pool_out, pool_total * sizeof(float)));

    int nb_pool = (pool_total + bs - 1) / bs;
    maxPool2D<<<nb_pool, bs>>>(d_in, d_pool_out, batchN, C, H, W, 2, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    maxPool2D<<<nb_pool, bs>>>(d_in, d_pool_out, batchN, C, H, W, 2, 2);
    CUDA_CHECK(cudaEventRecord(stop));
    float pool_ms = timeKernel(start, stop);

    printf("%-16s  %12s\n", "Layer", "Time (ms)");
    printf("%-16s  %12.4f\n", "Conv2D 3x3", conv_ms);
    printf("%-16s  %12.4f\n", "BatchNorm", bn_ms);
    printf("%-16s  %12.4f\n", "MaxPool 2x2", pool_ms);

    free(h_in); free(h_filter);
    free(h_gamma); free(h_beta); free(h_mean); free(h_var);
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_filter)); CUDA_CHECK(cudaFree(d_conv_out));
    CUDA_CHECK(cudaFree(d_gamma)); CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_mean)); CUDA_CHECK(cudaFree(d_var));
    CUDA_CHECK(cudaFree(d_pool_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

/* ===================================================================
 * main
 * =================================================================== */
int main() {
    srand(42);
    runGemmBenchmark();
    runCnnLayerBenchmark();

    printf("\nAll ex04 experiments completed.\n");
    return 0;
}
