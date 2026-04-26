/*
 * ex02_memory_hierarchy.cu
 * Problem 2: Parallel Reduction & Shared Memory Optimization
 *
 * Part A - Three reduction strategies (naive, shared-mem tree, warp shuffle)
 * Part B - Bank conflict profiling with strides + padding fix
 * Part C - Shared-memory histogram
 *
 * Compile: nvcc -O2 -arch=sm_86 ex02_memory_hierarchy.cu -o ex02
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  code=%d \"%s\"\n",          \
                    __FILE__, __LINE__, (int)err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define BLOCK_SIZE 256
#define NUM_BINS   256

/* ===================================================================
 * Part A — Three Reduction Strategies
 * =================================================================== */

// Strategy 1: naive single-thread sequential reduction (baseline)
__global__ void reduce_naive(const float *g_idata, float *g_odata, int n) {
    // only one thread does all the work — intentionally slow baseline
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) sum += g_idata[i];
        *g_odata = sum;
    }
}

// Strategy 2: shared-memory tree reduction  (TODO B2 — completed)
__global__ void reduce_tree(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0.0f;
    __syncthreads();

    // tree-based parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// TODO B2 — max reduction variant (shared memory tree)
__global__ void reduce_max_tree(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : -FLT_MAX;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// Strategy 3: warp-level reduction using __shfl_down_sync (TODO C1 — completed)
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void reduce_warp(const float *g_idata, float *g_odata, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = (i < n) ? g_idata[i] : 0.0f;

    // intra-warp reduction first
    sum = warpReduceSum(sum);

    // lane 0 of each warp writes to shared memory
    int warpId = tid / warpSize;
    int lane   = tid % warpSize;
    if (lane == 0) sdata[warpId] = sum;
    __syncthreads();

    // first warp reduces the partial sums from all warps in the block
    int numWarps = blockDim.x / warpSize;
    if (warpId == 0) {
        sum = (tid < numWarps) ? sdata[tid] : 0.0f;
        sum = warpReduceSum(sum);
    }

    if (tid == 0) atomicAdd(g_odata, sum);
}

/* ===================================================================
 * Part B — Bank Conflict Profiling  (TODO B3 — completed)
 * =================================================================== */

// kernel that reads shared memory with a configurable stride
__global__ void bankConflictDemo(float *out, int stride) {
    __shared__ float tile[1024];

    int tid = threadIdx.x;
    // initialise shared memory
    tile[tid] = (float)tid;
    __syncthreads();

    // read with given stride — stride=32 causes 32-way bank conflicts
    float val = tile[(tid * stride) % 1024];
    __syncthreads();

    // dummy write to prevent optimisation
    tile[tid] = val * 2.0f;
    __syncthreads();

    if (tid == 0) *out = tile[0];
}

// padding-based bank conflict fix: float tile[16][17] instead of [16][16]
__global__ void bankConflictPadded(float *out) {
    __shared__ float tile[16][17];   // +1 column avoids stride-16 conflicts

    int row = threadIdx.y;
    int col = threadIdx.x;

    tile[row][col] = (float)(row * 16 + col);
    __syncthreads();

    // column-major read — would normally conflict on [16][16]
    float val = tile[col][row];
    __syncthreads();

    tile[row][col] = val * 2.0f;
    __syncthreads();

    if (row == 0 && col == 0) *out = tile[0][0];
}

__global__ void bankConflictUnpadded(float *out) {
    __shared__ float tile[16][16];

    int row = threadIdx.y;
    int col = threadIdx.x;

    tile[row][col] = (float)(row * 16 + col);
    __syncthreads();

    float val = tile[col][row];
    __syncthreads();

    tile[row][col] = val * 2.0f;
    __syncthreads();

    if (row == 0 && col == 0) *out = tile[0][0];
}

/* ===================================================================
 * Part C — Histogram with Shared Memory  (TODO B4 — completed)
 * =================================================================== */

// per-block private histogram in shared memory → merge to global
__global__ void histogram_shared(const unsigned char *data, unsigned int *hist,
                                 int n) {
    __shared__ unsigned int local_hist[NUM_BINS];

    int tid = threadIdx.x;
    // zero out local histogram
    if (tid < NUM_BINS) local_hist[tid] = 0;
    __syncthreads();

    int i = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // accumulate into shared memory (much less contention than global atomics)
    while (i < n) {
        atomicAdd(&local_hist[data[i]], 1);
        i += stride;
    }
    __syncthreads();

    // merge local histogram into global histogram
    if (tid < NUM_BINS) {
        atomicAdd(&hist[tid], local_hist[tid]);
    }
}

/* ===================================================================
 * Benchmark runners
 * =================================================================== */

static float timeKernel(cudaEvent_t &start, cudaEvent_t &stop) {
    float ms = 0.0f;
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    return ms;
}

static void runReductionBenchmark() {
    printf("\n========== Part A: Reduction Strategies (N = 2^20) ==========\n");
    int n = 1 << 20;
    size_t bytes = (size_t)n * sizeof(float);
    float expected = (float)n;                  // all 1s → sum should be n

    float *h_in = (float *)malloc(bytes);
    for (int i = 0; i < n; ++i) h_in[i] = 1.0f;

    float *d_in;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // --- Naive ---
    float *d_naive_out;
    CUDA_CHECK(cudaMalloc(&d_naive_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_naive_out, 0, sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_naive<<<1, 1>>>(d_in, d_naive_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float naive_ms = timeKernel(start, stop);

    float naive_result;
    CUDA_CHECK(cudaMemcpy(&naive_result, d_naive_out, sizeof(float), cudaMemcpyDeviceToHost));

    // --- Tree ---
    float *d_tree_out;
    CUDA_CHECK(cudaMalloc(&d_tree_out, numBlocks * sizeof(float)));

    CUDA_CHECK(cudaEventRecord(start));
    reduce_tree<<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_in, d_tree_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float tree_ms = timeKernel(start, stop);

    // second pass to get final sum
    float *h_partial = (float *)malloc(numBlocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial, d_tree_out, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));
    float tree_result = 0.0f;
    for (int i = 0; i < numBlocks; ++i) tree_result += h_partial[i];

    // --- Warp shuffle ---
    float *d_warp_out;
    CUDA_CHECK(cudaMalloc(&d_warp_out, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_warp_out, 0, sizeof(float)));

    int sharedBytes = (BLOCK_SIZE / 32) * sizeof(float);
    CUDA_CHECK(cudaEventRecord(start));
    reduce_warp<<<numBlocks, BLOCK_SIZE, sharedBytes>>>(d_in, d_warp_out, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float warp_ms = timeKernel(start, stop);

    float warp_result;
    CUDA_CHECK(cudaMemcpy(&warp_result, d_warp_out, sizeof(float), cudaMemcpyDeviceToHost));

    // throughput: bytes read / time
    double gbps_naive = (bytes / 1e9) / (naive_ms / 1e3);
    double gbps_tree  = (bytes / 1e9) / (tree_ms / 1e3);
    double gbps_warp  = (bytes / 1e9) / (warp_ms / 1e3);

    printf("%-16s  %10s  %10s  %12s  %10s\n",
           "Strategy", "Time (us)", "GB/s", "Result", "Correct?");
    printf("%-16s  %10.1f  %10.2f  %12.0f  %10s\n",
           "Naive", naive_ms * 1e3, gbps_naive, naive_result,
           (fabsf(naive_result - expected) < 0.1f) ? "YES" : "NO");
    printf("%-16s  %10.1f  %10.2f  %12.0f  %10s\n",
           "Shared tree", tree_ms * 1e3, gbps_tree, tree_result,
           (fabsf(tree_result - expected) < 0.1f) ? "YES" : "NO");
    printf("%-16s  %10.1f  %10.2f  %12.0f  %10s\n",
           "Warp shuffle", warp_ms * 1e3, gbps_warp, warp_result,
           (fabsf(warp_result - expected) < 0.1f) ? "YES" : "NO");

    free(h_in); free(h_partial);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_naive_out));
    CUDA_CHECK(cudaFree(d_tree_out));
    CUDA_CHECK(cudaFree(d_warp_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

static void runBankConflictBenchmark() {
    printf("\n========== Part B: Bank Conflict Profiling ==========\n");

    float *d_out;
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int strides[] = {1, 2, 4, 8, 16, 32};
    int nIter = 100;

    printf("%-10s  %12s\n", "Stride", "Avg Time (us)");
    for (int s = 0; s < 6; ++s) {
        // warmup
        bankConflictDemo<<<1, 1024>>>(d_out, strides[s]);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(start));
        for (int r = 0; r < nIter; ++r)
            bankConflictDemo<<<1, 1024>>>(d_out, strides[s]);
        CUDA_CHECK(cudaEventRecord(stop));
        float ms = timeKernel(start, stop);

        printf("%-10d  %12.3f\n", strides[s], ms / nIter * 1e3);
    }

    printf("\nBank conflict explanation:\n"
           "  Shared memory has 32 banks.  Stride=1 means consecutive threads\n"
           "  access consecutive banks — no conflicts, full bandwidth.\n"
           "  Stride=32 means every thread hits the SAME bank (32 divides into\n"
           "  bank index modulo 32 = 0 for all), causing 32-way serialisation.\n");

    // padding benchmark
    printf("\n--- Padding fix: tile[16][17] vs tile[16][16] ---\n");

    // unpadded
    bankConflictUnpadded<<<1, dim3(16, 16)>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < nIter; ++r)
        bankConflictUnpadded<<<1, dim3(16, 16)>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    float unpadded_ms = timeKernel(start, stop);

    // padded
    bankConflictPadded<<<1, dim3(16, 16)>>>(d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int r = 0; r < nIter; ++r)
        bankConflictPadded<<<1, dim3(16, 16)>>>(d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    float padded_ms = timeKernel(start, stop);

    printf("Unpadded [16][16] : %.3f us avg\n", unpadded_ms / nIter * 1e3);
    printf("Padded   [16][17] : %.3f us avg\n", padded_ms / nIter * 1e3);
    printf("Speedup           : %.2fx\n", unpadded_ms / padded_ms);

    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

static void runHistogramBenchmark() {
    printf("\n========== Part C: Shared-Memory Histogram ==========\n");

    int n = 1 << 22;
    unsigned char *h_data = (unsigned char *)malloc(n);
    srand(42);
    for (int i = 0; i < n; ++i) h_data[i] = rand() % NUM_BINS;

    // CPU reference histogram
    unsigned int h_ref[NUM_BINS] = {0};
    for (int i = 0; i < n; ++i) h_ref[h_data[i]]++;

    unsigned char *d_data;
    unsigned int  *d_hist;
    CUDA_CHECK(cudaMalloc(&d_data, n));
    CUDA_CHECK(cudaMalloc(&d_hist, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemset(d_hist, 0, NUM_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n, cudaMemcpyHostToDevice));

    int bs = 256;
    int nb = (n + bs - 1) / bs;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    histogram_shared<<<nb, bs>>>(d_data, d_hist, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float ms = timeKernel(start, stop);

    unsigned int h_hist[NUM_BINS];
    CUDA_CHECK(cudaMemcpy(h_hist, d_hist, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    // verify
    bool correct = true;
    for (int i = 0; i < NUM_BINS; ++i) {
        if (h_hist[i] != h_ref[i]) { correct = false; break; }
    }
    printf("Histogram kernel time : %.3f ms\n", ms);
    printf("Correctness           : %s\n", correct ? "PASS" : "FAIL");

    free(h_data);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

/* ===================================================================
 * main
 * =================================================================== */
int main() {
    runReductionBenchmark();
    runBankConflictBenchmark();
    runHistogramBenchmark();

    printf("\nAll ex02 experiments completed.\n");
    return 0;
}
