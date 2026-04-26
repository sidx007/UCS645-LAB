/*
 * ex03_ml_primitives.cu
 * Problem 3: Custom ML Kernels — Activations, Loss & Backprop
 *
 * Part A - Activation function suite (sigmoid, tanh, leaky ReLU, ReLU backward)
 * Part B - Loss functions (BCE, cross-entropy with log-sum-exp, CE gradient)
 * Part C - Fused Adam optimizer kernel
 *
 * Compile: nvcc -O2 -arch=sm_86 ex03_ml_primitives.cu -o ex03 -lm
 */

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d  code=%d \"%s\"\n",          \
                    __FILE__, __LINE__, (int)err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

#define BLOCK 256

/* ===================================================================
 * Part A — Activation Function Kernels  (TODO B1–B4 completed)
 * =================================================================== */

// B1 — sigmoid forward
__global__ void sigmoid_fwd(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

// B2 — tanh forward
__global__ void tanh_fwd(const float *x, float *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tanhf(x[i]);
}

// B3 — leaky ReLU forward
__global__ void leaky_relu_fwd(const float *x, float *y, float alpha, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (x[i] > 0.0f) ? x[i] : alpha * x[i];
}

// B4 — ReLU backward (gradient)
__global__ void relu_backward(const float *grad_out, const float *x,
                              float *grad_in, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
}

/* ===================================================================
 * Part B — Loss Functions  (TODO C1, C2 completed)
 * =================================================================== */

// C1 — Binary Cross-Entropy loss with numerical clipping
__global__ void bce_loss(const float *preds, const float *labels,
                         float *loss, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float p = fminf(fmaxf(preds[i], 1e-7f), 1.0f - 1e-7f);
        loss[i] = -(labels[i] * logf(p) + (1.0f - labels[i]) * logf(1.0f - p));
    }
}

// C2 — Numerically stable cross-entropy using log-sum-exp trick
//       logits: [batch, num_classes]  row-major
//       labels: [batch]  (integer class indices stored as int)
//       loss:   [batch]  per-sample loss
__global__ void cross_entropy_loss(const float *logits, const int *labels,
                                   float *loss, int batch, int num_classes) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < batch) {
        const float *row = logits + b * num_classes;
        int label = labels[b];

        // find max for numerical stability (log-sum-exp trick)
        float maxVal = row[0];
        for (int c = 1; c < num_classes; ++c)
            maxVal = fmaxf(maxVal, row[c]);

        // log(sum(exp(x_c - max)))
        float sumExp = 0.0f;
        for (int c = 0; c < num_classes; ++c)
            sumExp += expf(row[c] - maxVal);

        float logSumExp = logf(sumExp) + maxVal;

        // CE = -logits[label] + log_sum_exp
        loss[b] = -row[label] + logSumExp;
    }
}

// Cross-entropy gradient: grad[c] = softmax(logits)[c] - one_hot(label)[c]
__global__ void cross_entropy_grad(const float *logits, const int *labels,
                                   float *grad, int batch, int num_classes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int b = idx / num_classes;
    int c = idx % num_classes;

    if (b < batch) {
        const float *row = logits + b * num_classes;

        // stable softmax
        float maxVal = row[0];
        for (int k = 1; k < num_classes; ++k)
            maxVal = fmaxf(maxVal, row[k]);

        float sumExp = 0.0f;
        for (int k = 0; k < num_classes; ++k)
            sumExp += expf(row[k] - maxVal);

        float softmax_c = expf(row[c] - maxVal) / sumExp;
        float one_hot_c = (c == labels[b]) ? 1.0f : 0.0f;

        grad[b * num_classes + c] = softmax_c - one_hot_c;
    }
}

/* ===================================================================
 * Part C — Fused Adam Optimizer Kernel
 * =================================================================== */

__global__ void adam_fused(float *param, const float *grad,
                           float *m, float *v,
                           float beta1, float beta2,
                           float lr, float eps,
                           int t, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad[i];
        // update biased first and second moment estimates
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        // bias-corrected estimates
        float m_hat = m[i] / (1.0f - powf(beta1, (float)t));
        float v_hat = v[i] / (1.0f - powf(beta2, (float)t));

        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

/* ===================================================================
 * CPU reference helpers
 * =================================================================== */

static void cpu_sigmoid(const float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) y[i] = 1.0f / (1.0f + expf(-x[i]));
}

static void cpu_tanh(const float *x, float *y, int n) {
    for (int i = 0; i < n; ++i) y[i] = tanhf(x[i]);
}

static float maxAbsError(const float *a, const float *b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > mx) mx = diff;
    }
    return mx;
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

/* ===================================================================
 * main — run all benchmarks and correctness tests
 * =================================================================== */
int main() {
    int n = 10000000;                // 10^7 as specified
    size_t bytes = (size_t)n * sizeof(float);

    // allocate host input
    float *h_x = (float *)malloc(bytes);
    srand(12345);
    for (int i = 0; i < n; ++i)
        h_x[i] = ((float)rand() / RAND_MAX) * 8.0f - 4.0f;   // range [-4, 4]

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int numBlocks = (n + BLOCK - 1) / BLOCK;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* ---------- Part A: activation benchmarks ---------- */
    printf("========== Part A: Activation Kernels (N = 10^7) ==========\n");
    printf("%-16s  %10s  %10s  %12s\n", "Kernel", "Time (ms)", "GB/s", "Max|err|");

    // helper lambda-like macro for benchmarking
    float *h_y_gpu = (float *)malloc(bytes);
    float *h_y_cpu = (float *)malloc(bytes);

    // sigmoid
    CUDA_CHECK(cudaEventRecord(start));
    sigmoid_fwd<<<numBlocks, BLOCK>>>(d_x, d_y, n);
    CUDA_CHECK(cudaEventRecord(stop));
    float ms = timeKernel(start, stop);
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, bytes, cudaMemcpyDeviceToHost));
    cpu_sigmoid(h_x, h_y_cpu, n);
    float err = maxAbsError(h_y_gpu, h_y_cpu, n);
    double gbps = (2.0 * bytes / 1e9) / (ms / 1e3);   // read + write
    printf("%-16s  %10.3f  %10.2f  %12.2e  %s\n",
           "sigmoid", ms, gbps, err, err < 1e-4 ? "PASS" : "FAIL");

    // tanh
    CUDA_CHECK(cudaEventRecord(start));
    tanh_fwd<<<numBlocks, BLOCK>>>(d_x, d_y, n);
    CUDA_CHECK(cudaEventRecord(stop));
    ms = timeKernel(start, stop);
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, bytes, cudaMemcpyDeviceToHost));
    cpu_tanh(h_x, h_y_cpu, n);
    err = maxAbsError(h_y_gpu, h_y_cpu, n);
    gbps = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("%-16s  %10.3f  %10.2f  %12.2e  %s\n",
           "tanh", ms, gbps, err, err < 1e-4 ? "PASS" : "FAIL");

    // leaky relu (alpha = 0.01)
    CUDA_CHECK(cudaEventRecord(start));
    leaky_relu_fwd<<<numBlocks, BLOCK>>>(d_x, d_y, 0.01f, n);
    CUDA_CHECK(cudaEventRecord(stop));
    ms = timeKernel(start, stop);
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y, bytes, cudaMemcpyDeviceToHost));
    // cpu leaky relu check
    for (int i = 0; i < n; ++i)
        h_y_cpu[i] = (h_x[i] > 0.0f) ? h_x[i] : 0.01f * h_x[i];
    err = maxAbsError(h_y_gpu, h_y_cpu, n);
    gbps = (2.0 * bytes / 1e9) / (ms / 1e3);
    printf("%-16s  %10.3f  %10.2f  %12.2e  %s\n",
           "leaky_relu", ms, gbps, err, err < 1e-4 ? "PASS" : "FAIL");

    // relu backward
    // use d_y as grad_out (filled from leaky relu output)
    float *d_grad_in;
    CUDA_CHECK(cudaMalloc(&d_grad_in, bytes));
    CUDA_CHECK(cudaEventRecord(start));
    relu_backward<<<numBlocks, BLOCK>>>(d_y, d_x, d_grad_in, n);
    CUDA_CHECK(cudaEventRecord(stop));
    ms = timeKernel(start, stop);
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_grad_in, bytes, cudaMemcpyDeviceToHost));
    // cpu relu backward check
    for (int i = 0; i < n; ++i)
        h_y_cpu[i] = (h_x[i] > 0.0f) ? h_y_gpu[i] : 0.0f;  // should match since grad_out*1 or 0
    // recalculate properly:
    float *h_grad_out = (float *)malloc(bytes);
    CUDA_CHECK(cudaMemcpy(h_grad_out, d_y, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i)
        h_y_cpu[i] = (h_x[i] > 0.0f) ? h_grad_out[i] : 0.0f;
    err = maxAbsError(h_y_gpu, h_y_cpu, n);
    gbps = (3.0 * bytes / 1e9) / (ms / 1e3);   // reads grad_out + x, writes grad_in
    printf("%-16s  %10.3f  %10.2f  %12.2e  %s\n",
           "relu_backward", ms, gbps, err, err < 1e-4 ? "PASS" : "FAIL");

    /* ---------- Part B: loss functions ---------- */
    printf("\n========== Part B: Loss Functions ==========\n");

    // cross-entropy test with small example
    int batch = 1024;
    int nclass = 10;
    int total = batch * nclass;
    size_t logit_bytes = total * sizeof(float);

    float *h_logits = (float *)malloc(logit_bytes);
    int   *h_labels = (int *)malloc(batch * sizeof(int));
    float *h_loss   = (float *)malloc(batch * sizeof(float));
    for (int i = 0; i < total; ++i)
        h_logits[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    for (int b = 0; b < batch; ++b)
        h_labels[b] = rand() % nclass;

    float *d_logits, *d_loss, *d_grad;
    int   *d_labels;
    CUDA_CHECK(cudaMalloc(&d_logits, logit_bytes));
    CUDA_CHECK(cudaMalloc(&d_labels, batch * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_loss,   batch * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad,   logit_bytes));
    CUDA_CHECK(cudaMemcpy(d_logits, h_logits, logit_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_labels, h_labels, batch * sizeof(int), cudaMemcpyHostToDevice));

    // run CE loss kernel
    int nb_ce = (batch + BLOCK - 1) / BLOCK;
    cross_entropy_loss<<<nb_ce, BLOCK>>>(d_logits, d_labels, d_loss, batch, nclass);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_loss, d_loss, batch * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference cross-entropy
    float *h_loss_ref = (float *)malloc(batch * sizeof(float));
    for (int b = 0; b < batch; ++b) {
        const float *row = h_logits + b * nclass;
        float mx = row[0];
        for (int c = 1; c < nclass; ++c) mx = fmaxf(mx, row[c]);
        float se = 0.0f;
        for (int c = 0; c < nclass; ++c) se += expf(row[c] - mx);
        h_loss_ref[b] = -(row[h_labels[b]] - mx - logf(se));
    }
    float ce_err = maxAbsError(h_loss, h_loss_ref, batch);
    printf("Cross-entropy loss max|err| = %.2e  ->  %s\n",
           ce_err, ce_err < 1e-4 ? "PASS" : "FAIL");

    // run CE gradient kernel
    int nb_grad = (total + BLOCK - 1) / BLOCK;
    cross_entropy_grad<<<nb_grad, BLOCK>>>(d_logits, d_labels, d_grad, batch, nclass);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_grad_gpu = (float *)malloc(logit_bytes);
    CUDA_CHECK(cudaMemcpy(h_grad_gpu, d_grad, logit_bytes, cudaMemcpyDeviceToHost));

    // CPU reference gradient
    float *h_grad_ref = (float *)malloc(logit_bytes);
    for (int b = 0; b < batch; ++b) {
        const float *row = h_logits + b * nclass;
        float mx = row[0];
        for (int c = 1; c < nclass; ++c) mx = fmaxf(mx, row[c]);
        float se = 0.0f;
        for (int c = 0; c < nclass; ++c) se += expf(row[c] - mx);
        for (int c = 0; c < nclass; ++c) {
            float sm = expf(row[c] - mx) / se;
            float oh = (c == h_labels[b]) ? 1.0f : 0.0f;
            h_grad_ref[b * nclass + c] = sm - oh;
        }
    }
    float grad_err = maxAbsError(h_grad_gpu, h_grad_ref, total);
    printf("CE gradient max|err|       = %.2e  ->  %s\n",
           grad_err, grad_err < 1e-4 ? "PASS" : "FAIL");

    /* ---------- Part C: Adam optimizer ---------- */
    printf("\n========== Part C: Fused Adam Optimizer ==========\n");

    int nparam = 4096;
    size_t pbytes = nparam * sizeof(float);

    float *h_param     = (float *)malloc(pbytes);
    float *h_grad_adam  = (float *)malloc(pbytes);
    float *h_m         = (float *)calloc(nparam, sizeof(float));
    float *h_v         = (float *)calloc(nparam, sizeof(float));
    float *h_param_ref = (float *)malloc(pbytes);
    float *h_m_ref     = (float *)calloc(nparam, sizeof(float));
    float *h_v_ref     = (float *)calloc(nparam, sizeof(float));

    for (int i = 0; i < nparam; ++i) {
        h_param[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        h_param_ref[i] = h_param[i];
        h_grad_adam[i] = ((float)rand() / RAND_MAX) * 0.1f - 0.05f;
    }

    float beta1 = 0.9f, beta2 = 0.999f, lr = 1e-3f, eps = 1e-8f;

    float *d_param, *d_grad_adam, *d_m, *d_v;
    CUDA_CHECK(cudaMalloc(&d_param,     pbytes));
    CUDA_CHECK(cudaMalloc(&d_grad_adam,  pbytes));
    CUDA_CHECK(cudaMalloc(&d_m,         pbytes));
    CUDA_CHECK(cudaMalloc(&d_v,         pbytes));
    CUDA_CHECK(cudaMemcpy(d_param,    h_param,    pbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grad_adam, h_grad_adam, pbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_m, 0, pbytes));
    CUDA_CHECK(cudaMemset(d_v, 0, pbytes));

    int nb_adam = (nparam + BLOCK - 1) / BLOCK;

    // run 100 steps of Adam on GPU and CPU
    for (int t = 1; t <= 100; ++t) {
        adam_fused<<<nb_adam, BLOCK>>>(d_param, d_grad_adam, d_m, d_v,
                                      beta1, beta2, lr, eps, t, nparam);

        // CPU reference Adam step
        for (int i = 0; i < nparam; ++i) {
            float g = h_grad_adam[i];
            h_m_ref[i] = beta1 * h_m_ref[i] + (1.0f - beta1) * g;
            h_v_ref[i] = beta2 * h_v_ref[i] + (1.0f - beta2) * g * g;
            float mh = h_m_ref[i] / (1.0f - powf(beta1, (float)t));
            float vh = h_v_ref[i] / (1.0f - powf(beta2, (float)t));
            h_param_ref[i] -= lr * mh / (sqrtf(vh) + eps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_param, d_param, pbytes, cudaMemcpyDeviceToHost));

    float adam_err = maxAbsError(h_param, h_param_ref, nparam);
    printf("After 100 Adam steps, max|err| = %.2e  ->  %s\n",
           adam_err, adam_err < 1e-3 ? "PASS" : "FAIL");

    /* cleanup */
    free(h_x); free(h_y_gpu); free(h_y_cpu); free(h_grad_out);
    free(h_logits); free(h_labels); free(h_loss); free(h_loss_ref);
    free(h_grad_gpu); free(h_grad_ref);
    free(h_param); free(h_grad_adam); free(h_m); free(h_v);
    free(h_param_ref); free(h_m_ref); free(h_v_ref);

    CUDA_CHECK(cudaFree(d_x)); CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_grad_in));
    CUDA_CHECK(cudaFree(d_logits)); CUDA_CHECK(cudaFree(d_labels));
    CUDA_CHECK(cudaFree(d_loss)); CUDA_CHECK(cudaFree(d_grad));
    CUDA_CHECK(cudaFree(d_param)); CUDA_CHECK(cudaFree(d_grad_adam));
    CUDA_CHECK(cudaFree(d_m)); CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nAll ex03 experiments completed.\n");
    return 0;
}
