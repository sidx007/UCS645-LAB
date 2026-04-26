

---

# Assignment 7: CUDA Part II



---

## Problem 1: Sum of First N Integers Using CUDA

### Objective

* Compute sum of first **N = 1024 integers**
* Two approaches:

  * Iterative (no formula)
  * Direct formula

---

### Approach

#### Steps:

* Define `N = 1024`
* Create input/output arrays
* Allocate GPU memory using `cudaMalloc`
* Initialize array with values `1 → N`
* Copy data from host → device
* Define grid and block size
* Launch kernel
* Copy result back to host

---

### Kernel 1: Iterative Sum

```cpp
__global__ void iterativeSum(int *arr, int *result, int N) {
    int sum = 0;
    for(int i = 0; i < N; i++) {
        sum += arr[i];
    }
    *result = sum;
}
```

---

### Kernel 2: Direct Formula

```cpp
__global__ void formulaSum(int *result, int N) {
    *result = (N * (N + 1)) / 2;
}
```

---

### Key Insight

* Iterative → O(N)
* Formula → O(1)
* GPU is overkill for formula (but assignment wants demonstration)

---

## Problem 2: Merge Sort

### (a) Parallel Merge Sort using Pipelining

#### Idea:

* Divide array into chunks
* Process chunks in stages (pipeline)
* Overlap computation

#### Steps:

* Split array
* Sort chunks (CPU threads or staged execution)
* Merge progressively

---

### (b) CUDA Parallel Merge Sort

#### Approach:

* Each thread handles merging
* Use shared memory for faster access
* Perform bottom-up merge

```cpp
__global__ void mergeKernel(int *arr, int *temp, int width, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int left = tid * 2 * width;
    int mid = min(left + width, N);
    int right = min(left + 2 * width, N);

    int i = left, j = mid, k = left;

    while(i < mid && j < right) {
        temp[k++] = (arr[i] < arr[j]) ? arr[i++] : arr[j++];
    }

    while(i < mid) temp[k++] = arr[i++];
    while(j < right) temp[k++] = arr[j++];
}
```

---

### (c) Performance Comparison

| Method     | Speed  | Parallelism | Complexity |
| ---------- | ------ | ----------- | ---------- |
| Pipelining | Medium | Limited     | Moderate   |
| CUDA       | High   | Massive     | Complex    |

---

### Key Insight

* CUDA wins for large N due to **massive parallelism**
* Pipeline useful when GPU not available

---

## Problem 3: Vector Addition + Profiling

---

### Basic Kernel

```cpp
__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        C[i] = A[i] + B[i];
    }
}
```

---

### 1.1 Static Global Memory

```cpp
__device__ float A[1024], B[1024], C[1024];
```

⚠️ Important:

* Do NOT pass device symbols directly to kernels

---

### 1.2 Kernel Timing

```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);

vectorAdd<<<grid, block>>>(A, B, C, N);

cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms;
cudaEventElapsedTime(&ms, start, stop);
printf("Kernel Time: %f ms\n", ms);
```

---

### 1.3 Theoretical Bandwidth

Formula:

```
theoreticalBW = memoryClockRate × memoryBusWidth × 2
```

Conversion:

* kHz → Hz
* bits → bytes
* → GB/s

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

double bw = 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0);
bw *= 1e3; // kHz → Hz
bw /= 1e9; // → GB/s

printf("Theoretical BW: %f GB/s\n", bw);
```

---

### 1.4 Measured Bandwidth

Formula:

```
measuredBW = (RBytes + WBytes) / t
```

For vector add:

* Reads: A + B → 2N
* Writes: C → N
* Total bytes = 3N × sizeof(float)

```cpp
double bytes = 3 * N * sizeof(float);
double seconds = ms / 1000.0;

double measuredBW = bytes / seconds / 1e9;

printf("Measured BW: %f GB/s\n", measuredBW);
```

---

### Profiling with nvprof

```bash
nvprof ./a.out
```

---

## Key Takeaways

* GPU memory handling = critical skill
* Kernel launch config impacts performance heavily
* Theoretical BW ≠ Measured BW (real-world bottlenecks)
* CUDA shines only when parallel workload is large

---


