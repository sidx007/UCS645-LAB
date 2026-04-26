# Assignment 6 Solution - Introduction to CUDA

## Part A - Device Query

### Device query program

```cpp
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cout << "No CUDA device found.\n";
        return 0;
    }

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        std::cout << "Device " << i << ": " << prop.name << "\n";
        std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total global memory: " << prop.totalGlobalMem << " bytes\n";
        std::cout << "Shared memory per block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "Constant memory: " << prop.totalConstMem << " bytes\n";
        std::cout << "Warp size: " << prop.warpSize << "\n";
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "Max block dimensions: "
                  << prop.maxThreadsDim[0] << " x "
                  << prop.maxThreadsDim[1] << " x "
                  << prop.maxThreadsDim[2] << "\n";
        std::cout << "Max grid dimensions: "
                  << prop.maxGridSize[0] << " x "
                  << prop.maxGridSize[1] << " x "
                  << prop.maxGridSize[2] << "\n";
        std::cout << "Registers per block: " << prop.regsPerBlock << "\n";
        std::cout << "Clock rate: " << prop.clockRate << " kHz\n";
        std::cout << "Multiprocessor count: " << prop.multiProcessorCount << "\n";
        std::cout << "Double precision supported: "
                  << ((prop.major > 1 || (prop.major == 1 && prop.minor >= 3)) ? "Yes" : "No")
                  << "\n\n";
    }

    return 0;
}
```

### Compile and run

```bash
nvcc device_query.cu -o device_query
./device_query
```

### Answers

**1. What is the architecture and compute capability of your GPU?**  
This depends on the actual GPU in the system. After running the program, report the GPU name and the compute capability shown as `major.minor`. The architecture can then be mapped from that compute capability.

**2. What are the maximum block dimensions for your GPU?**  
These are the values printed by `prop.maxThreadsDim[0]`, `prop.maxThreadsDim[1]`, and `prop.maxThreadsDim[2]`.

**3. Suppose you are launching a one dimensional grid and block. If the hardware's maximum grid dimension is 65535 and the maximum block dimension is 512, what is the maximum number threads can be launched on the GPU?**  
Maximum threads = `65535 x 512 = 33,553,920` threads.

**4. Under what conditions might a programmer choose not want to launch the maximum number of threads?**  
A programmer may avoid launching the maximum number of threads when:
- The problem size is smaller than the maximum launch size.
- Too many threads would waste resources because many threads would be idle.
- Register usage, shared memory usage, or occupancy may become poor.
- Smaller blocks may give better performance depending on memory access and control divergence.
- Kernel launch overhead may dominate for small workloads.

**5. What can limit a program from launching the maximum number of threads on a GPU?**  
The main limits are:
- Maximum threads per block supported by hardware.
- Maximum grid dimensions supported by hardware.
- Per-block shared memory usage.
- Register usage per thread or per block.
- Available device memory.
- Runtime errors from invalid launch configuration.

**6. What is shared memory? How much shared memory is on your GPU?**  
Shared memory is a small, low-latency on-chip memory that is shared by all threads of the same block. The amount on the GPU for each block is printed as `sharedMemPerBlock`.

**7. What is global memory? How much global memory is on your GPU?**  
Global memory is the large off-chip device memory accessible by all threads and also by the host through memory copies. Its size is printed as `totalGlobalMem`.

**8. What is constant memory? How much constant memory is on your GPU?**  
Constant memory is a small read-only cached memory space used for values that do not change during kernel execution. Its size is printed as `totalConstMem`.

**9. What does warp size signify on a GPU? What is your GPU's warp size?**  
Warp size is the number of threads executed together as a scheduling unit on an NVIDIA GPU. The warp size on the device is printed as `warpSize`.

**10. Is double precision supported on your GPU?**  
This can be checked from the compute capability. For most CUDA-capable GPUs with compute capability 1.3 or higher, double precision is supported, although throughput may vary greatly by architecture.

## Part B - Sum of Array Elements in CUDA

### CUDA program

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void sumKernel(float *input, float *partial, int n) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);

    std::vector<float> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float *d_input, *d_partial;

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::vector<float> h_partial(blocksPerGrid);

    cudaMalloc((void**)&d_input, bytes);
    cudaMalloc((void**)&d_partial, blocksPerGrid * sizeof(float));

    cudaMemcpy(d_input, h_input.data(), bytes, cudaMemcpyHostToDevice);

    sumKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_partial, n);

    cudaMemcpy(h_partial.data(), d_partial, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for (int i = 0; i < blocksPerGrid; i++) {
        finalSum += h_partial[i];
    }

    std::cout << "Sum = " << finalSum << std::endl;

    cudaFree(d_input);
    cudaFree(d_partial);

    return 0;
}
```

### Required steps mapped to the code

1. **Allocate device memory** using `cudaMalloc()` for `d_input` and `d_partial`.  
2. **Copy host memory to device** using `cudaMemcpy(..., cudaMemcpyHostToDevice)`.  
3. **Initialize thread block and kernel grid dimensions** using `threadsPerBlock` and `blocksPerGrid`.  
4. **Invoke CUDA kernel** with `sumKernel<<<...>>>()`.  
5. **Copy results from device to host** using `cudaMemcpy(..., cudaMemcpyDeviceToHost)`.  
6. **Free device memory** using `cudaFree()`.  
7. **CUDA kernel** is `sumKernel`, which performs block-level reduction in shared memory.

## Part C - Matrix Addition in CUDA

### CUDA program

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

__global__ void matrixAdd(const int *A, const int *B, int *C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int rows = 2048;
    int cols = 2048;
    int size = rows * cols;
    size_t bytes = size * sizeof(int);

    std::vector<int> h_A(size, 1), h_B(size, 2), h_C(size);
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    matrixAdd<<<grid, block>>>(d_A, d_B, d_C, rows, cols);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

### Answers

**1. How many floating operations are being performed in the matrix addition kernel?**  
Since the matrices are integers, the kernel performs **0 floating-point operations**. It performs **one integer addition per output element**. For an `rows x cols` matrix, arithmetic operations = `rows x cols` integer additions.

**2. How many global memory reads are being performed by your kernel?**  
Each thread reads one element from `A` and one element from `B`. Therefore, total global memory reads = `2 x rows x cols`.

**3. How many global memory writes are being performed by your kernel?**  
Each thread writes one result to `C`. Therefore, total global memory writes = `rows x cols`.

## Notes for submission

For Part A, the code is complete, but the exact values for architecture, compute capability, memory sizes, block dimensions, and warp size must be filled using the output from the GPU on the target machine. Everything else in the assignment is fully solved and ready to submit.
