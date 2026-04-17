# UCS645 Assignment 3 - Correlation with OpenMP

## Objective
This project computes pairwise correlation coefficients between all rows of an input matrix.

Given an input matrix of size ny x nx, the implemented function computes:
- result[i + j * ny] for all 0 <= j <= i < ny
- where each value is the correlation between row i and row j

## Required Interface
The following interface is implemented:

```cpp
void correlate(int ny, int nx, const float* data, float* result);
```

An additional parallel implementation is also included:

```cpp
void correlate_parallel(int ny, int nx, const float* data, float* result, int num_threads);
```

## Files
- main.cpp: command-line driver, benchmarking, correctness checks
- corelate.cpp: sequential and OpenMP correlation implementations
- correlate.h: function declarations
- Makefile: build, run, and perf targets
- required_outputs.txt: sample benchmark outputs

## Implementation Summary
1. Sequential baseline (correlate)
- Uses double-precision arithmetic for all intermediate computations.
- Normalizes each row (zero mean, unit norm).
- Computes lower-triangular correlation matrix using dot products.

2. Parallel version (correlate_parallel)
- Parallel row normalization using OpenMP.
- Parallelized outer loop across row pairs.
- SIMD reduction on inner dot-product loop for better CPU utilization.

3. Correctness
- main.cpp compares sequential and parallel outputs.
- Reports maximum absolute difference (MaxAbsDiff).

## Build
### Linux
```bash
cd demo1
make
```

### Windows (MinGW)
```powershell
cd demo1
mingw32-make
```

## Run
Program format:

```text
./correlate_lab <ny> <nx> [max_threads] [mode]
```

- ny: number of input vectors (rows)
- nx: number of elements in each vector (columns)
- max_threads: optional, default is OpenMP maximum
- mode:
  - all: run sequential + thread sweep for parallel
  - seq: run only sequential baseline
  - par: run only parallel mode with max_threads

Examples:

```bash
./correlate_lab 512 512 8 all
./correlate_lab 1024 1024 1 seq
./correlate_lab 1024 1024 8 par
```

## Makefile Targets
- make: build executable
- make run NY=<rows> NX=<cols> THREADS=<t> MODE=<all|seq|par>
- make clean: remove build artifacts
- make perf-seq NY=<rows> NX=<cols>
- make perf-par NY=<rows> NX=<cols> THREADS=<t>

## perf Stats (Linux)
Use perf to evaluate sequential and parallel runs:

```bash
make perf-seq NY=512 NX=512
make perf-par NY=512 NX=512 THREADS=8
```

## Sample Results
See required_outputs.txt for captured outputs.

Measured sample runs include:
- ny=256, nx=256, threads up to 8
- ny=512, nx=512, threads up to 8

Both runs show:
- clear speedup with increasing threads
- MaxAbsDiff = 0.000 (sequential and parallel outputs match)

## Notes
- OpenMP is enabled through -fopenmp in the Makefile.
- The current Makefile uses optimization flags suitable for CPU benchmarking.
