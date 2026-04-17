# UCS645 Assignment 4 - Introduction to MPI

## Overview
This folder contains a complete Assignment 4 MPI lab implementation, including:
- core MPI examples from the handout
- all four required laboratory exercises
- a performance measurement program using MPI_Wtime
- a Makefile for easy build and run workflows

## Folder Structure
- src/hello_mpi.c
- src/send_recv.c
- src/broadcast.c
- src/reduce_sum.c
- src/matrix_mult.c
- src/ring_comm.c
- src/array_sum.c
- src/max_min.c
- src/dot_product.c
- src/perf_metrics.c
- Makefile

## Programs and Purpose
1. hello_mpi
- Basic MPI initialization and process rank identification.

2. send_recv
- Point-to-point communication using MPI_Send and MPI_Recv.

3. broadcast
- Collective broadcast using MPI_Bcast.

4. reduce_sum
- Collective reduction using MPI_Reduce with MPI_SUM.

5. matrix_mult
- Parallel matrix-vector multiplication using scatter, broadcast, and gather.
- Supports matrix size from command line.

6. ring_comm (Exercise 1)
- Ring communication across all processes.
- Starts with value 100 at process 0 and returns final value to process 0.

7. array_sum (Exercise 2)
- Parallel sum of array values using MPI_Scatterv and MPI_Reduce.
- Default array size is 100 and supports command-line size.

8. max_min (Exercise 3)
- Each process generates random numbers and computes local min/max.
- Uses MPI_MAXLOC and MPI_MINLOC to report value and process rank.

9. dot_product (Exercise 4)
- Parallel dot product for:
  - A = [1,2,3,4,5,6,7,8]
  - B = [8,7,6,5,4,3,2,1]
- Expected result is 120.

10. perf_metrics
- Measures execution time, communication time, and computation time.
- Reports communication and computation percentages.

## Build Instructions
### Linux
1. Install MPI (example with MPICH):
   - sudo apt-get update
   - sudo apt-get install mpich
2. Build all programs:
   - make

### Windows
Use an MPI setup that provides mpicc and mpirun in PATH. Then run:
- mingw32-make

## Run Instructions
General run command from this folder:
- make run NP=4 PROG=hello_mpi

Optional argument passing:
- make run NP=4 PROG=matrix_mult ARGS=8
- make run NP=4 PROG=array_sum ARGS=100
- make run NP=8 PROG=perf_metrics ARGS=1024

Run all reference examples:
- make run-examples

Run all exercises:
- make run-exercises

## Expected Checkpoints
- ring_comm: process 0 receives the final accumulated value.
- array_sum with default size 100: global sum should be 5050.
- dot_product: result should be 120.
- perf_metrics: report includes Tp, communication time, and computation time.

## Performance Analysis Guide
For Assignment 4 analysis, collect runs for p = 1, 2, 4, 8:
- mpirun -np 1 bin/perf_metrics 1024
- mpirun -np 2 bin/perf_metrics 1024
- mpirun -np 4 bin/perf_metrics 1024
- mpirun -np 8 bin/perf_metrics 1024

Compute the metrics:
- Speedup Sp = T1 / Tp
- Efficiency Ep = Sp / p

Suggested report table columns:
- Processes (p)
- Time Tp (s)
- Speedup Sp
- Efficiency Ep
- Communication percent

## Clean
- make clean

## Notes
- Some MPI runtimes require explicit permission to run as root inside containers.
- For best reproducibility, keep process count and input size fixed during comparisons.
