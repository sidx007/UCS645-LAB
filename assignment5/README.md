# UCS645 Assignment 5 - MPI Part II

## Aim
This assignment explores:
- blocking vs non-blocking MPI communication
- collective communication with MPI_Bcast and MPI_Reduce
- performance metrics such as execution time, speedup, and efficiency

## What Is Implemented
The project includes all required exercise questions (Q1 to Q5) and supporting demos.

### Required Question Programs
- src/q1_daxpy.c
- src/q2_broadcast_race.c
- src/q3_distributed_dot.c
- src/q4_primes_master_slave.c
- src/q5_perfect_master_slave.c

### Supporting Demos
- src/demo_blocking_ring.c
- src/demo_nonblocking_ring.c
- src/demo_pi_bcast_reduce.c

## Q1: DAXPY MPI vs Uniprocessor
Program: q1_daxpy

Operation:
- X[i] = a * X[i] + Y[i]

Features:
- Uses vectors of default size 2^16
- Runs sequential baseline on rank 0
- Runs distributed MPI version with scatter/gather
- Prints:
  - T1 (sequential time)
  - Tp (MPI time)
  - Speedup = T1 / Tp
  - Efficiency = Speedup / p

Run:
- make run NP=4 PROG=q1_daxpy ARGS='65536 2.5'

## Q2: Broadcast Race (MyBcast vs MPI_Bcast)
Program: q2_broadcast_race

Features:
- Allocates a massive array (default 10,000,000 doubles)
- Part A: custom linear broadcast from rank 0 using MPI_Send/MPI_Recv
- Part B: optimized MPI_Bcast
- Measures and compares timings

Run:
- make run NP=2 PROG=q2_broadcast_race ARGS='10000000'
- make run NP=4 PROG=q2_broadcast_race ARGS='10000000'
- make run NP=8 PROG=q2_broadcast_race ARGS='10000000'
- make run NP=16 PROG=q2_broadcast_race ARGS='10000000'

## Q3: Distributed Dot Product and Amdahl Context
Program: q3_distributed_dot

Features:
- Global size default: 500,000,000
- Rank 0 broadcasts multiplier to all ranks
- Each rank generates and computes local chunk contribution
- MPI_Reduce with MPI_SUM creates final result at rank 0
- Reports communication time and computation time

Run:
- make run NP=1 PROG=q3_distributed_dot ARGS='500000000 1.0'
- make run NP=2 PROG=q3_distributed_dot ARGS='500000000 1.0'
- make run NP=4 PROG=q3_distributed_dot ARGS='500000000 1.0'
- make run NP=8 PROG=q3_distributed_dot ARGS='500000000 1.0'

## Q4: Master-Slave Prime Search
Program: q4_primes_master_slave

Features:
- Master uses MPI_Recv with MPI_ANY_SOURCE for dynamic scheduling
- Slaves return +n for prime and -n for non-prime
- Master sends next number with MPI_Send

Run:
- make run NP=4 PROG=q4_primes_master_slave ARGS='1000'

## Q5: Master-Slave Perfect Number Search
Program: q5_perfect_master_slave

Features:
- Same request/reply style as Q4
- Slaves return +n for perfect, -n for non-perfect
- Master aggregates and prints all perfect numbers up to max value

Run:
- make run NP=4 PROG=q5_perfect_master_slave ARGS='10000'

## Supporting Demos
1. Blocking ring transfer (can deadlock for very large messages):
- make run NP=4 PROG=demo_blocking_ring ARGS='100'

2. Non-blocking ring transfer with overlap:
- make run NP=4 PROG=demo_nonblocking_ring ARGS='10000000'

3. Parallel Pi with broadcast and reduction:
- make run NP=4 PROG=demo_pi_bcast_reduce ARGS='1000000'

## Build Instructions
### Linux
1. Install MPI (example MPICH):
   - sudo apt-get update
   - sudo apt-get install mpich
2. Build:
   - make

### Windows
Use an MPI installation that provides both mpicc and mpirun in PATH.
Then build with:
- mingw32-make

## Makefile Targets
- make: build all targets
- make required: build only Q1 to Q5
- make demos: build only demo programs
- make run NP=<p> PROG=<target> ARGS='<args>'
- make clean

## Suggested Performance Table
For Q1 and Q3, record values for p = 1, 2, 4, 8:

- Processes p
- Time Tp
- Speedup Sp = T1 / Tp
- Efficiency Ep = Sp / p

For Q2, compare:
- MyBcast time vs MPI_Bcast time as p increases

## Notes
- Q4 and Q5 require at least 2 processes.
- Very large message sizes can require high memory.
- Blocking ring demo intentionally represents deadlock-prone ordering for large transfers.
