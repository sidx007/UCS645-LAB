#ifndef CORRELATE_H
#define CORRELATE_H

// Sequential baseline using double-precision arithmetic.
void correlate(int ny, int nx, const float* data, float* result);

// OpenMP parallel implementation with configurable thread count.
void correlate_parallel(int ny, int nx, const float* data, float* result, int num_threads);

#endif
