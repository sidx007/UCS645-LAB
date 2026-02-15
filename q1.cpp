#include <iostream>
#include <omp.h>
#include <cstdlib>

#define N 1000000

int main() {
    double *A = new double[N];
    double *B = new double[N];
    double *C = new double[N];
    for (long i = 0; i < N; i++) {
        A[i] = i * 1.0;
        B[i] = i * 2.0;
    }
    for(int threads=1;threads<12;threads++){
	    omp_set_num_threads(threads);
	    double start = omp_get_wtime();
	    #pragma omp parallel for
	    for (long i = 0; i < N; i++){
	        C[i] = A[i] + B[i];
           	double end = omp_get_wtime
	    printf("Time: %f seconds\n", end - start);
	    printf("Threads used: %d\n", omp_get_max_threads());
    }
    delete[] A;
    delete[] B;
    delete[] C;
    return 0;
}
