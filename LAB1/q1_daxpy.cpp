/*
    Q1 - DAXPY Loop
    X[i] = a*X[i] + Y[i]
    comparing speedup with different threads
*/
#include <iostream>
#include <omp.h>
using namespace std;

#define N 65536  // 2^16

double X[N], Y[N];

void init() {
    for(int i = 0; i < N; i++) {
        X[i] = i + 1;
        Y[i] = i + 2;
    }
}

int main() {
    double a = 2.5;
    double t1, t2;
    int threads[] = {1, 2, 4, 8, 16};
    
    cout << "DAXPY: X[i] = a*X[i] + Y[i]" << endl;
    cout << "Size = " << N << endl << endl;
    
    // serial version
    init();
    t1 = omp_get_wtime();
    for(int k = 0; k < 1000; k++) {
        init();
        for(int i = 0; i < N; i++) {
            X[i] = a * X[i] + Y[i];
        }
    }
    t2 = omp_get_wtime();
    double serial_time = t2 - t1;
    cout << "Serial time: " << serial_time << " sec" << endl << endl;
    
    // parallel versions
    cout << "Threads\tTime\t\tSpeedup" << endl;
    cout << "--------------------------------" << endl;
    
    double best = 0;
    int best_t = 1;
    
    for(int j = 0; j < 5; j++) {
        int nthreads = threads[j];
        omp_set_num_threads(nthreads);
        
        init();
        t1 = omp_get_wtime();
        for(int k = 0; k < 1000; k++) {
            init();
            #pragma omp parallel for
            for(int i = 0; i < N; i++) {
                X[i] = a * X[i] + Y[i];
            }
        }
        t2 = omp_get_wtime();
        
        double ptime = t2 - t1;
        double speedup = serial_time / ptime;
        
        cout << nthreads << "\t" << ptime << "\t" << speedup << "x" << endl;
        
        if(speedup > best) {
            best = speedup;
            best_t = nthreads;
        }
    }
    
    cout << "\nBest speedup: " << best << "x with " << best_t << " threads" << endl;
    cout << "\nObservations:" << endl;
    cout << "- Speedup increases upto " << best_t << " threads" << endl;
    cout << "- After that overhead increases and speedup drops" << endl;
    cout << "- DAXPY is memory bound so cant get linear speedup" << endl;
    
    return 0;
}
