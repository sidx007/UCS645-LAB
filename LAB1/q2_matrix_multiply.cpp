/*
    Q2 - Matrix Multiplication
    1000x1000 matrices
    1D threading vs 2D threading
*/
#include <iostream>
#include <omp.h>
#include <cstdlib>
using namespace std;

#define N 1000

double A[N][N], B[N][N], C[N][N];

void init_matrix() {
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            A[i][j] = rand() % 100 / 100.0;
            B[i][j] = rand() % 100 / 100.0;
            C[i][j] = 0;
        }
    }
}

void clear_C() {
    for(int i = 0; i < N; i++)
        for(int j = 0; j < N; j++)
            C[i][j] = 0;
}

int main() {
    double t1, t2;
    int threads[] = {1, 2, 4, 8, 16};
    
    srand(42);
    init_matrix();
    
    cout << "Matrix Multiplication " << N << "x" << N << endl;
    cout << "======================================" << endl << endl;
    
    // serial
    cout << "Running serial..." << endl;
    t1 = omp_get_wtime();
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            C[i][j] = 0;
            for(int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    t2 = omp_get_wtime();
    double serial = t2 - t1;
    cout << "Serial time: " << serial << " sec" << endl << endl;
    
    // 1D parallel (outer loop only)
    cout << "=== 1D Threading ===" << endl;
    cout << "Threads\tTime\t\tSpeedup" << endl;
    
    for(int j = 0; j < 5; j++) {
        int nt = threads[j];
        omp_set_num_threads(nt);
        clear_C();
        
        t1 = omp_get_wtime();
        #pragma omp parallel for
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                C[i][j] = 0;
                for(int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        t2 = omp_get_wtime();
        
        cout << nt << "\t" << (t2-t1) << "\t" << serial/(t2-t1) << "x" << endl;
    }
    
    // 2D parallel (collapse)
    cout << "\n=== 2D Threading (collapse) ===" << endl;
    cout << "Threads\tTime\t\tSpeedup" << endl;
    
    for(int j = 0; j < 5; j++) {
        int nt = threads[j];
        omp_set_num_threads(nt);
        clear_C();
        
        t1 = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                C[i][j] = 0;
                for(int k = 0; k < N; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        t2 = omp_get_wtime();
        
        cout << nt << "\t" << (t2-t1) << "\t" << serial/(t2-t1) << "x" << endl;
    }
    
    cout << "\nConclusion:" << endl;
    cout << "- 1D threading parallelizes rows" << endl;
    cout << "- 2D threading parallelizes both i and j loops" << endl;
    cout << "- 1D is usually better coz of cache locality" << endl;
    
    return 0;
}
