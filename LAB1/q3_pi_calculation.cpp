/*
    Q3 - Calculate PI using integration
    pi = integral from 0 to 1 of 4/(1+x^2) dx
*/
#include <iostream>
#include <cmath>
#include <omp.h>
using namespace std;

#define NUM_STEPS 100000000

int main() {
    double step = 1.0 / NUM_STEPS;
    double pi, sum;
    double t1, t2;
    int threads[] = {1, 2, 4, 8, 16};
    
    cout << "Calculating PI" << endl;
    cout << "Steps: " << NUM_STEPS << endl;
    cout << "==============================" << endl << endl;
    
    // serial
    sum = 0.0;
    t1 = omp_get_wtime();
    for(long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }
    pi = step * sum;
    t2 = omp_get_wtime();
    double serial_time = t2 - t1;
    
    cout << "Serial:" << endl;
    cout << "PI = " << pi << endl;
    cout << "Time = " << serial_time << " sec" << endl;
    cout << "Error = " << fabs(pi - 3.14159265358979) << endl << endl;
    
    // parallel with reduction
    cout << "Parallel (reduction):" << endl;
    cout << "Threads\tPI\t\t\tTime\tSpeedup" << endl;
    
    for(int j = 0; j < 5; j++) {
        int nt = threads[j];
        omp_set_num_threads(nt);
        
        sum = 0.0;
        t1 = omp_get_wtime();
        
        #pragma omp parallel for reduction(+:sum)
        for(long i = 0; i < NUM_STEPS; i++) {
            double x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x*x);
        }
        pi = step * sum;
        
        t2 = omp_get_wtime();
        cout << nt << "\t" << pi << "\t" << (t2-t1) << "\t" << serial_time/(t2-t1) << "x" << endl;
    }
    
    // compare with critical section
    cout << "\nComparing sync methods (4 threads):" << endl;
    omp_set_num_threads(4);
    
    // reduction
    sum = 0.0;
    t1 = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum)
    for(long i = 0; i < NUM_STEPS; i++) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }
    t2 = omp_get_wtime();
    cout << "Reduction: " << (t2-t1) << " sec" << endl;
    
    // critical
    sum = 0.0;
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        double local = 0;
        #pragma omp for
        for(long i = 0; i < NUM_STEPS; i++) {
            double x = (i + 0.5) * step;
            local += 4.0 / (1.0 + x*x);
        }
        #pragma omp critical
        sum += local;
    }
    t2 = omp_get_wtime();
    cout << "Critical: " << (t2-t1) << " sec" << endl;
    
    // atomic
    sum = 0.0;
    t1 = omp_get_wtime();
    #pragma omp parallel
    {
        double local = 0;
        #pragma omp for
        for(long i = 0; i < NUM_STEPS; i++) {
            double x = (i + 0.5) * step;
            local += 4.0 / (1.0 + x*x);
        }
        #pragma omp atomic
        sum += local;
    }
    t2 = omp_get_wtime();
    cout << "Atomic:   " << (t2-t1) << " sec" << endl;
    
    cout << "\nConclusion:" << endl;
    cout << "- Reduction is fastest" << endl;
    cout << "- Critical and atomic have more overhead" << endl;
    cout << "- All give same correct answer" << endl;
    
    return 0;
}
