#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <numeric>

using namespace std;
using namespace std::chrono;


const int    NX          = 1000;    
const int    NY          = 1000;    
const int    TIME_STEPS  = 500;     
const double ALPHA       = 0.01;   
const double DX          = 1.0;    
const double DY          = 1.0;    
const double DT          = 0.25 * DX * DX / ALPHA; 


void initialize(vector<vector<double>>& T, int nx, int ny) {
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            T[i][j] = 0.0;
        }
    }
    
    int cx = nx / 2, cy = ny / 2;
    int size = nx / 10;
    for (int i = cx - size; i <= cx + size; i++) {
        for (int j = cy - size; j <= cy + size; j++) {
            if (i >= 0 && i < nx && j >= 0 && j < ny)
                T[i][j] = 100.0;
        }
    }
}




double heat_diffusion_serial(vector<vector<double>>& T,
                              vector<vector<double>>& T_new,
                              int nx, int ny, int steps) {
    double rx = ALPHA * DT / (DX * DX);
    double ry = ALPHA * DT / (DY * DY);
    double total_heat = 0.0;

    for (int t = 0; t < steps; t++) {
        for (int i = 1; i < nx - 1; i++) {
            for (int j = 1; j < ny - 1; j++) {
                T_new[i][j] = T[i][j]
                    + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                    + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
            }
        }
        swap(T, T_new);
    }

    
    total_heat = 0.0;
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            total_heat += T[i][j];

    return total_heat;
}




double heat_diffusion_parallel(vector<vector<double>>& T,
                                vector<vector<double>>& T_new,
                                int nx, int ny, int steps,
                                const string& sched) {
    double rx = ALPHA * DT / (DX * DX);
    double ry = ALPHA * DT / (DY * DY);
    double total_heat = 0.0;

    for (int t = 0; t < steps; t++) {
        if (sched == "static") {
            #pragma omp parallel for schedule(static)
            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                        + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                }
            }
        } else if (sched == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 16)
            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                        + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                }
            }
        } else { 
            #pragma omp parallel for schedule(guided)
            for (int i = 1; i < nx - 1; i++) {
                for (int j = 1; j < ny - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                        + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                }
            }
        }
        swap(T, T_new);
    }

    
    total_heat = 0.0;
    #pragma omp parallel for reduction(+:total_heat)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            total_heat += T[i][j];

    return total_heat;
}




double heat_diffusion_tiled(vector<vector<double>>& T,
                             vector<vector<double>>& T_new,
                             int nx, int ny, int steps,
                             int tile_size) {
    double rx = ALPHA * DT / (DX * DX);
    double ry = ALPHA * DT / (DY * DY);
    double total_heat = 0.0;

    for (int t = 0; t < steps; t++) {
        #pragma omp parallel for schedule(dynamic, 1) collapse(2)
        for (int ti = 1; ti < nx - 1; ti += tile_size) {
            for (int tj = 1; tj < ny - 1; tj += tile_size) {
                int i_end = min(ti + tile_size, nx - 1);
                int j_end = min(tj + tile_size, ny - 1);
                for (int i = ti; i < i_end; i++) {
                    for (int j = tj; j < j_end; j++) {
                        T_new[i][j] = T[i][j]
                            + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                            + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                    }
                }
            }
        }
        swap(T, T_new);
    }

    total_heat = 0.0;
    #pragma omp parallel for reduction(+:total_heat)
    for (int i = 0; i < nx; i++)
        for (int j = 0; j < ny; j++)
            total_heat += T[i][j];

    return total_heat;
}

int main() {
    int max_threads = omp_get_max_threads();

    cout << "========================================================" << endl;
    cout << "  Heat Diffusion Simulation (2D Finite Difference)" << endl;
    cout << "  Grid: " << NX << " x " << NY << endl;
    cout << "  Time steps: " << TIME_STEPS << endl;
    cout << "  Alpha: " << ALPHA << " | dt: " << DT << endl;
    cout << "  Available threads: " << max_threads << endl;
    cout << "========================================================" << endl;

    
    vector<vector<double>> T(NX, vector<double>(NY, 0.0));
    vector<vector<double>> T_new(NX, vector<double>(NY, 0.0));
    initialize(T, NX, NY);

    auto start = high_resolution_clock::now();
    double heat_serial = heat_diffusion_serial(T, T_new, NX, NY, TIME_STEPS);
    auto end = high_resolution_clock::now();
    double t_serial = duration<double>(end - start).count();

    cout << "\n--- Serial Baseline ---" << endl;
    cout << "  Time:       " << fixed << setprecision(6) << t_serial << " s" << endl;
    cout << "  Total heat: " << setprecision(4) << heat_serial << endl;

    
    cout << "\n--- Strong Scaling (static schedule) ---" << endl;
    cout << left << setw(10) << "Threads"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << setw(15) << "Efficiency"
         << "Total Heat" << endl;
    cout << string(65, '-') << endl;

    vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);
    if (thread_counts.back() != max_threads) thread_counts.push_back(max_threads);

    for (int nthreads : thread_counts) {
        omp_set_num_threads(nthreads);

        initialize(T, NX, NY);
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                T_new[i][j] = 0.0;

        start = high_resolution_clock::now();
        double heat_par = heat_diffusion_parallel(T, T_new, NX, NY, TIME_STEPS, "static");
        end = high_resolution_clock::now();
        double t_par = duration<double>(end - start).count();

        double speedup = t_serial / t_par;
        double efficiency = speedup / nthreads;

        cout << left << setw(10) << nthreads
             << setw(15) << fixed << setprecision(6) << t_par
             << setw(12) << setprecision(2) << speedup << "x"
             << setw(14) << setprecision(2) << (efficiency * 100) << "%"
             << setprecision(4) << heat_par << endl;
    }

    
    cout << "\n--- Scheduling Strategy Comparison (" << max_threads << " threads) ---" << endl;
    cout << left << setw(15) << "Schedule"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << "Total Heat" << endl;
    cout << string(55, '-') << endl;

    omp_set_num_threads(max_threads);

    const char* schedules[] = {"static", "dynamic", "guided"};
    for (const char* sched : schedules) {
        initialize(T, NX, NY);
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                T_new[i][j] = 0.0;

        start = high_resolution_clock::now();
        double h = heat_diffusion_parallel(T, T_new, NX, NY, TIME_STEPS, sched);
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();

        cout << left << setw(15) << sched
             << setw(15) << fixed << setprecision(6) << t
             << setw(12) << setprecision(2) << (t_serial / t) << "x"
             << setprecision(4) << h << endl;
    }

    
    cout << "\n--- Cache Tiling Comparison (" << max_threads << " threads) ---" << endl;
    cout << left << setw(15) << "Tile Size"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << "Total Heat" << endl;
    cout << string(55, '-') << endl;

    int tile_sizes[] = {16, 32, 64, 128, 256};
    for (int ts : tile_sizes) {
        initialize(T, NX, NY);
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                T_new[i][j] = 0.0;

        start = high_resolution_clock::now();
        double h = heat_diffusion_tiled(T, T_new, NX, NY, TIME_STEPS, ts);
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();

        cout << left << setw(15) << ts
             << setw(15) << fixed << setprecision(6) << t
             << setw(12) << setprecision(2) << (t_serial / t) << "x"
             << setprecision(4) << h << endl;
    }

    
    cout << "\n--- Per-Thread Timing (Load Imbalance, " << max_threads << " threads) ---" << endl;
    {
        omp_set_num_threads(max_threads);
        initialize(T, NX, NY);
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                T_new[i][j] = 0.0;

        double rx = ALPHA * DT / (DX * DX);
        double ry = ALPHA * DT / (DY * DY);
        vector<double> thread_times(max_threads, 0.0);

        double wall_start = omp_get_wtime();

        for (int t = 0; t < TIME_STEPS; t++) {
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                double t0 = omp_get_wtime();

                #pragma omp for schedule(static)
                for (int i = 1; i < NX - 1; i++) {
                    for (int j = 1; j < NY - 1; j++) {
                        T_new[i][j] = T[i][j]
                            + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                            + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                    }
                }

                double t1 = omp_get_wtime();
                thread_times[tid] += (t1 - t0);
            }
            swap(T, T_new);
        }

        double wall_end = omp_get_wtime();

        double t_max = *max_element(thread_times.begin(), thread_times.end());
        double t_min = *min_element(thread_times.begin(), thread_times.end());
        double t_avg = accumulate(thread_times.begin(), thread_times.end(), 0.0) / max_threads;
        double imbalance = (t_avg > 0) ? (t_max - t_avg) / t_avg : 0.0;

        cout << "  Wall time:     " << fixed << setprecision(6) << (wall_end - wall_start) << " s" << endl;
        cout << "  T_max:         " << t_max << " s" << endl;
        cout << "  T_min:         " << t_min << " s" << endl;
        cout << "  T_avg:         " << t_avg << " s" << endl;
        cout << "  Imbalance:     " << setprecision(2) << (imbalance * 100) << "%" << endl;

        cout << "\n  Per-thread cumulative work time:" << endl;
        for (int t = 0; t < max_threads; t++) {
            double barrier_wait = t_max - thread_times[t];
            cout << "    Thread " << setw(2) << t
                 << " | Work: " << fixed << setprecision(6) << thread_times[t] << " s"
                 << " | Barrier wait: " << barrier_wait << " s" << endl;
        }
    }

    
    cout << "\n--- Memory Bandwidth Estimation ---" << endl;
    {
        omp_set_num_threads(max_threads);
        initialize(T, NX, NY);
        for (int i = 0; i < NX; i++)
            for (int j = 0; j < NY; j++)
                T_new[i][j] = 0.0;

        double rx = ALPHA * DT / (DX * DX);
        double ry = ALPHA * DT / (DY * DY);

        auto bw_start = high_resolution_clock::now();
        for (int t = 0; t < TIME_STEPS; t++) {
            #pragma omp parallel for schedule(static)
            for (int i = 1; i < NX - 1; i++) {
                for (int j = 1; j < NY - 1; j++) {
                    T_new[i][j] = T[i][j]
                        + rx * (T[i+1][j] - 2.0 * T[i][j] + T[i-1][j])
                        + ry * (T[i][j+1] - 2.0 * T[i][j] + T[i][j-1]);
                }
            }
            swap(T, T_new);
        }
        auto bw_end = high_resolution_clock::now();
        double bw_time = duration<double>(bw_end - bw_start).count();

        
        
        double bytes_per_step = (double)(NX - 2) * (NY - 2) * 6.0 * sizeof(double);
        double total_bytes = bytes_per_step * TIME_STEPS;
        double bandwidth_gb_s = (total_bytes / bw_time) / 1e9;

        cout << "  Time:            " << fixed << setprecision(6) << bw_time << " s" << endl;
        cout << "  Data moved:      " << setprecision(2) << (total_bytes / 1e9) << " GB" << endl;
        cout << "  Eff. Bandwidth:  " << bandwidth_gb_s << " GB/s" << endl;
    }

    cout << "\n========================================================" << endl;
    cout << "  Experiment Complete." << endl;
    cout << "========================================================" << endl;

    return 0;
}
