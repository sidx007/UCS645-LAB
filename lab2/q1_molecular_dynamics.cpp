#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <omp.h>
#include <iomanip>
#include <chrono>
#include <numeric>
#include <algorithm>

using namespace std;
using namespace std::chrono;

const double EPSILON = 1.0;   
const double SIGMA   = 1.0;   
const double CUTOFF  = 2.5 * SIGMA; 
const double CUTOFF2 = CUTOFF * CUTOFF;

struct Particle {
    double x, y, z;    
    double fx, fy, fz; 
};


void initialize_particles(vector<Particle>& particles, int N, double box_size) {
    int n_per_side = (int)ceil(cbrt((double)N));
    double spacing = box_size / n_per_side;
    int idx = 0;
    for (int ix = 0; ix < n_per_side && idx < N; ix++) {
        for (int iy = 0; iy < n_per_side && idx < N; iy++) {
            for (int iz = 0; iz < n_per_side && idx < N; iz++) {
                particles[idx].x = (ix + 0.5) * spacing;
                particles[idx].y = (iy + 0.5) * spacing;
                particles[idx].z = (iz + 0.5) * spacing;
                particles[idx].fx = 0.0;
                particles[idx].fy = 0.0;
                particles[idx].fz = 0.0;
                idx++;
            }
        }
    }
}




double compute_forces_serial(vector<Particle>& particles, int N, double box_size) {
    
    for (int i = 0; i < N; i++) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }

    double total_energy = 0.0;

    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double dz = particles[i].z - particles[j].z;

            
            dx -= box_size * round(dx / box_size);
            dy -= box_size * round(dy / box_size);
            dz -= box_size * round(dz / box_size);

            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < CUTOFF2) {
                double r2_inv = 1.0 / r2;
                double r6_inv = r2_inv * r2_inv * r2_inv;
                double sigma6 = pow(SIGMA, 6);
                double sigma12 = sigma6 * sigma6;

                
                double lj_potential = 4.0 * EPSILON * (sigma12 * r6_inv * r6_inv - sigma6 * r6_inv);
                total_energy += lj_potential;

                
                
                double force_scalar = 24.0 * EPSILON * (2.0 * sigma12 * r6_inv * r6_inv - sigma6 * r6_inv) * r2_inv;

                particles[i].fx += force_scalar * dx;
                particles[i].fy += force_scalar * dy;
                particles[i].fz += force_scalar * dz;
                particles[j].fx -= force_scalar * dx;
                particles[j].fy -= force_scalar * dy;
                particles[j].fz -= force_scalar * dz;
            }
        }
    }

    return total_energy;
}





double compute_forces_parallel(vector<Particle>& particles, int N, double box_size) {
    
    for (int i = 0; i < N; i++) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }

    double total_energy = 0.0;
    int num_threads = omp_get_max_threads();

    
    
    vector<vector<double>> local_fx(num_threads, vector<double>(N, 0.0));
    vector<vector<double>> local_fy(num_threads, vector<double>(N, 0.0));
    vector<vector<double>> local_fz(num_threads, vector<double>(N, 0.0));

    
    
    
    
    #pragma omp parallel reduction(+:total_energy)
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = particles[i].x - particles[j].x;
                double dy = particles[i].y - particles[j].y;
                double dz = particles[i].z - particles[j].z;

                
                dx -= box_size * round(dx / box_size);
                dy -= box_size * round(dy / box_size);
                dz -= box_size * round(dz / box_size);

                double r2 = dx * dx + dy * dy + dz * dz;

                if (r2 < CUTOFF2) {
                    double r2_inv = 1.0 / r2;
                    double r6_inv = r2_inv * r2_inv * r2_inv;
                    double sigma6 = pow(SIGMA, 6);
                    double sigma12 = sigma6 * sigma6;

                    double lj_potential = 4.0 * EPSILON * (sigma12 * r6_inv * r6_inv - sigma6 * r6_inv);
                    total_energy += lj_potential;

                    double force_scalar = 24.0 * EPSILON * (2.0 * sigma12 * r6_inv * r6_inv - sigma6 * r6_inv) * r2_inv;

                    
                    local_fx[tid][i] += force_scalar * dx;
                    local_fy[tid][i] += force_scalar * dy;
                    local_fz[tid][i] += force_scalar * dz;
                    local_fx[tid][j] -= force_scalar * dx;
                    local_fy[tid][j] -= force_scalar * dy;
                    local_fz[tid][j] -= force_scalar * dz;
                }
            }
        }
    }

    
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        for (int t = 0; t < num_threads; t++) {
            particles[i].fx += local_fx[t][i];
            particles[i].fy += local_fy[t][i];
            particles[i].fz += local_fz[t][i];
        }
    }

    return total_energy;
}




double compute_forces_parallel_atomic(vector<Particle>& particles, int N, double box_size) {
    for (int i = 0; i < N; i++) {
        particles[i].fx = 0.0;
        particles[i].fy = 0.0;
        particles[i].fz = 0.0;
    }

    double total_energy = 0.0;

    #pragma omp parallel for schedule(dynamic, 4) reduction(+:total_energy)
    for (int i = 0; i < N - 1; i++) {
        for (int j = i + 1; j < N; j++) {
            double dx = particles[i].x - particles[j].x;
            double dy = particles[i].y - particles[j].y;
            double dz = particles[i].z - particles[j].z;

            dx -= box_size * round(dx / box_size);
            dy -= box_size * round(dy / box_size);
            dz -= box_size * round(dz / box_size);

            double r2 = dx * dx + dy * dy + dz * dz;

            if (r2 < CUTOFF2) {
                double r2_inv = 1.0 / r2;
                double r6_inv = r2_inv * r2_inv * r2_inv;
                double sigma6 = pow(SIGMA, 6);
                double sigma12 = sigma6 * sigma6;

                double lj_potential = 4.0 * EPSILON * (sigma12 * r6_inv * r6_inv - sigma6 * r6_inv);
                total_energy += lj_potential;

                double force_scalar = 24.0 * EPSILON * (2.0 * sigma12 * r6_inv * r6_inv - sigma6 * r6_inv) * r2_inv;

                double fx = force_scalar * dx;
                double fy = force_scalar * dy;
                double fz = force_scalar * dz;

                
                particles[i].fx += fx;
                particles[i].fy += fy;
                particles[i].fz += fz;

                
                #pragma omp atomic
                particles[j].fx -= fx;
                #pragma omp atomic
                particles[j].fy -= fy;
                #pragma omp atomic
                particles[j].fz -= fz;
            }
        }
    }

    return total_energy;
}

int main() {
    
    const int N = 1000;           
    const double BOX_SIZE = 10.0; 
    const int max_threads = omp_get_max_threads();

    cout << "========================================================" << endl;
    cout << "  Molecular Dynamics - Lennard-Jones Force Calculation" << endl;
    cout << "  Particles: " << N << "  |  Box Size: " << BOX_SIZE << endl;
    cout << "  Available threads: " << max_threads << endl;
    cout << "========================================================" << endl;

    
    vector<Particle> particles_serial(N);
    initialize_particles(particles_serial, N, BOX_SIZE);

    auto start = high_resolution_clock::now();
    double energy_serial = compute_forces_serial(particles_serial, N, BOX_SIZE);
    auto end = high_resolution_clock::now();
    double t_serial = duration<double>(end - start).count();

    cout << "\n--- Serial Baseline ---" << endl;
    cout << "  Energy:  " << fixed << setprecision(6) << energy_serial << endl;
    cout << "  Time:    " << t_serial << " s" << endl;

    
    cout << "\n--- Strong Scaling: Thread-Local Force Arrays ---" << endl;
    cout << left << setw(10) << "Threads"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << setw(15) << "Efficiency"
         << "Energy" << endl;
    cout << string(70, '-') << endl;

    vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);
    if (thread_counts.back() != max_threads) thread_counts.push_back(max_threads);

    for (int nthreads : thread_counts) {
        omp_set_num_threads(nthreads);

        vector<Particle> particles_par(N);
        initialize_particles(particles_par, N, BOX_SIZE);

        start = high_resolution_clock::now();
        double energy_par = compute_forces_parallel(particles_par, N, BOX_SIZE);
        end = high_resolution_clock::now();
        double t_par = duration<double>(end - start).count();

        double speedup = t_serial / t_par;
        double efficiency = speedup / nthreads;

        cout << left << setw(10) << nthreads
             << setw(15) << fixed << setprecision(6) << t_par
             << setw(12) << setprecision(2) << speedup << "x"
             << setw(14) << setprecision(2) << (efficiency * 100) << "%"
             << setprecision(6) << energy_par << endl;
    }

    
    cout << "\n--- Scheduling Strategy Comparison (" << max_threads << " threads) ---" << endl;
    cout << left << setw(15) << "Schedule"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup" << endl;
    cout << string(42, '-') << endl;

    omp_set_num_threads(max_threads);

    
    {
        vector<Particle> p(N);
        initialize_particles(p, N, BOX_SIZE);
        for (int i = 0; i < N; i++) { p[i].fx = p[i].fy = p[i].fz = 0.0; }
        double en = 0.0;

        start = high_resolution_clock::now();
        #pragma omp parallel for schedule(static) reduction(+:en)
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, dz = p[i].z - p[j].z;
                dx -= BOX_SIZE * round(dx / BOX_SIZE);
                dy -= BOX_SIZE * round(dy / BOX_SIZE);
                dz -= BOX_SIZE * round(dz / BOX_SIZE);
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < CUTOFF2) {
                    double r2i = 1.0/r2, r6i = r2i*r2i*r2i;
                    double s6 = pow(SIGMA,6), s12 = s6*s6;
                    en += 4.0*EPSILON*(s12*r6i*r6i - s6*r6i);
                }
            }
        }
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();
        cout << left << setw(15) << "static" << setw(15) << t << (t_serial/t) << "x" << endl;
    }

    
    {
        vector<Particle> p(N);
        initialize_particles(p, N, BOX_SIZE);
        for (int i = 0; i < N; i++) { p[i].fx = p[i].fy = p[i].fz = 0.0; }
        double en = 0.0;

        start = high_resolution_clock::now();
        #pragma omp parallel for schedule(dynamic, 4) reduction(+:en)
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, dz = p[i].z - p[j].z;
                dx -= BOX_SIZE * round(dx / BOX_SIZE);
                dy -= BOX_SIZE * round(dy / BOX_SIZE);
                dz -= BOX_SIZE * round(dz / BOX_SIZE);
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < CUTOFF2) {
                    double r2i = 1.0/r2, r6i = r2i*r2i*r2i;
                    double s6 = pow(SIGMA,6), s12 = s6*s6;
                    en += 4.0*EPSILON*(s12*r6i*r6i - s6*r6i);
                }
            }
        }
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();
        cout << left << setw(15) << "dynamic,4" << setw(15) << t << (t_serial/t) << "x" << endl;
    }

    
    {
        vector<Particle> p(N);
        initialize_particles(p, N, BOX_SIZE);
        for (int i = 0; i < N; i++) { p[i].fx = p[i].fy = p[i].fz = 0.0; }
        double en = 0.0;

        start = high_resolution_clock::now();
        #pragma omp parallel for schedule(guided) reduction(+:en)
        for (int i = 0; i < N - 1; i++) {
            for (int j = i + 1; j < N; j++) {
                double dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, dz = p[i].z - p[j].z;
                dx -= BOX_SIZE * round(dx / BOX_SIZE);
                dy -= BOX_SIZE * round(dy / BOX_SIZE);
                dz -= BOX_SIZE * round(dz / BOX_SIZE);
                double r2 = dx*dx + dy*dy + dz*dz;
                if (r2 < CUTOFF2) {
                    double r2i = 1.0/r2, r6i = r2i*r2i*r2i;
                    double s6 = pow(SIGMA,6), s12 = s6*s6;
                    en += 4.0*EPSILON*(s12*r6i*r6i - s6*r6i);
                }
            }
        }
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();
        cout << left << setw(15) << "guided" << setw(15) << t << (t_serial/t) << "x" << endl;
    }

    
    cout << "\n--- Per-Thread Timing (Load Imbalance Analysis, " << max_threads << " threads) ---" << endl;
    {
        omp_set_num_threads(max_threads);
        vector<Particle> p(N);
        initialize_particles(p, N, BOX_SIZE);
        vector<double> thread_times(max_threads, 0.0);

        double wall_start = omp_get_wtime();
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double t0 = omp_get_wtime();

            double local_energy = 0.0;
            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < N - 1; i++) {
                for (int j = i + 1; j < N; j++) {
                    double dx = p[i].x - p[j].x, dy = p[i].y - p[j].y, dz = p[i].z - p[j].z;
                    dx -= BOX_SIZE * round(dx / BOX_SIZE);
                    dy -= BOX_SIZE * round(dy / BOX_SIZE);
                    dz -= BOX_SIZE * round(dz / BOX_SIZE);
                    double r2 = dx*dx + dy*dy + dz*dz;
                    if (r2 < CUTOFF2) {
                        double r2i = 1.0/r2, r6i = r2i*r2i*r2i;
                        double s6 = pow(SIGMA,6), s12 = s6*s6;
                        local_energy += 4.0*EPSILON*(s12*r6i*r6i - s6*r6i);
                    }
                }
            }

            double t1 = omp_get_wtime();
            thread_times[tid] = t1 - t0;
        }
        double wall_end = omp_get_wtime();

        double t_max = *max_element(thread_times.begin(), thread_times.end());
        double t_min = *min_element(thread_times.begin(), thread_times.end());
        double t_avg = accumulate(thread_times.begin(), thread_times.end(), 0.0) / max_threads;
        double imbalance = (t_max - t_avg) / t_avg;

        cout << "  Wall time:    " << (wall_end - wall_start) << " s" << endl;
        cout << "  T_max:        " << t_max << " s" << endl;
        cout << "  T_min:        " << t_min << " s" << endl;
        cout << "  T_avg:        " << t_avg << " s" << endl;
        cout << "  Imbalance:    " << fixed << setprecision(2) << (imbalance * 100) << "%" << endl;

        cout << "\n  Per-thread breakdown:" << endl;
        for (int t = 0; t < max_threads; t++) {
            double barrier_wait = t_max - thread_times[t];
            cout << "    Thread " << setw(2) << t
                 << " | Work: " << fixed << setprecision(6) << thread_times[t] << " s"
                 << " | Barrier wait: " << barrier_wait << " s" << endl;
        }
    }

    cout << "\n========================================================" << endl;
    cout << "  Experiment Complete." << endl;
    cout << "========================================================" << endl;

    return 0;
}
