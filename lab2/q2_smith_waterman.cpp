#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <numeric>

using namespace std;
using namespace std::chrono;

const int MATCH    =  2;
const int MISMATCH = -1;
const int GAP      = -1;

string generate_dna(int length) {
    const char bases[] = "ACGT";
    string seq(length, ' ');
    for (int i = 0; i < length; i++) {
        seq[i] = bases[rand() % 4];
    }
    return seq;
}

inline int score(char a, char b) {
    return (a == b) ? MATCH : MISMATCH;
}

int smith_waterman_serial(const string& seq1, const string& seq2,
                          vector<vector<int>>& H) {
    int m = seq1.length();
    int n = seq2.length();
    int max_score = 0;


    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            H[i][j] = 0;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int up   = H[i-1][j] + GAP;
            int left = H[i][j-1] + GAP;
            H[i][j] = max({0, diag, up, left});
            if (H[i][j] > max_score)
                max_score = H[i][j];
        }
    }
    return max_score;
}

int smith_waterman_wavefront(const string& seq1, const string& seq2,
                              vector<vector<int>>& H, const string& sched_name) {
    int m = seq1.length();
    int n = seq2.length();
    int max_score = 0;

    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            H[i][j] = 0;




    int total_diags = m + n - 1;

    for (int d = 0; d < total_diags; d++) {
        int i_start = max(1, d + 2 - n);
        int i_end   = min(m, d + 1);
        int diag_len = i_end - i_start + 1;

        int local_max = 0;

    
        if (sched_name == "static") {
            #pragma omp parallel for schedule(static) reduction(max:local_max)
            for (int idx = 0; idx < diag_len; idx++) {
                int i = i_start + idx;
                int j = d + 2 - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j] + GAP;
                int left = H[i][j-1] + GAP;
                H[i][j] = max({0, diag, up, left});
                if (H[i][j] > local_max)
                    local_max = H[i][j];
            }
        } else if (sched_name == "dynamic") {
            #pragma omp parallel for schedule(dynamic, 16) reduction(max:local_max)
            for (int idx = 0; idx < diag_len; idx++) {
                int i = i_start + idx;
                int j = d + 2 - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j] + GAP;
                int left = H[i][j-1] + GAP;
                H[i][j] = max({0, diag, up, left});
                if (H[i][j] > local_max)
                    local_max = H[i][j];
            }
        } else {
            #pragma omp parallel for schedule(guided) reduction(max:local_max)
            for (int idx = 0; idx < diag_len; idx++) {
                int i = i_start + idx;
                int j = d + 2 - i;
                int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                int up   = H[i-1][j] + GAP;
                int left = H[i][j-1] + GAP;
                H[i][j] = max({0, diag, up, left});
                if (H[i][j] > local_max)
                    local_max = H[i][j];
            }
        }

        if (local_max > max_score)
            max_score = local_max;
    }

    return max_score;
}

int smith_waterman_row_parallel(const string& seq1, const string& seq2,
                                 vector<vector<int>>& H) {
    int m = seq1.length();
    int n = seq2.length();
    int max_score = 0;

    for (int i = 0; i <= m; i++)
        for (int j = 0; j <= n; j++)
            H[i][j] = 0;





    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            int diag = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
            int up   = H[i-1][j] + GAP;
            int left = H[i][j-1] + GAP;
            H[i][j] = max({0, diag, up, left});
            if (H[i][j] > max_score)
                max_score = H[i][j];
        }
    }

    return max_score;
}

void traceback(const vector<vector<int>>& H, const string& seq1, const string& seq2,
               string& align1, string& align2) {
    int m = seq1.length();
    int n = seq2.length();


    int max_i = 0, max_j = 0, max_score = 0;
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (H[i][j] > max_score) {
                max_score = H[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    align1 = "";
    align2 = "";
    int i = max_i, j = max_j;

    while (i > 0 && j > 0 && H[i][j] > 0) {
        if (H[i][j] == H[i-1][j-1] + score(seq1[i-1], seq2[j-1])) {
            align1 = seq1[i-1] + align1;
            align2 = seq2[j-1] + align2;
            i--; j--;
        } else if (H[i][j] == H[i-1][j] + GAP) {
            align1 = seq1[i-1] + align1;
            align2 = '-' + align2;
            i--;
        } else {
            align1 = '-' + align1;
            align2 = seq2[j-1] + align2;
            j--;
        }
    }
}

int main() {
    srand(42);
    int max_threads = omp_get_max_threads();


    const int SEQ_LEN1 = 5000;
    const int SEQ_LEN2 = 5000;

    string seq1 = generate_dna(SEQ_LEN1);
    string seq2 = generate_dna(SEQ_LEN2);

    cout << "========================================================" << endl;
    cout << "  Smith-Waterman DNA Sequence Alignment" << endl;
    cout << "  Sequence 1 length: " << SEQ_LEN1 << endl;
    cout << "  Sequence 2 length: " << SEQ_LEN2 << endl;
    cout << "  Available threads: " << max_threads << endl;
    cout << "========================================================" << endl;


    vector<vector<int>> H(SEQ_LEN1 + 1, vector<int>(SEQ_LEN2 + 1, 0));


    auto start = high_resolution_clock::now();
    int score_serial = smith_waterman_serial(seq1, seq2, H);
    auto end = high_resolution_clock::now();
    double t_serial = duration<double>(end - start).count();


    string align1, align2;
    traceback(H, seq1, seq2, align1, align2);

    cout << "\n--- Serial Baseline ---" << endl;
    cout << "  Max alignment score: " << score_serial << endl;
    cout << "  Time: " << fixed << setprecision(6) << t_serial << " s" << endl;
    cout << "  Alignment length: " << align1.length() << endl;
    if (align1.length() <= 80) {
        cout << "  Seq1: " << align1 << endl;
        cout << "  Seq2: " << align2 << endl;
    } else {
        cout << "  Seq1: " << align1.substr(0, 60) << "..." << endl;
        cout << "  Seq2: " << align2.substr(0, 60) << "..." << endl;
    }


    cout << "\n--- Strong Scaling: Wavefront Parallelization ---" << endl;
    cout << left << setw(10) << "Threads"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << setw(15) << "Efficiency"
         << "Score" << endl;
    cout << string(65, '-') << endl;

    vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);
    if (thread_counts.back() != max_threads) thread_counts.push_back(max_threads);

    for (int nthreads : thread_counts) {
        omp_set_num_threads(nthreads);

        start = high_resolution_clock::now();
        int sc = smith_waterman_wavefront(seq1, seq2, H, "static");
        end = high_resolution_clock::now();
        double t_par = duration<double>(end - start).count();

        double speedup = t_serial / t_par;
        double efficiency = speedup / nthreads;

        cout << left << setw(10) << nthreads
             << setw(15) << fixed << setprecision(6) << t_par
             << setw(12) << setprecision(2) << speedup << "x"
             << setw(14) << setprecision(2) << (efficiency * 100) << "%"
             << sc << endl;
    }


    cout << "\n--- Scheduling Strategy Comparison (" << max_threads << " threads) ---" << endl;
    cout << left << setw(15) << "Schedule"
         << setw(15) << "Time (s)"
         << setw(12) << "Speedup"
         << "Score" << endl;
    cout << string(55, '-') << endl;

    omp_set_num_threads(max_threads);

    const char* schedules[] = {"static", "dynamic", "guided"};
    for (const char* sched : schedules) {
        start = high_resolution_clock::now();
        int sc = smith_waterman_wavefront(seq1, seq2, H, sched);
        end = high_resolution_clock::now();
        double t = duration<double>(end - start).count();

        cout << left << setw(15) << sched
             << setw(15) << fixed << setprecision(6) << t
             << setw(12) << setprecision(2) << (t_serial / t) << "x"
             << sc << endl;
    }


    cout << "\n--- Per-Thread Timing Analysis (" << max_threads << " threads, wavefront) ---" << endl;
    {
        omp_set_num_threads(max_threads);
        vector<double> thread_work_time(max_threads, 0.0);

        for (int i = 0; i <= SEQ_LEN1; i++)
            for (int j = 0; j <= SEQ_LEN2; j++)
                H[i][j] = 0;

        int m = SEQ_LEN1, n = SEQ_LEN2;
        int total_diags = m + n - 1;
        int final_score = 0;

        for (int d = 0; d < total_diags; d++) {
            int i_start = max(1, d + 2 - n);
            int i_end   = min(m, d + 1);
            int diag_len = i_end - i_start + 1;

            int local_max = 0;

            #pragma omp parallel reduction(max:local_max)
            {
                int tid = omp_get_thread_num();
                double t0 = omp_get_wtime();

                #pragma omp for schedule(dynamic, 16)
                for (int idx = 0; idx < diag_len; idx++) {
                    int i = i_start + idx;
                    int j = d + 2 - i;
                    int diag_val = H[i-1][j-1] + score(seq1[i-1], seq2[j-1]);
                    int up   = H[i-1][j] + GAP;
                    int left = H[i][j-1] + GAP;
                    H[i][j] = max({0, diag_val, up, left});
                    if (H[i][j] > local_max)
                        local_max = H[i][j];
                }

                double t1 = omp_get_wtime();
                thread_work_time[tid] += (t1 - t0);
            }

            if (local_max > final_score)
                final_score = local_max;
        }

        double t_max = *max_element(thread_work_time.begin(), thread_work_time.end());
        double t_min = *min_element(thread_work_time.begin(), thread_work_time.end());
        double t_avg = accumulate(thread_work_time.begin(), thread_work_time.end(), 0.0) / max_threads;
        double imbalance = (t_max - t_avg) / t_avg;

        cout << "  T_max (cumulative): " << fixed << setprecision(6) << t_max << " s" << endl;
        cout << "  T_min (cumulative): " << t_min << " s" << endl;
        cout << "  T_avg (cumulative): " << t_avg << " s" << endl;
        cout << "  Imbalance:          " << setprecision(2) << (imbalance * 100) << "%" << endl;

        cout << "\n  Per-thread cumulative work time:" << endl;
        for (int t = 0; t < max_threads; t++) {
            cout << "    Thread " << setw(2) << t
                 << " | Work: " << fixed << setprecision(6) << thread_work_time[t] << " s" << endl;
        }
    }

    cout << "\n========================================================" << endl;
    cout << "  Experiment Complete." << endl;
    cout << "========================================================" << endl;

    return 0;
}
