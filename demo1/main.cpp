#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <omp.h>

#include "correlate.h"

using Clock = std::chrono::high_resolution_clock;

struct RunStats {
    int threads;
    double time_ms;
    double speedup;
    double max_abs_diff;
};

void fill_random(std::vector<float>& data, unsigned int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);
    for (float& value : data) {
        value = dist(gen);
    }
}

template <typename Func>
double time_ms(Func&& func) {
    const auto start = Clock::now();
    func();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double max_abs_difference(const std::vector<float>& a, const std::vector<float>& b) {
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const double diff = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

std::vector<int> thread_sweep(int max_threads) {
    std::vector<int> threads;
    threads.push_back(1);

    int current = 1;
    while (current < max_threads) {
        current *= 2;
        if (current <= max_threads) {
            threads.push_back(current);
        }
    }

    if (threads.back() != max_threads) {
        threads.push_back(max_threads);
    }

    return threads;
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <ny> <nx> [max_threads] [mode]\n";
    std::cerr << "  mode options: all | seq | par\n";
}

int parse_positive_int(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);

    if (end == text || *end != '\0' || value <= 0 || value > std::numeric_limits<int>::max()) {
        std::cerr << "Invalid value for " << name << ": " << text << "\n";
        std::exit(1);
    }

    return static_cast<int>(value);
}

int main(int argc, char** argv) {
    const int default_ny = 512;
    const int default_nx = 512;

    int ny = default_ny;
    int nx = default_nx;
    int requested_threads = omp_get_max_threads();
    std::string mode = "all";

    if (argc > 1) {
        ny = parse_positive_int(argv[1], "ny");
    }
    if (argc > 2) {
        nx = parse_positive_int(argv[2], "nx");
    }
    if (argc > 3) {
        requested_threads = parse_positive_int(argv[3], "max_threads");
    }
    if (argc > 4) {
        mode = argv[4];
    }
    if (argc > 5) {
        print_usage(argv[0]);
        return 1;
    }

    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

    const int max_threads = std::max(1, std::min(requested_threads, omp_get_max_threads()));
    const std::size_t matrix_size = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
    const std::size_t result_size = static_cast<std::size_t>(ny) * static_cast<std::size_t>(ny);

    std::vector<float> data(matrix_size);
    std::vector<float> result_seq(result_size, 0.0f);
    std::vector<float> result_par(result_size, 0.0f);
    fill_random(data, 42U);

    std::cout << "=============================================\n";
    std::cout << "Correlation Assignment Benchmark\n";
    std::cout << "=============================================\n";
    std::cout << "Rows (ny): " << ny << "\n";
    std::cout << "Cols (nx): " << nx << "\n";
    std::cout << "OpenMP max threads requested: " << requested_threads << "\n";
    std::cout << "OpenMP max threads used: " << max_threads << "\n";
    std::cout << "Mode: " << mode << "\n\n";

    if (mode == "seq") {
        const double seq_ms = time_ms([&]() { correlate(ny, nx, data.data(), result_seq.data()); });
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Sequential time: " << seq_ms << " ms\n";
        return 0;
    }

    if (mode == "par") {
        const double par_ms =
            time_ms([&]() { correlate_parallel(ny, nx, data.data(), result_par.data(), max_threads); });
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Parallel time (" << max_threads << " threads): " << par_ms << " ms\n";
        return 0;
    }

    if (mode != "all") {
        print_usage(argv[0]);
        return 1;
    }

    const double seq_ms = time_ms([&]() { correlate(ny, nx, data.data(), result_seq.data()); });
    const std::vector<int> threads_to_test = thread_sweep(max_threads);

    std::vector<RunStats> stats;
    stats.reserve(threads_to_test.size());

    for (const int threads : threads_to_test) {
        std::fill(result_par.begin(), result_par.end(), 0.0f);
        const double par_ms =
            time_ms([&]() { correlate_parallel(ny, nx, data.data(), result_par.data(), threads); });

        const double diff = max_abs_difference(result_seq, result_par);
        const double speedup = seq_ms / par_ms;

        stats.push_back({threads, par_ms, speedup, diff});
    }

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Sequential time: " << seq_ms << " ms\n\n";

    std::cout << std::left << std::setw(10) << "Threads" << std::setw(16) << "Parallel(ms)" << std::setw(12)
              << "Speedup" << std::setw(16) << "MaxAbsDiff" << "\n";
    std::cout << "----------------------------------------------------------\n";

    for (const RunStats& row : stats) {
        std::cout << std::left << std::setw(10) << row.threads << std::setw(16) << row.time_ms << std::setw(12)
                  << row.speedup << std::setw(16) << row.max_abs_diff << "\n";
    }

    const auto best_it = std::min_element(
        stats.begin(), stats.end(), [](const RunStats& a, const RunStats& b) { return a.time_ms < b.time_ms; });

    if (best_it != stats.end()) {
        std::cout << "\nBest parallel time: " << best_it->time_ms << " ms (" << best_it->threads
                  << " threads)\n";
        std::cout << "Best speedup: " << best_it->speedup << "x\n";
    }

    std::cout << "\nFor Linux perf stats you can run:\n";
    std::cout << "  make perf-seq NY=" << ny << " NX=" << nx << "\n";
    std::cout << "  make perf-par NY=" << ny << " NX=" << nx << " THREADS=" << max_threads << "\n";

    return 0;
}