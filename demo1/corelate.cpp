#include "correlate.h"

#include <cmath>
#include <cstddef>
#include <vector>

#include <omp.h>

namespace {

void normalize_rows_seq(int ny, int nx, const float* data, std::vector<double>& normalized) {
    normalized.assign(static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx), 0.0);

    for (int y = 0; y < ny; ++y) {
        const float* row = data + static_cast<std::size_t>(y) * static_cast<std::size_t>(nx);
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int x = 0; x < nx; ++x) {
            const double value = static_cast<double>(row[x]);
            sum += value;
            sum_sq += value * value;
        }

        const double mean = sum / static_cast<double>(nx);
        const double centered_sq_sum = sum_sq - static_cast<double>(nx) * mean * mean;
        const double inv_norm = (centered_sq_sum > 0.0) ? 1.0 / std::sqrt(centered_sq_sum) : 0.0;

        double* normalized_row = normalized.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(nx);
        for (int x = 0; x < nx; ++x) {
            normalized_row[x] = (static_cast<double>(row[x]) - mean) * inv_norm;
        }
    }
}

void normalize_rows_par(int ny, int nx, const float* data, std::vector<double>& normalized, int num_threads) {
    normalized.assign(static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx), 0.0);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (int y = 0; y < ny; ++y) {
        const float* row = data + static_cast<std::size_t>(y) * static_cast<std::size_t>(nx);
        double sum = 0.0;
        double sum_sq = 0.0;

        for (int x = 0; x < nx; ++x) {
            const double value = static_cast<double>(row[x]);
            sum += value;
            sum_sq += value * value;
        }

        const double mean = sum / static_cast<double>(nx);
        const double centered_sq_sum = sum_sq - static_cast<double>(nx) * mean * mean;
        const double inv_norm = (centered_sq_sum > 0.0) ? 1.0 / std::sqrt(centered_sq_sum) : 0.0;

        double* normalized_row = normalized.data() + static_cast<std::size_t>(y) * static_cast<std::size_t>(nx);
        for (int x = 0; x < nx; ++x) {
            normalized_row[x] = (static_cast<double>(row[x]) - mean) * inv_norm;
        }
    }
}

} // namespace

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> normalized;
    normalize_rows_seq(ny, nx, data, normalized);

    for (int i = 0; i < ny; ++i) {
        const double* row_i = normalized.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(nx);

        for (int j = 0; j <= i; ++j) {
            const double* row_j = normalized.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(nx);
            double dot = 0.0;

            for (int x = 0; x < nx; ++x) {
                dot += row_i[x] * row_j[x];
            }

            result[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ny)] = static_cast<float>(dot);
        }
    }
}

void correlate_parallel(int ny, int nx, const float* data, float* result, int num_threads) {
    if (num_threads < 1) {
        num_threads = 1;
    }

    std::vector<double> normalized;
    normalize_rows_par(ny, nx, data, normalized, num_threads);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (int i = 0; i < ny; ++i) {
        const double* row_i = normalized.data() + static_cast<std::size_t>(i) * static_cast<std::size_t>(nx);

        for (int j = 0; j <= i; ++j) {
            const double* row_j = normalized.data() + static_cast<std::size_t>(j) * static_cast<std::size_t>(nx);
            double dot = 0.0;

            #pragma omp simd reduction(+:dot)
            for (int x = 0; x < nx; ++x) {
                dot += row_i[x] * row_j[x];
            }

            result[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * static_cast<std::size_t>(ny)] = static_cast<float>(dot);
        }
    }
}
