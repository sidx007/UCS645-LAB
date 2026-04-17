#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void build_counts_displs(int n, int size, int* counts, int* displs) {
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = n / size + (r < (n % size) ? 1 : 0);
        displs[r] = offset;
        offset += counts[r];
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 1 << 16;
    double a = 2.5;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        a = atof(argv[2]);
    }

    if (n <= 0) {
        if (rank == 0) {
            printf("Vector size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int* counts = (int*)malloc((size_t)size * sizeof(int));
    int* displs = (int*)malloc((size_t)size * sizeof(int));
    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    double* local_x = (double*)malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(double));
    double* local_y = (double*)malloc((size_t)(local_n > 0 ? local_n : 1) * sizeof(double));

    double* x = NULL;
    double* y = NULL;
    double* x_seq = NULL;
    double* x_mpi = NULL;

    if (rank == 0) {
        x = (double*)malloc((size_t)n * sizeof(double));
        y = (double*)malloc((size_t)n * sizeof(double));
        x_seq = (double*)malloc((size_t)n * sizeof(double));
        x_mpi = (double*)malloc((size_t)n * sizeof(double));

        for (int i = 0; i < n; ++i) {
            x[i] = 1.0 + 0.001 * (double)i;
            y[i] = 2.0 - 0.0005 * (double)i;
            x_seq[i] = x[i];
        }
    }

    double seq_time = 0.0;
    if (rank == 0) {
        double t0 = MPI_Wtime();
        for (int i = 0; i < n; ++i) {
            x_seq[i] = a * x_seq[i] + y[i];
        }
        seq_time = MPI_Wtime() - t0;
    }

    MPI_Scatterv(x, counts, displs, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(y, counts, displs, MPI_DOUBLE, local_y, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < local_n; ++i) {
        local_x[i] = a * local_x[i] + local_y[i];
    }
    double local_time = MPI_Wtime() - t0;

    double mpi_time = 0.0;
    MPI_Reduce(&local_time, &mpi_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    MPI_Gatherv(local_x, local_n, MPI_DOUBLE, x_mpi, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double max_abs_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            double diff = fabs(x_seq[i] - x_mpi[i]);
            if (diff > max_abs_diff) {
                max_abs_diff = diff;
            }
        }

        double speedup = (mpi_time > 0.0) ? (seq_time / mpi_time) : 0.0;
        double efficiency = (size > 0) ? (speedup / (double)size) : 0.0;

        printf("Q1 DAXPY (n=%d, a=%.6f)\n", n, a);
        printf("Sequential Time (T1): %.6f s\n", seq_time);
        printf("MPI Time (Tp): %.6f s\n", mpi_time);
        printf("Speedup (T1/Tp): %.6f\n", speedup);
        printf("Efficiency (Speedup/p): %.6f\n", efficiency);
        printf("Max absolute difference: %.6e\n", max_abs_diff);
    }

    free(local_y);
    free(local_x);
    free(displs);
    free(counts);

    if (rank == 0) {
        free(x_mpi);
        free(x_seq);
        free(y);
        free(x);
    }

    MPI_Finalize();
    return 0;
}
