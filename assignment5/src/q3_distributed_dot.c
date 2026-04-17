#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static long long local_chunk(long long n, int rank, int size) {
    long long base = n / size;
    long long rem = n % size;
    return base + (rank < rem ? 1LL : 0LL);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 500000000LL;
    double multiplier = 1.0;

    if (argc > 1) {
        n = atoll(argv[1]);
    }
    if (argc > 2) {
        multiplier = atof(argv[2]);
    }

    if (n <= 0) {
        if (rank == 0) {
            printf("Total element count must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Q3 Distributed Dot Product\n");
        printf("Total elements: %lld\n", n);
        printf("Broadcast multiplier: %.6f\n", multiplier);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    double comm_time = 0.0;
    double comp_time = 0.0;

    double t0 = MPI_Wtime();
    MPI_Bcast(&multiplier, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    long long my_n = local_chunk(n, rank, size);

    t0 = MPI_Wtime();
    double local_dot = 0.0;
    for (long long i = 0; i < my_n; ++i) {
        double a = 1.0;
        double b = 2.0 * multiplier;
        local_dot += a * b;
    }
    comp_time += MPI_Wtime() - t0;

    t0 = MPI_Wtime();
    double global_dot = 0.0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    comm_time += MPI_Wtime() - t0;

    double total_local = MPI_Wtime() - total_start;

    double total_time = 0.0;
    double max_comm = 0.0;
    double max_comp = 0.0;

    MPI_Reduce(&total_local, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double expected = (double)n * 2.0 * multiplier;
        double abs_err = fabs(expected - global_dot);
        printf("Final dot product: %.6f\n", global_dot);
        printf("Expected result: %.6f\n", expected);
        printf("Absolute error: %.6e\n", abs_err);
        printf("Execution Time (Tp): %.6f s\n", total_time);
        printf("Communication Time: %.6f s\n", max_comm);
        printf("Computation Time: %.6f s\n", max_comp);
    }

    MPI_Finalize();
    return 0;
}
