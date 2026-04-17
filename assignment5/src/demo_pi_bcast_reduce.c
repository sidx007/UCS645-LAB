#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long num_steps = 1000000;
    if (argc > 1) {
        num_steps = atol(argv[1]);
    }
    if (num_steps <= 0) {
        if (rank == 0) {
            printf("num_steps must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Rank 0 broadcasting total steps: %ld\n", num_steps);
    }
    MPI_Bcast(&num_steps, 1, MPI_LONG, 0, MPI_COMM_WORLD);

    double step = 1.0 / (double)num_steps;
    double sum = 0.0;

    for (long i = rank; i < num_steps; i += size) {
        double x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x * x);
    }

    double local_pi = step * sum;
    double total_pi = 0.0;

    MPI_Reduce(&local_pi, &total_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Final Pi calculated by rank 0: %.10f\n", total_pi);
    }

    MPI_Finalize();
    return 0;
}
