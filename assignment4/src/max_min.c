#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int local_count = 10;
    int local_values[10];

    unsigned int seed = (unsigned int)time(NULL) ^ (unsigned int)(rank * 2654435761U);
    for (int i = 0; i < local_count; ++i) {
        local_values[i] = rand_r(&seed) % 1001;
    }

    int local_max = local_values[0];
    int local_min = local_values[0];
    for (int i = 1; i < local_count; ++i) {
        if (local_values[i] > local_max) {
            local_max = local_values[i];
        }
        if (local_values[i] < local_min) {
            local_min = local_values[i];
        }
    }

    int max_pair[2] = {local_max, rank};
    int min_pair[2] = {local_min, rank};
    int global_max_pair[2] = {0, 0};
    int global_min_pair[2] = {0, 0};

    MPI_Reduce(max_pair, global_max_pair, 1, MPI_2INT, MPI_MAXLOC, 0, MPI_COMM_WORLD);
    MPI_Reduce(min_pair, global_min_pair, 1, MPI_2INT, MPI_MINLOC, 0, MPI_COMM_WORLD);

    printf("Process %d local min=%d local max=%d\n", rank, local_min, local_max);

    if (rank == 0) {
        printf("Global maximum: %d (from process %d)\n", global_max_pair[0], global_max_pair[1]);
        printf("Global minimum: %d (from process %d)\n", global_min_pair[0], global_min_pair[1]);
        printf("Total processes used: %d\n", size);
    }

    MPI_Finalize();
    return 0;
}
