#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    int world_size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int local_value = world_rank + 1;
    printf("Process %d has local value: %d\n", world_rank, local_value);

    int global_sum = 0;
    MPI_Reduce(&local_value, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\nGlobal sum: %d\n", global_sum);
        printf("Expected: %d\n", (world_size * (world_size + 1)) / 2);
    }

    MPI_Finalize();
    return 0;
}
