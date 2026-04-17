#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int data = 0;
    if (world_rank == 0) {
        data = 100;
        printf("Process 0 broadcasting value: %d\n", data);
    }

    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank != 0) {
        printf("Process %d received broadcasted value: %d\n", world_rank, data);
    }

    MPI_Finalize();
    return 0;
}
