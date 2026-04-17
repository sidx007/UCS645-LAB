#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int next_rank = (rank + 1) % size;
    const int prev_rank = (rank - 1 + size) % size;

    int value = 0;

    if (rank == 0) {
        value = 100;
        printf("Process %d initial value: %d\n", rank, value);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);

        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received final value: %d\n", rank, value);
    } else {
        MPI_Recv(&value, 1, MPI_INT, prev_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value += rank;
        printf("Process %d updated value to: %d\n", rank, value);
        MPI_Send(&value, 1, MPI_INT, next_rank, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
