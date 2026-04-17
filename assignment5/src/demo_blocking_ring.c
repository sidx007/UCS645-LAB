#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements = 100;
    if (argc > 1) {
        elements = atoi(argv[1]);
    }
    if (elements <= 0) {
        if (rank == 0) {
            printf("Element count must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int right_neighbor = (rank + 1) % size;
    int left_neighbor = (rank - 1 + size) % size;

    int* send_buf = (int*)malloc((size_t)elements * sizeof(int));
    int* recv_buf = (int*)malloc((size_t)elements * sizeof(int));

    for (int i = 0; i < elements; ++i) {
        send_buf[i] = rank;
    }

    printf("[Blocking] Rank %d sending to %d and then receiving from %d (elements=%d)\n",
           rank, right_neighbor, left_neighbor, elements);

    MPI_Send(send_buf, elements, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD);
    MPI_Recv(recv_buf, elements, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("[Blocking] Rank %d completed transfer. recv_buf[0]=%d\n", rank, recv_buf[0]);

    free(recv_buf);
    free(send_buf);

    MPI_Finalize();
    return 0;
}
