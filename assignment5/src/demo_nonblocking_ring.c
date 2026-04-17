#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int elements = 10000000;
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
    if (send_buf == NULL || recv_buf == NULL) {
        printf("Rank %d failed to allocate buffers.\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    for (int i = 0; i < elements; ++i) {
        send_buf[i] = rank + i;
        recv_buf[i] = 0;
    }

    MPI_Request requests[2];

    printf("[NonBlocking] Rank %d posting Irecv/Isend (elements=%d)\n", rank, elements);

    MPI_Irecv(recv_buf, elements, MPI_INT, left_neighbor, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Isend(send_buf, elements, MPI_INT, right_neighbor, 0, MPI_COMM_WORLD, &requests[1]);

    double work = 0.0;
    for (long i = 0; i < 5000000; ++i) {
        work += (double)(i % 7) * 0.125;
    }

    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

    printf("[NonBlocking] Rank %d completed transfer. recv_buf[0]=%d, work=%.3f\n",
           rank, recv_buf[0], work);

    free(recv_buf);
    free(send_buf);

    MPI_Finalize();
    return 0;
}
