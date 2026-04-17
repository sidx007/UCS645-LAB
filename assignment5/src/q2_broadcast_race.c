#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void my_bcast(double* buffer, int count, int rank, int size) {
    const int tag = 101;
    if (rank == 0) {
        for (int dest = 1; dest < size; ++dest) {
            MPI_Send(buffer, count, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(buffer, count, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

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

    double* buffer = (double*)malloc((size_t)elements * sizeof(double));
    if (buffer == NULL) {
        printf("Rank %d failed to allocate %.2f MB.\n", rank,
               ((double)elements * sizeof(double)) / (1024.0 * 1024.0));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    if (rank == 0) {
        for (int i = 0; i < elements; ++i) {
            buffer[i] = 0.5 * (double)i;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    my_bcast(buffer, elements, rank, size);
    double local_my_time = MPI_Wtime() - t0;

    double my_time = 0.0;
    MPI_Reduce(&local_my_time, &my_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_probe = buffer[0] + buffer[elements / 2] + buffer[elements - 1];
    double probe_sum = 0.0;
    MPI_Reduce(&local_probe, &probe_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < elements; ++i) {
            buffer[i] = 1.0 + 0.25 * (double)i;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    MPI_Bcast(buffer, elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double local_mpi_time = MPI_Wtime() - t0;

    double mpi_bcast_time = 0.0;
    MPI_Reduce(&local_mpi_time, &mpi_bcast_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    local_probe = buffer[0] + buffer[elements / 2] + buffer[elements - 1];
    double probe_sum_2 = 0.0;
    MPI_Reduce(&local_probe, &probe_sum_2, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double gain = (mpi_bcast_time > 0.0) ? (my_time / mpi_bcast_time) : 0.0;
        printf("Q2 Broadcast Race (elements=%d, processes=%d)\n", elements, size);
        printf("MyBcast Time: %.6f s\n", my_time);
        printf("MPI_Bcast Time: %.6f s\n", mpi_bcast_time);
        printf("MPI_Bcast speedup over MyBcast: %.6f x\n", gain);
        printf("Checksum probe (MyBcast total): %.6e\n", probe_sum);
        printf("Checksum probe (MPI_Bcast total): %.6e\n", probe_sum_2);
    }

    free(buffer);
    MPI_Finalize();
    return 0;
}
