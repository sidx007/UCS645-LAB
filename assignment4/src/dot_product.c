#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void build_counts_displs(int total, int size, int* counts, int* displs) {
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        counts[r] = total / size + (r < (total % size) ? 1 : 0);
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

    const int n = 8;
    int* counts = (int*)malloc((size_t)size * sizeof(int));
    int* displs = (int*)malloc((size_t)size * sizeof(int));
    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    int* local_a = (int*)malloc((size_t)local_n * sizeof(int));
    int* local_b = (int*)malloc((size_t)local_n * sizeof(int));

    int a[8] = {0};
    int b[8] = {0};

    if (rank == 0) {
        int temp_a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
        int temp_b[8] = {8, 7, 6, 5, 4, 3, 2, 1};
        for (int i = 0; i < n; ++i) {
            a[i] = temp_a[i];
            b[i] = temp_b[i];
        }
    }

    MPI_Scatterv(a, counts, displs, MPI_INT,
                 local_a, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, counts, displs, MPI_INT,
                 local_b, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    int local_dot = 0;
    for (int i = 0; i < local_n; ++i) {
        local_dot += local_a[i] * local_b[i];
    }

    int global_dot = 0;
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Dot product: %d\n", global_dot);
        printf("Expected: 120\n");
    }

    free(local_b);
    free(local_a);
    free(displs);
    free(counts);

    MPI_Finalize();
    return 0;
}
