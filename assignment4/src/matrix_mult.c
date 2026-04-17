#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void build_counts_displs(int total_rows, int cols, int size, int* counts, int* displs) {
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        int rows = total_rows / size + (r < (total_rows % size) ? 1 : 0);
        counts[r] = rows * cols;
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

    int n = 4;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        if (rank == 0) {
            printf("Matrix size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int* matrix = NULL;
    int* vector = (int*)malloc((size_t)n * sizeof(int));
    int* result = NULL;

    int* send_counts = (int*)malloc((size_t)size * sizeof(int));
    int* send_displs = (int*)malloc((size_t)size * sizeof(int));
    int* recv_counts = (int*)malloc((size_t)size * sizeof(int));
    int* recv_displs = (int*)malloc((size_t)size * sizeof(int));

    build_counts_displs(n, n, size, send_counts, send_displs);
    build_counts_displs(n, 1, size, recv_counts, recv_displs);

    int local_elems = send_counts[rank];
    int local_rows = recv_counts[rank];
    int* local_matrix = (int*)malloc((size_t)local_elems * sizeof(int));
    int* local_result = (int*)malloc((size_t)local_rows * sizeof(int));

    if (rank == 0) {
        matrix = (int*)malloc((size_t)n * (size_t)n * sizeof(int));
        result = (int*)malloc((size_t)n * sizeof(int));

        printf("Matrix (%d x %d):\n", n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                matrix[(size_t)i * (size_t)n + (size_t)j] = i + j;
                printf("%d ", matrix[(size_t)i * (size_t)n + (size_t)j]);
            }
            printf("\n");
        }

        printf("\nVector:\n");
        for (int i = 0; i < n; ++i) {
            vector[i] = i;
            printf("%d ", vector[i]);
        }
        printf("\n\n");
    }

    MPI_Scatterv(matrix, send_counts, send_displs, MPI_INT,
                 local_matrix, local_elems, MPI_INT,
                 0, MPI_COMM_WORLD);

    MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_rows; ++i) {
        local_result[i] = 0;
        for (int j = 0; j < n; ++j) {
            local_result[i] += local_matrix[(size_t)i * (size_t)n + (size_t)j] * vector[j];
        }
    }

    if (rank == 0 && result == NULL) {
        result = (int*)malloc((size_t)n * sizeof(int));
    }

    MPI_Gatherv(local_result, local_rows, MPI_INT,
                result, recv_counts, recv_displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Result:\n");
        for (int i = 0; i < n; ++i) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    free(local_result);
    free(local_matrix);
    free(recv_displs);
    free(recv_counts);
    free(send_displs);
    free(send_counts);
    free(vector);
    if (rank == 0) {
        free(result);
        free(matrix);
    }

    MPI_Finalize();
    return 0;
}
