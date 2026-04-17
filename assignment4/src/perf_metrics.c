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

    int n = 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        if (rank == 0) {
            printf("n must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int* counts_scatter = (int*)malloc((size_t)size * sizeof(int));
    int* displs_scatter = (int*)malloc((size_t)size * sizeof(int));
    int* counts_gather = (int*)malloc((size_t)size * sizeof(int));
    int* displs_gather = (int*)malloc((size_t)size * sizeof(int));

    build_counts_displs(n, n, size, counts_scatter, displs_scatter);
    build_counts_displs(n, 1, size, counts_gather, displs_gather);

    int local_elems = counts_scatter[rank];
    int local_rows = counts_gather[rank];

    int* matrix = NULL;
    int* vector = (int*)malloc((size_t)n * sizeof(int));
    int* result = NULL;
    int* local_matrix = (int*)malloc((size_t)local_elems * sizeof(int));
    int* local_result = (int*)malloc((size_t)local_rows * sizeof(int));

    if (rank == 0) {
        matrix = (int*)malloc((size_t)n * (size_t)n * sizeof(int));
        result = (int*)malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; ++i) {
            vector[i] = i % 11;
            for (int j = 0; j < n; ++j) {
                matrix[(size_t)i * (size_t)n + (size_t)j] = (i + j) % 13;
            }
        }
    }

    double comm_time = 0.0;
    double comp_time = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double total_start = MPI_Wtime();

    double t0 = MPI_Wtime();
    MPI_Scatterv(matrix, counts_scatter, displs_scatter, MPI_INT,
                 local_matrix, local_elems, MPI_INT,
                 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, n, MPI_INT, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    comm_time += (t1 - t0);

    t0 = MPI_Wtime();
    for (int i = 0; i < local_rows; ++i) {
        int sum = 0;
        for (int j = 0; j < n; ++j) {
            sum += local_matrix[(size_t)i * (size_t)n + (size_t)j] * vector[j];
        }
        local_result[i] = sum;
    }
    t1 = MPI_Wtime();
    comp_time += (t1 - t0);

    t0 = MPI_Wtime();
    MPI_Gatherv(local_result, local_rows, MPI_INT,
                result, counts_gather, displs_gather, MPI_INT,
                0, MPI_COMM_WORLD);
    t1 = MPI_Wtime();
    comm_time += (t1 - t0);

    double total_end = MPI_Wtime();
    double total_time = total_end - total_start;

    double max_total = 0.0;
    double max_comm = 0.0;
    double max_comp = 0.0;

    MPI_Reduce(&total_time, &max_total, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comp_time, &max_comp, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double comm_pct = (max_total > 0.0) ? (100.0 * max_comm / max_total) : 0.0;
        double comp_pct = (max_total > 0.0) ? (100.0 * max_comp / max_total) : 0.0;
        printf("n=%d, processes=%d\n", n, size);
        printf("Execution time (Tp): %.6f s\n", max_total);
        printf("Communication time: %.6f s (%.2f%%)\n", max_comm, comm_pct);
        printf("Computation time: %.6f s (%.2f%%)\n", max_comp, comp_pct);
    }

    free(local_result);
    free(local_matrix);
    free(vector);
    free(displs_gather);
    free(counts_gather);
    free(displs_scatter);
    free(counts_scatter);
    if (rank == 0) {
        free(result);
        free(matrix);
    }

    MPI_Finalize();
    return 0;
}
