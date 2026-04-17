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

    int n = 100;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        if (rank == 0) {
            printf("Array size must be positive.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int* counts = (int*)malloc((size_t)size * sizeof(int));
    int* displs = (int*)malloc((size_t)size * sizeof(int));
    build_counts_displs(n, size, counts, displs);

    int local_n = counts[rank];
    int* local_data = (int*)malloc((size_t)local_n * sizeof(int));

    int* data = NULL;
    if (rank == 0) {
        data = (int*)malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; ++i) {
            data[i] = i + 1;
        }
    }

    MPI_Scatterv(data, counts, displs, MPI_INT,
                 local_data, local_n, MPI_INT,
                 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < local_n; ++i) {
        local_sum += local_data[i];
    }

    long long global_sum = 0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long expected = (long long)n * (long long)(n + 1) / 2LL;
        double average = (double)global_sum / (double)n;
        printf("Global sum: %lld\n", global_sum);
        printf("Expected sum: %lld\n", expected);
        printf("Average: %.6f\n", average);
    }

    free(local_data);
    free(displs);
    free(counts);
    if (rank == 0) {
        free(data);
    }

    MPI_Finalize();
    return 0;
}
