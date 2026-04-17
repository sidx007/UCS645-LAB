#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static int is_perfect_number(int n) {
    if (n < 2) {
        return 0;
    }

    int sum = 1;
    for (int d = 2; d * d <= n; ++d) {
        if (n % d == 0) {
            sum += d;
            int other = n / d;
            if (other != d) {
                sum += other;
            }
        }
    }

    return sum == n;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int max_n = 10000;
    if (argc > 1) {
        max_n = atoi(argv[1]);
    }

    if (size < 2) {
        if (rank == 0) {
            printf("Q5 requires at least 2 MPI processes.\n");
        }
        MPI_Finalize();
        return 1;
    }

    if (max_n < 2) {
        if (rank == 0) {
            printf("Maximum value must be >= 2.\n");
        }
        MPI_Finalize();
        return 1;
    }

    const int tag_report = 1;
    const int tag_work = 2;
    const int stop_signal = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0) {
        char* perfect_flags = (char*)calloc((size_t)max_n + 1U, sizeof(char));
        int next_value = 2;
        int active_workers = size - 1;

        while (active_workers > 0) {
            int report = 0;
            MPI_Status status;
            MPI_Recv(&report, 1, MPI_INT, MPI_ANY_SOURCE, tag_report, MPI_COMM_WORLD, &status);
            int worker = status.MPI_SOURCE;

            if (report > 1 && report <= max_n) {
                perfect_flags[report] = 1;
            }

            if (next_value <= max_n) {
                MPI_Send(&next_value, 1, MPI_INT, worker, tag_work, MPI_COMM_WORLD);
                next_value++;
            } else {
                MPI_Send((void*)&stop_signal, 1, MPI_INT, worker, tag_work, MPI_COMM_WORLD);
                active_workers--;
            }
        }

        double elapsed = MPI_Wtime() - start;

        int count = 0;
        printf("Q5 Perfect numbers up to %d:\n", max_n);
        for (int n = 2; n <= max_n; ++n) {
            if (perfect_flags[n]) {
                printf("%d ", n);
                count++;
            }
        }
        printf("\nTotal perfect numbers found: %d\n", count);
        printf("Elapsed Time: %.6f s\n", elapsed);

        free(perfect_flags);
    } else {
        int report = 0;
        while (1) {
            MPI_Send(&report, 1, MPI_INT, 0, tag_report, MPI_COMM_WORLD);

            int number_to_test = 0;
            MPI_Recv(&number_to_test, 1, MPI_INT, 0, tag_work, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            if (number_to_test == stop_signal) {
                break;
            }

            report = is_perfect_number(number_to_test) ? number_to_test : -number_to_test;
        }
    }

    MPI_Finalize();
    return 0;
}
