#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);                // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    int M, N, *array = NULL, *sub_array = NULL;
    double local_avg = 0.0, total_avg = 0.0;

    if (rank == 0) {
        // Root process inputs data
        printf("Enter the value of M (elements per process): ");
        scanf("%d", &M);
        N = size; // Number of processes

        // Allocate memory for the 1D array
        array = (int *)malloc(N * M * sizeof(int));
        printf("Enter %d elements for the array:\n", N * M);
        for (int i = 0; i < N * M; i++) {
            scanf("%d", &array[i]);
        }

        printf("\nRoot process: Array initialized with %d elements.\n", N * M);
        printf("Root process: Distributing %d elements to each process.\n", M);
    }

    // Broadcast the value of M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for sub-array on each process
    sub_array = (int *)malloc(M * sizeof(int));

    // Scatter the array to all processes
    MPI_Scatter(array, M, MPI_INT, sub_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Log received data in each process
    printf("Process %d: Received elements [", rank);
    for (int i = 0; i < M; i++) {
        printf("%d", sub_array[i]);
        if (i < M - 1) printf(", ");
    }
    printf("]. Calculating average.\n");

    // Calculate the local average
    int local_sum = 0;
    for (int i = 0; i < M; i++) {
        local_sum += sub_array[i];
    }
    local_avg = local_sum / (double)M;

    // Gather all averages at the root process
    double *averages = NULL;
    if (rank == 0) {
        averages = (double *)malloc(size * sizeof(double));
    }
    MPI_Gather(&local_avg, 1, MPI_DOUBLE, averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Root process calculates the total average
    if (rank == 0) {
        double sum_of_averages = 0.0;
        printf("\nRoot process: Received averages from all processes:\n");
        for (int i = 0; i < size; i++) {
            printf("Process %d: Average = %.2f\n", i, averages[i]);
            sum_of_averages += averages[i];
        }
        total_avg = sum_of_averages / size;
        printf("\nFinal Total Average = %.2f\n", total_avg);

        // Free allocated memory
        free(array);
        free(averages);
    }

    // Free allocated memory for sub-array
    free(sub_array);

    MPI_Finalize(); // Finalize MPI
    return 0;
}
