#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N;
    int *A = NULL, *local_A = NULL, *even_count = NULL, *odd_count = NULL;
    int local_even_count = 0, local_odd_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads the array and initializes the size
    if (rank == 0) {
        printf("Enter the value of N: ");
        scanf("%d", &N);
        A = (int *)malloc(N * sizeof(int));
        printf("Enter %d elements of array A: ", N);
        for (int i = 0; i < N; i++) {
            scanf("%d", &A[i]);
        }
    }

    // Scatter the array to all processes
    local_A = (int *)malloc(N / size * sizeof(int));  // Each process gets N/size elements
    MPI_Scatter(A, N / size, MPI_INT, local_A, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process replaces even with 0 and odd with 1, and counts even/odd elements
    for (int i = 0; i < N / size; i++) {
        if (local_A[i] % 2 == 0) {
            local_A[i] = 0;
            local_even_count++;
        } else {
            local_A[i] = 1;
            local_odd_count++;
        }
    }

    // Gather the modified array back to root
    MPI_Gather(local_A, N / size, MPI_INT, A, N / size, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process prints the result and counts
    if (rank == 0) {
        printf("Modified Array: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", A[i]);
        }
        printf("\n");

        // Gather the count of even and odd elements from all processes
        even_count = (int *)malloc(size * sizeof(int));
        odd_count = (int *)malloc(size * sizeof(int));
        MPI_Gather(&local_even_count, 1, MPI_INT, even_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&local_odd_count, 1, MPI_INT, odd_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

        int total_even_count = 0, total_odd_count = 0;
        for (int i = 0; i < size; i++) {
            total_even_count += even_count[i];
            total_odd_count += odd_count[i];
        }

        printf("Total even count: %d\n", total_even_count);
        printf("Total odd count: %d\n", total_odd_count);

        // Clean up
        free(A);
        free(even_count);
        free(odd_count);
    }

    free(local_A);
    MPI_Finalize();
    return 0;
}
