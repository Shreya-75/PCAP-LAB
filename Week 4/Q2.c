#include <stdio.h>
#include <mpi.h>

#define SIZE 3

int main(int argc, char *argv[]) {
    int rank, size;
    int matrix[SIZE][SIZE];
    int search_element;
    int local_count = 0, global_count = 0;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process enters the matrix and the search element
        printf("Process %d: Enter a 3x3 matrix (9 elements):\n", rank);
        for (int i = 0; i < SIZE; i++) {
            for (int j = 0; j < SIZE; j++) {
                scanf("%d", &matrix[i][j]);
            }
        }
        printf("Process %d: Enter the element to search: ", rank);
        scanf("%d", &search_element);
    }

    // Broadcast the matrix and search element to all processes
    MPI_Bcast(&matrix, SIZE * SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&search_element, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Process %d: Received matrix and search element %d\n", rank, search_element);

    // Each process checks a different part of the matrix
    for (int i = rank; i < SIZE; i += size) {
        for (int j = 0; j < SIZE; j++) {
            if (matrix[i][j] == search_element) {
                local_count++;
                printf("Process %d: Found element %d at position (%d, %d)\n", rank, search_element, i, j);
            }
        }
    }

    // Reduce local counts to get the global count
    MPI_Reduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Process %d: The element %d occurred %d times in the matrix.\n", rank, search_element, global_count);
    }

    MPI_Finalize();
    return 0;
}
