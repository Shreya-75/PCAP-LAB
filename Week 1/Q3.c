#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size;
    int a, b;
    MPI_Init(&argc, &argv);
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // Only rank 0 asks for user input
    if (rank == 0) {
        printf("Enter the first number: ");
        scanf("%d", &a);
        printf("Enter the second number: ");
        scanf("%d", &b);
    }

    // Broadcast the values of 'a' and 'b' from rank 0 to all other processes
    MPI_Bcast(&a, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Rank 0 (Addition): %d\n", (a + b));  // Addition
    }
    else if (rank == 1) {
        printf("Rank 1 (Subtraction): %d\n", (a - b));  // Subtraction
    }
    else if (rank == 2) {
        printf("Rank 2 (Multiplication): %d\n", (a * b));  // Multiplication
    }
    else if (rank == 3) {
        if (b != 0) {
            printf("Rank 3 (Division): %d\n", (a / b));  // Division (check for division by zero)
        } else {
            printf("Rank 3 (Division): Error - Division by zero\n");
        }
    }
    MPI_Finalize();
    return 0;
}
