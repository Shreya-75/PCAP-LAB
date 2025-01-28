#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to reverse digits of a number
int reverse(int num) {
    int rev = 0;
    while (num != 0) {
        rev = rev * 10 + num % 10;
        num /= 10;
    }
    return rev;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int array[9] = {18, 523, 301, 1234, 2, 14, 108, 150, 1928};
    int result[9];
    int local_value, reversed_value;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 9) {
        if (rank == 0) {
            printf("This program requires exactly 9 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Each process works on one element of the array
    local_value = array[rank];
    reversed_value = reverse(local_value);

    // Gather the reversed values into the result array
    MPI_Gather(&reversed_value, 1, MPI_INT, result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Print the result at rank 0
    if (rank == 0) {
        printf("Output Array (Reversed Digits):\n");
        for (int i = 0; i < 9; i++) {
            printf("%d ", result[i]);
        }
        printf("\n");
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
