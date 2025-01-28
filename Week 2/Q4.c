#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int value = 0;  // The initial value to be passed along

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Process 0 sends the initial value to process 1
    if (rank == 0) {
        value = 0;  // Set the initial value
        MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);  // Send to process 1
    } else if (rank == size - 1) {
        // Last process receives the value and sends it back to process 0
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value++;  // Increment the value
        printf("Process %d received value %d and sending it back to process 0\n", rank, value);
        MPI_Send(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);  // Send back to process 0
    } else {
        // Intermediate process receives, increments, and sends to the next process
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        value++;  // Increment the value
        printf("Process %d received value %d and sending it to process %d\n", rank, value, rank + 1);
        MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);  // Send to the next process
    }

    // Process 0 receives the final value
    if (rank == 0) {
        MPI_Recv(&value, 1, MPI_INT, size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process 0 received final value: %d\n", value);
    }

    MPI_Finalize();
    return 0;
}
