#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int number = 10;  // A sample number to send

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    
    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Get the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Master process (rank 0) sends the number to all slave processes
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Send(&number, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send number to each process
        }
    } else {
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive the number from rank 0
        printf("Process %d received number: %d\n", rank, number);  // Print the received number
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}

