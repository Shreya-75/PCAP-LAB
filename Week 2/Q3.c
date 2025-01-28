#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size, N = 5;
    int arr[5] = {1, 2, 3, 4, 5};  // Example array of size N
    int result;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate a larger buffer for MPI_Bsend (at least 1024 bytes)
    int buffer[1024];  // Large enough buffer to hold the messages
    MPI_Buffer_attach(buffer, sizeof(buffer));

    // Root process (rank 0) sends array elements to slave processes
    if (rank == 0) {
        for (int i = 1; i < size; i++) {
            MPI_Bsend(&arr[i-1], 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // Send element to each process
        }
    } else {
        MPI_Recv(&result, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive element
        if (rank % 2 == 0) result = result * result;  // Even rank, square the element
        else result = result * result * result;  // Odd rank, cube the element
        printf("Process %d result: %d\n", rank, result);  // Print result
    }

    // Detach the buffer after usage
    MPI_Buffer_detach(&buffer, &size);

    MPI_Finalize();
    return 0;
}
