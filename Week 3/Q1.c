#include <mpi.h>
#include <stdio.h>

// Function to calculate factorial
int factorial(int n) {
    if (n == 0 || n == 1) return 1;
    return n * factorial(n - 1);
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Root process
        int N = size - 1; // Number of worker processes
        int data[N];
        int results[N];

        // Prepare numbers to send
        for (int i = 0; i < N; i++) {
            data[i] = i + 1; // Assign numbers 1 to N
        }

        // Send one number to each worker process
        for (int i = 1; i <= N; i++) {
            printf("Root process: Sending %d to process %d.\n", data[i - 1], i);
            MPI_Send(&data[i - 1], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        // Receive results from worker processes
        int sum = 0;
        for (int i = 1; i <= N; i++) {
            MPI_Recv(&results[i - 1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Root process: Received factorial %d from process %d.\n", results[i - 1], i);
            sum += results[i - 1];
        }

        // Display final results
        printf("\nFinal Result: Factorials = [");
        for (int i = 0; i < N; i++) {
            printf("%d", results[i]);
            if (i < N - 1) printf(", ");
        }
        printf("]\n");
        printf("Sum of Factorials: %d\n", sum);

    } else {
        // Worker process
        int number;
        MPI_Recv(&number, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: Received number %d. Calculating factorial.\n", rank, number);

        // Calculate factorial
        int fact = factorial(number);

        // Send result back to the root process
        MPI_Send(&fact, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
        printf("Process %d: Sent factorial %d back to root.\n", rank, fact);
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}
