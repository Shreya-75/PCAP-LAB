#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>

// Function to check if a number is prime
bool is_prime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i * i <= num; i++) {
        if (num % i == 0) return false;
    }
    return true;
}

int main(int argc, char *argv[]) {
    int rank, size;
    int start, end;
    int primes[50];  // To store prime numbers for each process
    int prime_count = 0; // To count the primes found by each process

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Ensure there are exactly 2 processes
    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly 2 processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Define the range for each process
    if (rank == 0) {
        start = 1;
        end = 50;
    } else {
        start = 51;
        end = 100;
    }

    // Find primes in the given range for each process
    for (int i = start; i <= end; i++) {
        if (is_prime(i)) {
            primes[prime_count++] = i;
        }
    }

    // Prepare to gather all primes at rank 0
    int total_prime_count = 0;
    MPI_Reduce(&prime_count, &total_prime_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 gathers the prime numbers from both processes
    int all_primes[total_prime_count];
    if (rank == 0) {
        // Gather the prime numbers found by process 0 and process 1
        int counts[2] = {prime_count, total_prime_count - prime_count}; // Process 0 and process 1 primes count
        MPI_Gather(primes, prime_count, MPI_INT, all_primes, prime_count, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(primes + prime_count, counts[1], MPI_INT, all_primes + prime_count, counts[1], MPI_INT, 0, MPI_COMM_WORLD);

        // Print the prime numbers at rank 0
        printf("Prime numbers between 1 and 100:\n");
        for (int i = 0; i < total_prime_count; i++) {
            printf("%d ", all_primes[i]);
        }
        printf("\n");
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
