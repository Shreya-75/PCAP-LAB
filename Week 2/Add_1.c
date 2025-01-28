#include <stdio.h>
#include <math.h>
#include <mpi.h>

int is_prime(int num) {
    if (num <= 1) return 0;
    for (int i = 2; i <= sqrt(num); i++) {
        if (num % i == 0) return 0;
    }
    return 1;
}

int main(int argc, char *argv[]) {
    int rank, size, N = 10;
    int arr[10] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11};  // Example array
    int local_arr[2], local_count = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide the work among processes
    int chunk_size = N / size;
    MPI_Scatter(arr, chunk_size, MPI_INT, local_arr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process checks the primality of its portion
    for (int i = 0; i < chunk_size; i++) {
        if (is_prime(local_arr[i])) local_count++;
    }

    // Gather the results from all processes
    int total_count = 0;
    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the total number of primes
    if (rank == 0) {
        printf("Total prime numbers: %d\n", total_count);
    }

    MPI_Finalize();
    return 0;
}
