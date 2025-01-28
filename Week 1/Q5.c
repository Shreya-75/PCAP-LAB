#include "mpi.h"
#include <stdio.h>

// Function to compute the factorial of a number
long long factorial(int n) {
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Function to compute the Fibonacci number of a given rank
long long fibonacci(int n) {
    if (n <= 1) return n;
    long long a = 0, b = 1, temp;
    for (int i = 2; i <= n; i++) {
        temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

int main(int argc, char *argv[]) {
    int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank of the process and the size (number of processes)
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Even ranked processes print the factorial of their rank
    if (rank % 2 == 0) {
        long long fact = factorial(rank);
        printf("Rank %d (Even) - Factorial: %lld\n", rank, fact);
    }
    // Odd ranked processes print the Fibonacci number of their rank
    else {
        long long fib = fibonacci(rank);
        printf("Rank %d (Odd) - Fibonacci: %lld\n", rank, fib);
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
 