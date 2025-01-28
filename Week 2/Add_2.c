#include <stdio.h>
#include <mpi.h>

int factorial(int num) {
    int result = 1;
    for (int i = 1; i <= num; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char *argv[]) {
    int rank, size, N = 6;
    int result = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Each process calculates its part of the expression
    for (int i = rank; i < N; i += size) {
        if (i % 2 == 0) {
            result += factorial(i + 1);  // Even indices: factorial
        } else {
            result += (i + 1) * (i + 2) / 2;  // Odd indices: sum of numbers
        }
    }

    // Sum the results from all processes
    int total_result = 0;
    MPI_Reduce(&result, &total_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Root process prints the result
    if (rank == 0) {
        printf("Result of the expression: %d\n", total_result);
    }

    MPI_Finalize();
    return 0;
}
