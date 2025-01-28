#include<stdio.h>
#include<mpi.h>
#include<string.h>
#include<stdlib.h>

long factorial(int x){
    if(x == 0 || x == 1)
        return 1;

    return x * factorial(x-1);
}

int main(int argc, char *argv[]){
    int size, rank;
    int error_code;
    char error_string[MPI_MAX_ERROR_STRING];
    int error_string_length;
    long local_factorial = 0, sum_factorials = 0;

    MPI_Init(&argc, &argv);
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Comm world = MPI_COMM_NULL;

    local_factorial = factorial(rank + 1);
    printf("Process %d calculated local factorial: %ld\n", rank, local_factorial);

    // Perform the scan operation
    error_code = MPI_Scan(&local_factorial, &sum_factorials, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    if(error_code != MPI_SUCCESS){
        int error_class;
        MPI_Error_class(error_code, &error_class);
        MPI_Error_string(error_class, error_string, &error_string_length);
        fprintf(stderr, "This is a Custom error message .\n");
        fprintf(stderr, "Error code: %d\n", error_code);
        fprintf(stderr, "Error class: %d\n", error_class);
        fprintf(stderr, "Error string: %s\n", error_string);
    }
    else{
        // Print the sum of factorials only on the last process
        if(rank == size - 1)
            printf("Process %d: Sum of factorials from 1 to %d is: %ld\n", rank, size, sum_factorials);
    }

    MPI_Finalize();
    return 0;
}
