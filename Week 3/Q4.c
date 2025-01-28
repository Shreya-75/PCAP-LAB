#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);                // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    char *S1 = NULL;         // Full string S1 (root process only)
    char *S2 = NULL;         // Full string S2 (root process only)
    char *sub_S1 = NULL;     // Substring of S1 for each process
    char *sub_S2 = NULL;     // Substring of S2 for each process
    char *sub_result = NULL; // Substring of the resultant string for each process
    char *result = NULL;     // Resultant interleaved string (root process only)

    int string_length, chunk_size;

    if (rank == 0) {
        // Root process reads input strings
        printf("Enter string S1: ");
        char temp1[1024];
        scanf("%s", temp1);
        printf("Enter string S2: ");
        char temp2[1024];
        scanf("%s", temp2);

        // Ensure strings are of the same length
        string_length = strlen(temp1);
        if (string_length != strlen(temp2)) {
            printf("Error: Strings must have the same length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Check if string length is divisible by the number of processes
        if (string_length % size != 0) {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory for the strings
        S1 = (char *)malloc((string_length + 1) * sizeof(char));
        S2 = (char *)malloc((string_length + 1) * sizeof(char));
        strcpy(S1, temp1);
        strcpy(S2, temp2);

        // Log the input strings and their length
        printf("\nRoot process: Received strings S1 = '%s' and S2 = '%s'.\n", S1, S2);
        printf("Root process: String length = %d. Dividing into %d chunks.\n", string_length, size);
    }

    // Broadcast the string length to all processes
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = string_length / size; // Compute chunk size for each process

    // Allocate memory for substrings
    sub_S1 = (char *)malloc((chunk_size + 1) * sizeof(char));
    sub_S2 = (char *)malloc((chunk_size + 1) * sizeof(char));
    sub_result = (char *)malloc((2 * chunk_size + 1) * sizeof(char));

    // Scatter strings S1 and S2 to all processes
    MPI_Scatter(S1, chunk_size, MPI_CHAR, sub_S1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, sub_S2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Add null terminators for safety
    sub_S1[chunk_size] = '\0';
    sub_S2[chunk_size] = '\0';

    // Log received substrings
    printf("Process %d: Received substring S1 = '%s' and S2 = '%s'.\n", rank, sub_S1, sub_S2);

    // Interleave characters from sub_S1 and sub_S2
    for (int i = 0; i < chunk_size; i++) {
        sub_result[2 * i] = sub_S1[i];
        sub_result[2 * i + 1] = sub_S2[i];
    }
    sub_result[2 * chunk_size] = '\0'; // Null terminate the resultant substring

    // Gather all resultant substrings at the root process
    if (rank == 0) {
        result = (char *)malloc((2 * string_length + 1) * sizeof(char));
    }
    MPI_Gather(sub_result, 2 * chunk_size, MPI_CHAR, result, 2 * chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Root process logs and prints the final resultant string
    if (rank == 0) {
        result[2 * string_length] = '\0'; // Null terminate the final result
        printf("\nRoot process: Received interleaved substrings from all processes.\n");
        printf("Final Resultant String = '%s'\n", result);

        // Free memory allocated in the root process
        free(S1);
        free(S2);
        free(result);
    }

    // Free memory allocated in all processes
    free(sub_S1);
    free(sub_S2);
    free(sub_result);

    MPI_Finalize(); // Finalize MPI
    return 0;
}
