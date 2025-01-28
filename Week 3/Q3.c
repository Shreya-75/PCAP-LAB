#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// Function to count non-vowels in a substring
int count_non_vowels(const char *substring, int length) {
    int count = 0;
    for (int i = 0; i < length; i++) {
        char ch = tolower(substring[i]);
        if (isalpha(ch) && !(ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u')) {
            count++;
        }
    }
    return count;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);                // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get process rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get number of processes

    char *input_string = NULL; // Full string (root process only)
    char *substring = NULL;    // Substring for each process
    int string_length, chunk_size;
    int local_count = 0, total_non_vowels = 0;

    if (rank == 0) {
        // Root process reads the input string
        printf("Enter a string: ");
        char temp[1024];
        scanf("%s", temp);
        string_length = strlen(temp);

        // Check divisibility of string length by number of processes
        if (string_length % size != 0) {
            printf("Error: String length must be evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Allocate memory for the string
        input_string = (char *)malloc((string_length + 1) * sizeof(char));
        strcpy(input_string, temp);

        printf("\nRoot process: Received string '%s' of length %d.\n", input_string, string_length);
        printf("Root process: Dividing string into %d chunks of size %d.\n", size, string_length / size);
    }

    // Broadcast the string length and chunk size to all processes
    MPI_Bcast(&string_length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = string_length / size;

    // Allocate memory for substring
    substring = (char *)malloc((chunk_size + 1) * sizeof(char));

    // Scatter the string to all processes
    MPI_Scatter(input_string, chunk_size, MPI_CHAR, substring, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Add null terminator to substring for safety
    substring[chunk_size] = '\0';

    // Log received substring in each process
    printf("Process %d: Received substring '%s'. Calculating non-vowels.\n", rank, substring);

    // Calculate the number of non-vowels in the substring
    local_count = count_non_vowels(substring, chunk_size);

    // Gather all local counts to the root process
    int *counts = NULL;
    if (rank == 0) {
        counts = (int *)malloc(size * sizeof(int));
    }
    MPI_Gather(&local_count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Root process computes the total non-vowels and prints results
    if (rank == 0) {
        printf("\nRoot process: Received non-vowel counts from all processes:\n");
        total_non_vowels = 0;
        for (int i = 0; i < size; i++) {
            printf("Process %d: Non-vowel count = %d\n", i, counts[i]);
            total_non_vowels += counts[i];
        }
        printf("\nTotal Number of Non-Vowels = %d\n", total_non_vowels);

        // Free allocated memory
        free(input_string);
        free(counts);
    }

    // Free memory for substring
    free(substring);

    MPI_Finalize(); // Finalize MPI
    return 0;
}
