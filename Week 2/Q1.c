#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    char word[100] = "Hello";  // A sample word to toggle

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int word_length = strlen(word);  // Length of the word to toggle

    // Ensure the number of processes does not exceed the length of the word
    if (size > word_length) {
        if (rank == 0) {
            printf("The number of processes exceeds the word length. Exiting.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Each process toggles a specific letter based on its rank
    if (rank == 0) {
        MPI_Ssend(word, word_length + 1, MPI_CHAR, 1, 0, MPI_COMM_WORLD);  // Send initial word to process 1
    } else if (rank == size - 1) {
        MPI_Recv(word, word_length + 1, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive from previous process
        word[rank] = (isupper(word[rank])) ? tolower(word[rank]) : toupper(word[rank]);  // Toggle character
        MPI_Ssend(word, word_length + 1, MPI_CHAR, 0, 1, MPI_COMM_WORLD);  // Send back to process 0
    } else {
        MPI_Recv(word, word_length + 1, MPI_CHAR, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive from previous process
        word[rank] = (isupper(word[rank])) ? tolower(word[rank]) : toupper(word[rank]);  // Toggle character
        MPI_Ssend(word, word_length + 1, MPI_CHAR, rank + 1, 0, MPI_COMM_WORLD);  // Send to next process
    }

    if (rank == 0) {
        MPI_Recv(word, word_length + 1, MPI_CHAR, size - 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // Receive back the modified word
        printf("Final word: %s\n", word);
    }

    MPI_Finalize();
    return 0;
}
