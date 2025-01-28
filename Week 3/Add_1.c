#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    int N, M;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Root process reads N and M
    if (rank == 0) {
        printf("Enter the value of N: ");
        scanf("%d", &N);
        printf("Enter the value of M: ");
        scanf("%d", &M);
    }

    // Broadcast N and M to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int start = rank * M + 1;
    int end = (rank + 1) * M;

    printf("Process %d calculating from %d to %d\n", rank, start, end);

    for (int i = start; i <= end; i++) {
        if (i <= N / 2) {
            printf("Process %d computed square of %d: %d\n", rank, i, i * i);
        } else {
            printf("Process %d computed cube of %d: %d\n", rank, i, i * i * i);
        }
    }

    MPI_Finalize();
    return 0;
}
