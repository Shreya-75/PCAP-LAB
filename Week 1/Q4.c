#include "mpi.h"
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include <ctype.h>
void main(int argc, char** argv)
{
    char txt[]="Hello";
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    txt[rank]=tolower(txt[rank]);
    printf("char changed %c, rank %d string %s \n",txt[rank],rank, txt);
    MPI_Finalize();
}