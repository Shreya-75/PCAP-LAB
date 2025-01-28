#include "mpi.h"
#include<stdio.h>
int pows(int x, int y){
    double d=1;
    // if(y==0){
    //     return 1;
    // }
    for(int i=0;i<y;i++){
        d*=x;
    }
    return d;
}
 int main(int argc, char* argv[])
 {
    int rank,size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    printf("Rank of this process is %d with power as %d \n",rank,pows(2,rank));
    MPI_Finalize();
    return 0;
 }