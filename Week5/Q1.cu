#include <stdio.h>
#include <cuda.h>

#define N 1024  // Define the size of the vectors

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int idx = threadIdx.x;  // Get thread index within the block

    if (idx < n) {  // Ensure within bounds
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int *h_A, *h_B, *h_C;  // Host vectors
    int *d_A, *d_B, *d_C;  // Device vectors
    int size = N * sizeof(int);

    // Allocate host memory
    h_A = (int*)malloc(size);
    h_B = (int*)malloc(size);
    h_C = (int*)malloc(size);

    // Initialize vectors with random values
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block and N threads
    vectorAdd<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some values for verification
    printf("Sample Results:\n");
    for (int i = 0; i < 5; i++) {
        printf("%d + %d = %d\n", h_A[i], h_B[i], h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
