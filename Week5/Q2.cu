#include <stdio.h>
#include <cuda.h>

#define N 10000  // Vector size (change as needed)
#define THREADS_PER_BLOCK 256  // Fixed number of threads per block

// CUDA Kernel for vector addition
__global__ void vectorAdd(int *A, int *B, int *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Global index

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

    // Calculate grid size (number of blocks)
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel with variable blocks and fixed 256 threads per block
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

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
