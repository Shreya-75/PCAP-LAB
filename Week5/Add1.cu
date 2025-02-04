#include <stdio.h>
#include <cuda.h>

#define N 10000  // Vector size
#define THREADS_PER_BLOCK 256  // Fixed number of threads per block

// CUDA Kernel for the linear algebra operation: y = αx + y
__global__ void saxpy(float *x, float *y, float alpha, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Compute global index

    if (idx < n) {  // Ensure within bounds
        y[idx] = alpha * x[idx] + y[idx];  // Perform the operation
    }
}

int main() {
    float *h_x, *h_y;  // Host vectors
    float *d_x, *d_y;  // Device vectors
    float alpha = 2.5f;  // Scalar multiplier
    int size = N * sizeof(float);

    // Allocate host memory
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);

    // Initialize vectors
    for (int i = 0; i < N; i++) {
        h_x[i] = (float)(i % 100) * 0.1f;  // Some values for x
        h_y[i] = (float)(i % 50) * 0.2f;   // Some values for y
    }

    // Allocate device memory
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);

    // Copy data from host to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Calculate grid size (number of blocks)
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel with variable blocks and fixed 256 threads per block
    saxpy<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_x, d_y, alpha, N);

    // Copy result from device to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);

    // Print some sample results for verification
    printf("Sample Results (y = αx + y):\n");
    printf(" x[i]  | y[i] (updated)\n");
    printf("-------------------------\n");
    for (int i = 0; i < 5; i++) {
        printf("%.2f  | %.2f\n", h_x[i], h_y[i]);
    }

    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);

    // Free host memory
    free(h_x);
    free(h_y);

    return 0;
}
