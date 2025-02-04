#include <stdio.h>
#include <cuda.h>
#include <math.h>

#define N 10000  // Number of angles (change as needed)
#define THREADS_PER_BLOCK 256  // Fixed number of threads per block

// CUDA Kernel to compute sine of angles
__global__ void computeSine(float *angles, float *sine_values, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // Compute global index

    if (idx < n) {  // Ensure within bounds
        sine_values[idx] = sinf(angles[idx]);  // Compute sine using CUDA's sinf function
    }
}

int main() {
    float *h_angles, *h_sine_values;  // Host arrays
    float *d_angles, *d_sine_values;  // Device arrays
    int size = N * sizeof(float);

    // Allocate host memory
    h_angles = (float*)malloc(size);
    h_sine_values = (float*)malloc(size);

    // Initialize angles in radians (e.g., 0 to 2π)
    for (int i = 0; i < N; i++) {
        h_angles[i] = ((float)i / N) * 2.0f * M_PI;  // Generating values between 0 and 2π
    }

    // Allocate device memory
    cudaMalloc((void**)&d_angles, size);
    cudaMalloc((void**)&d_sine_values, size);

    // Copy data from host to device
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    // Calculate grid size (number of blocks)
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel with dynamically determined blocks and fixed 256 threads per block
    computeSine<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_angles, d_sine_values, N);

    // Copy result from device to host
    cudaMemcpy(h_sine_values, d_sine_values, size, cudaMemcpyDeviceToHost);

    // Print some sample results for verification
    printf("Angle (radians) | Sine Value\n");
    printf("-----------------------------\n");
    for (int i = 0; i < 5; i++) {
        printf("%.4f        | %.4f\n", h_angles[i], h_sine_values[i]);
    }

    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_sine_values);

    // Free host memory
    free(h_angles);
    free(h_sine_values);

    return 0;
}
