#include <stdio.h>
#include <cuda.h>

#define KERNEL_SIZE 5
#define N 1024  // Input size
#define THREADS_PER_BLOCK 256

// Constant memory for kernel (filter)
__constant__ float d_kernel[KERNEL_SIZE];

// 1D Convolution kernel using shared memory and synchronization
__global__ void conv1D(float *d_input, float *d_output, int inputSize) {
    __shared__ float tile[THREADS_PER_BLOCK + KERNEL_SIZE - 1];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int halo_index_left = gid - KERNEL_SIZE / 2;

    // Load data into shared memory with boundary checks
    if (halo_index_left >= 0 && halo_index_left < inputSize)
        tile[tid] = d_input[halo_index_left];
    else
        tile[tid] = 0;
    __syncthreads();

    // Perform convolution if within bounds
    if (gid < inputSize) {
        float sum = 0.0f;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum += tile[tid + k] * d_kernel[k];
        }
        d_output[gid] = sum;
    }
}

// Function to initialize input data
void initArray(float *arr, int size) {
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10 + 1; // Random values between 1 and 10
    }
}

int main() {
    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_input = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);
    float h_kernel[KERNEL_SIZE] = {0.2, 0.4, 0.6, 0.4, 0.2}; // Example filter

    // Initialize input data
    initArray(h_input, N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * sizeof(float)); // Copy to constant memory

    // Define grid and block size
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    conv1D<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Print some output values
    printf("Input and Output Samples:\n");
    for (int i = 0; i < 10; i++) {
        printf("h_input[%d] = %.2f, h_output[%d] = %.2f\n", i, h_input[i], i, h_output[i]);
    }

    // Free memory
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
