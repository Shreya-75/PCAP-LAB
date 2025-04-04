#include <stdio.h>
#include <cuda.h>

#define N 1024  // Input size
#define THREADS_PER_BLOCK 256

// Kernel for inclusive scan using shared memory
__global__ void inclusiveScan(float *d_input, float *d_output, int size) {
    __shared__ float temp[THREADS_PER_BLOCK];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (gid < size)
        temp[tid] = d_input[gid];
    else
        temp[tid] = 0;
    __syncthreads();

    // Inclusive scan using work-efficient algorithm
    for (int offset = 1; offset < blockDim.x; offset *= 2) {
        float val = 0;
        if (tid >= offset)
            val = temp[tid - offset];
        __syncthreads();
        temp[tid] += val;
        __syncthreads();
    }

    // Write result to global memory
    if (gid < size)
        d_output[gid] = temp[tid];
}

// Function to initialize input array
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

    // Initialize input array
    initArray(h_input, N);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    // Define grid and block sizes
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    inclusiveScan<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost);

    // Print some output values
    printf("Input and Inclusive Scan Output (First 10 Elements):\n");
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