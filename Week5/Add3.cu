#include <stdio.h>
#include <cuda.h>

#define N 10  // Number of elements in the array
#define THREADS_PER_BLOCK 256  // Maximum threads per block

// CUDA Kernel for Odd-Even Transposition Sort
__global__ void oddEvenSort(int *arr, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;  // Global thread index

    for (int phase = 0; phase < n; phase++) {
        if (phase % 2 == 0) {  // Even phase
            if (tid % 2 == 0 && tid + 1 < n) {
                if (arr[tid] > arr[tid + 1]) {
                    // Swap adjacent elements
                    int temp = arr[tid];
                    arr[tid] = arr[tid + 1];
                    arr[tid + 1] = temp;
                }
            }
        } else {  // Odd phase
            if (tid % 2 == 1 && tid + 1 < n) {
                if (arr[tid] > arr[tid + 1]) {
                    // Swap adjacent elements
                    int temp = arr[tid];
                    arr[tid] = arr[tid + 1];
                    arr[tid + 1] = temp;
                }
            }
        }
        __syncthreads();  // Synchronization after each phase
    }
}

int main() {
    int h_arr[N];  // Host array
    int *d_arr;  // Device array
    int size = N * sizeof(int);

    // Initialize array with random numbers
    printf("Original Array:\n");
    for (int i = 0; i < N; i++) {
        h_arr[i] = rand() % 100;  // Random numbers from 0-99
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Allocate device memory
    cudaMalloc((void**)&d_arr, size);

    // Copy data from host to device
    cudaMemcpy(d_arr, h_arr, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    oddEvenSort<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_arr, N);

    // Copy sorted array back to host
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Print sorted array
    printf("Sorted Array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}
