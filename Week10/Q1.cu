#include <stdio.h>
#include <cuda.h>

#define N 4  // Matrix size (N x N)
#define TILE_SIZE 2 // Block size for tiling

// Constant memory for small matrices
__constant__ float d_B[N * N];

// Kernel for matrix multiplication using shared memory and synchronization
__global__ void matrixMulKernel(float *A, float *C, int n) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    for (int t = 0; t < n / TILE_SIZE; t++) {
        if (row < n && (t * TILE_SIZE + threadIdx.x) < n)
            tile_A[threadIdx.y][threadIdx.x] = A[row * n + (t * TILE_SIZE + threadIdx.x)];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        if (row < n && col < n) {
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += tile_A[threadIdx.y][k] * d_B[(t * TILE_SIZE + k) * n + col];
            }
        }
        __syncthreads();
    }

    if (row < n && col < n)
        C[row * n + col] = sum;
}

// Function to initialize matrix with random values
void initMatrix(float *mat, int n) {
    for (int i = 0; i < n * n; i++) {
        mat[i] = rand() % 10; // Random values between 0 and 9
    }
}

// Function to print matrix
void printMatrix(float *mat, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.2f ", mat[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    
    // Initialize matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);
    
    // Allocate device memory
    float *d_A, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_C, bytes);
    
    // Copy data from host to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_B, h_B, bytes); // Copy B to constant memory
    
    // Define grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch the kernel
    matrixMulKernel<<<gridDim, blockDim>>>(d_A, d_C, N);
    
    // Copy result from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    
    // Print results
    printf("Matrix A:\n");
    printMatrix(h_A, N);
    printf("\nMatrix B:\n");
    printMatrix(h_B, N);
    printf("\nResultant Matrix C:\n");
    printMatrix(h_C, N);
    
    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_C);
    
    return 0;
}
