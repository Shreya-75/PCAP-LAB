#include <stdio.h>
#include <cuda.h>

#define ROWS 5   // Number of rows in the matrix
#define COLS 8   // Number of columns in the matrix

// CUDA Kernel to perform selection sort on each row
__global__ void selectionSortRows(int *matrix, int rows, int cols) {
    int row = blockIdx.x;  // Each block handles one row

    if (row < rows) {  
        int *row_ptr = &matrix[row * cols];  // Get pointer to the row

        // Selection Sort for the row
        for (int i = 0; i < cols - 1; i++) {
            int minIdx = i;
            for (int j = i + 1; j < cols; j++) {
                if (row_ptr[j] < row_ptr[minIdx]) {
                    minIdx = j;
                }
            }
            // Swap row_ptr[i] and row_ptr[minIdx]
            if (minIdx != i) {
                int temp = row_ptr[i];
                row_ptr[i] = row_ptr[minIdx];
                row_ptr[minIdx] = temp;
            }
        }
    }
}

int main() {
    int h_matrix[ROWS][COLS];  // Host matrix
    int *d_matrix;  // Device matrix
    int size = ROWS * COLS * sizeof(int);

    // Initialize matrix with random values
    printf("Original Matrix:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            h_matrix[i][j] = rand() % 100;  // Random numbers from 0-99
            printf("%2d ", h_matrix[i][j]);
        }
        printf("\n");
    }

    // Allocate device memory
    cudaMalloc((void**)&d_matrix, size);

    // Copy matrix from host to device
    cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);

    // Launch kernel with one block per row and COLS threads per block
    selectionSortRows<<<ROWS, 1>>>(d_matrix, ROWS, COLS);

    // Copy sorted matrix back to host
    cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);

    // Print sorted matrix
    printf("\nSorted Matrix (Each Row Sorted):\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            printf("%2d ", h_matrix[i][j]);
        }
        printf("\n");
    }

    // Free device memory
    cudaFree(d_matrix);

    return 0;
}
