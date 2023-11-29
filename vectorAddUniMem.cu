// This code is written while learning CUDA from youtube
// program to compute the sum of two vectors of length n using unified memory (virtual memory) where we dont
// need to call cudaMemcpy expxlicitly cudaMallocManaged handles it automatically

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

// Kernel for vector addition

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{

    // to calculate thread ID
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (tid < n)
    {
        c[tid] = a[tid] + b[tid];
    }
}

// initialize vector of size n between 0-99

void matrix_init(int *a, int *b, int n)
{
    for (int i = 0; i < n; i++)
    {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }
}

int main()
{
    int n = 1 << 16; // vector of size 2^16 (65536 elements)

    size_t bytes = sizeof(int) * n; // allocation size for all vectors

    int *a, *b, *c;

    // Allocation of memory

    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // initializing vectors a and b with random values between 0 and 99
    matrix_init(a, b, n);

    // block size
    int BLOCK_SIZE = 256;

    // grid size
    int GRID_SIZE = (int)ceil(n / BLOCK_SIZE);

    // launch kernel
    vectorAdd<<<BLOCK_SIZE, GRID_SIZE>>>(a, b, c, n);

    // wait for all the previous operations before using values
    cudaDeviceSynchronize();

    printf("Completed Successfully!\n");

    return 0;
}