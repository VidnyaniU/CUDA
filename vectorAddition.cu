//This code is written while learning CUDA from youtube
// program to compute the sum of two vectors of length n

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

//initialize vector of size n between 0-99

void matrix_init(int*a , int n){
    for (int i = 0; i < n; i++)
    {
        a[i]= rand()%100;
    }
    
}

int main()
{
    int n = 1 << 16; // vector of size 2^16 (65536 elements)

    int *h_a, *h_b, *h_c; // host vector pointers

    int *d_a, *d_b, *d_c; // device vector pointers

    size_t bytes = sizeof(int)*n; //allocation size for all vectors

    //allocating host memory (CPU)
    h_a = (int*)malloc(bytes);
    h_b = (int*)malloc (bytes);
    h_c = (int*)malloc (bytes);

    //allocating device memory (GPU)
    cudaMalloc(&d_a,bytes);
    cudaMalloc(&d_b,bytes);
    cudaMalloc(&d_c,bytes);

    //initializing vectors a and b with random values between 0 and 99
    matrix_init(h_a,n);
    matrix_init(h_b,n);

    //copy data from host to device
    cudaMemcpy(d_a,h_a,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,bytes,cudaMemcpyHostToDevice);

    //threadblock size
    int NUM_THREADS = 256;

    //grid size
    int NUM_BLOCKS = (int)ceil(n/NUM_THREADS);

    //launch kernel
    vectorAdd<<<NUM_BLOCKS,NUM_THREADS>>>(d_a,d_b,d_c,n);

    cudaMemcpy(h_c,d_c ,bytes ,cudaMemcpyDeviceToHost);

    printf("Completed Successfully!\n");

    return 0;


}