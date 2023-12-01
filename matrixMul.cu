// This code is written while learning CUDA from YT.
// Author : Vidnyani Umathe
// Date   : 01/12/2023
// Matrix Multiplication

#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <functional>

// STL
using std::cout;
using std::generate;
using std::vector;

__global__ void matrixMul(const int *a, const int *b, int *c, int N)
{

    // compute row and column index for each thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // iterate over row and column and put the result in c
    c[row * N + col] = 0;
    for (int i = 0; i < N; i++)
    {
        c[row * N + col] += a[row * N + i] * b[i * N + col];
    }
}

int main()
{

    int N = 1 << 10; // 1024 x  1024 matrix size

    size_t bytes = N * N * sizeof(int);

    // host vector
    vector<int> h_a(N * N);
    vector<int> h_b(N * N);
    vector<int> h_c(N * N);

    // initialize matrices
    generate(h_a.begin(), h_a.end(), []()
             { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []()
             { return rand() % 100; });

    // allocate device memory
    int *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // copy data from host to device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = 32;

    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    // launch kernel
    matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, N);

    cout << "COMPLETED SUCCESSFULLY\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}