#include <stdio.h>
#include <conio.h>
#include <cuda.h>
__global__ void even(int *darr, int n)
{
    int k = threadIdx.x;
    int temp;
    k = k * 2; // for even positions
    if (k < n - 1)
    {
        if (darr[k] > darr[k + 1])
        {
            temp = darr[k];
            darr[k] = darr[k + 1];
            darr[k + 1] = temp;
        }
    }
}

__global__ void odd(int *darr, int n)
{
    int k = threadIdx.x;
    int temp;
    k = k * 2 + 1; // for odd positions
    if (k < n - 1)
    {
        if (darr[k] > darr[k + 1])
        {
            temp = darr[k];
            darr[k] = darr[k + 1];
            darr[k + 1] = temp;
        }
    }
}
int main()
{
    int *arr, *darr; // declaring host and device arrays
    int n, i;

    printf("Enter the size of the array: \n");
    scanf("%d", &n);
    arr = (int *)malloc(sizeof(int) * n); // allocating memory in host

    printf("Enter the numbers in the array: \n");
    for (int i = 0; i < n; i++)
    {
        scanf("%d", &arr[i]);
    }

    cudaMalloc(&darr, n * sizeof(int));                            // allocating memory in device
    cudaMemcpy(darr, arr, n * sizeof(int), cudaMemcpyHostToDevice) // copying the array from host to device

        for (int i = 0; i < n / 2; i++)
    {
        even<<<1, n>>>(darr, n); // kernel for even positions
        odd<<<1, n>>>(darr, n);  // kernel for odd positions
    }
    cudaMemcpy(arr, darr, n * sizeof(int), cudaMemcpyDeviceToHost) // copying the array from device to host
        printf("Sorted Array: ");
    for (int i = 0; i < n; i++)
    {
        printf("%d  ", arr[i]);
    }

    free(arr);
    cudafree(darr);
    return 0;
}