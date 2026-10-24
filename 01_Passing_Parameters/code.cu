/**************************************************
 * S.G.M.P
 * @file: code.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Friday 2026-10-23 20:43:32
 **************************************************/

#include <cuda.h>
#include <stdio.h>

__global__ void add(int a, int b, int *c)
{
    // Code
    *c = a + b;
}

int main(void)
{
    // Code
    cudaError_t cuError;
    int c;
    int *dev_c;

    cuError = cudaMalloc((void **)&dev_c, sizeof(int));
    if (cudaSuccess != cuError)
    {
        fprintf(stderr, "main() => CUDA Error: %s\n", cudaGetErrorString(cuError));
        exit(1);
    }

    add<<<1, 1>>>(10, 20, dev_c);

    cuError = cudaMemcpy(&c, dev_c, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaSuccess != cuError)
    {
        fprintf(stderr, "main() => CUDA Error: %s\n", cudaGetErrorString(cuError));
        exit(1);
    }

    printf("From Device => 10 + 20 = %d", c);

    if (dev_c)
    {
        cudaFree(dev_c);
        dev_c = NULL;
    }

    return (0);
}
