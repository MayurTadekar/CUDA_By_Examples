/**************************************************
 * S.G.M.P
 * @file: code.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Friday 2026-10-23 21:59:05
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX 100000000

#define CHECKCUDAERROR(caller, functionCall)                                                            \
    {                                                                                                   \
        cudaError cuError = functionCall;                                                               \
        if (cudaSuccess != cuError)                                                                     \
        {                                                                                               \
            fprintf(stderr, "%s -> CUDA Error %s - %d", caller, cudaGetErrorString(cuError), __LINE__); \
            exit(1);                                                                                    \
        }                                                                                               \
    }

int a[MAX], b[MAX], c[MAX];

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
void add(int *a, int *b, int *c)
{
    // Code
    for (int tid = 0; tid < MAX; ++tid)
    {
        c[tid] = a[tid] + b[tid];
    }
}

/***********************************************************
 * @brief CUDA Kernel for add
 *
 * @param
 * @return
 ***********************************************************/
__global__ void dev_add(int *a, int *b, int *c)
{
    // Code
    int tid = blockIdx.x; // handle the data at this line

    if (tid < MAX)
        c[tid] = a[tid] + b[tid];
}

/***********************************************************
 * @brief Entry Point Function
 *
 * @param
 * @return
 ***********************************************************/
int main(void)
{
    // Code
    int *dev_a, *dev_b, *dev_c;

    CHECKCUDAERROR("main()", cudaMalloc((void **)&dev_a, sizeof(int) * MAX));
    CHECKCUDAERROR("main()", cudaMalloc((void **)&dev_b, sizeof(int) * MAX));
    CHECKCUDAERROR("main()", cudaMalloc((void **)&dev_c, sizeof(int) * MAX));

    for (int le = 0; le < MAX; ++le)
    {
        a[le] = -le;
        b[le] = le * le;
    }

    CHECKCUDAERROR("main()", cudaMemcpy(dev_a, a, sizeof(int) * MAX, cudaMemcpyHostToDevice));
    CHECKCUDAERROR("main()", cudaMemcpy(dev_b, b, sizeof(int) * MAX, cudaMemcpyHostToDevice));

    dev_add<<<MAX, 1>>>(dev_a, dev_b, dev_c);

    CHECKCUDAERROR("main()", cudaMemcpy(c, dev_c, sizeof(int) * MAX, cudaMemcpyDeviceToHost));

    // for (int le = 0; le < MAX; ++le)
    // {
    //     printf(" %d + %d = %d\n", a[le], b[le], c[le]);
    // }

    if (dev_a)
    {
        cudaFree(dev_a);
        dev_a = NULL;
    }

    if (dev_b)
    {
        cudaFree(dev_b);
        dev_b = NULL;
    }

    if (dev_c)
    {
        cudaFree(dev_c);
        dev_c = NULL;
    }

    return (0);
}
