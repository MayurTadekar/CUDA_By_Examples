/**************************************************
 * S.G.M.P
 * @file: julia.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Saturday 2026-10-24 07:37:19
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#include "ripple.h"

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
__global__ void kernel(unsigned char *ptr, int ticks)
{
    // Code
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    // calculate the value at the position
    float fx = x - DIMENSION / 2;
    float fy = y - DIMENSION / 2;
    float d = sqrt(fx * fx + fy * fy);

    unsigned char grey = (unsigned char)(128.0f + 127.0f * cos(d / 10.0f - ticks / 7.0f) / (d / 10.0f + 1.0f));

    ptr[offset * 4 + 0] = 0;
    ptr[offset * 4 + 1] = grey * 0.5;
    ptr[offset * 4 + 2] = grey;
    ptr[offset * 4 + 3] = 255;
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
void kernelInterface(unsigned char *ptr, int ticks, dim3 blocks, dim3 threads)
{
    // Code
    kernel<<<blocks, threads>>>(ptr, ticks);
}
