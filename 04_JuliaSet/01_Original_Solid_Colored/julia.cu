/**************************************************
 * S.G.M.P
 * @file: julia.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Saturday 2026-10-24 07:37:19
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#include "julia.h"

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
__device__ int juliaGPU(int x, int y)
{
    // Code
    const float scale = 1.5;
    float jx = scale * (float)(DIMENSION / 2 - x) / (DIMENSION / 2);
    float jy = scale * (float)(DIMENSION / 2 - y) / (DIMENSION / 2);

    float c[2] = {-0.8, 0.156};
    float a[2] = {jx, jy};

    for (int le = 0; le < 200; ++le)
    {
        float b[2] = {a[0] * a[0] - a[1] * a[1], a[1] * a[0] + a[0] * a[1]};
        float d[2] = {b[0] + c[0], b[1] + c[1]};

        a[0] = d[0];
        a[1] = d[1];

        float mag = a[0] * a[0] + a[1] * a[1];

        if (mag > 4)
            return (0);
    }

    return (1);
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
__global__ void kernel(unsigned char *ptr)
{
    // Code
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int offset = y * gridDim.x + x;

    // Now calculate the value at that position
    int juliaValue = juliaGPU(x, y);
    ptr[offset * 4 + 1] = 255 * juliaValue; // Red
    ptr[offset * 4 + 0] = 0;                // Green
    ptr[offset * 4 + 2] = 0;                // Blue
    ptr[offset * 4 + 3] = 255;              // Alpha
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
void kernelInterface(unsigned char *ptr, dim3 grid)
{
    // Code
    kernel<<<grid, 1>>>(ptr);
}
