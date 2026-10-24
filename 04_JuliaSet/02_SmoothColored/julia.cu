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
__device__ float juliaGPU(int x, int y)
{
    // Code
    const float scale = 1.5;
    float jx = scale * (float)(DIMENSION / 2 - x) / (DIMENSION / 2);
    float jy = scale * (float)(DIMENSION / 2 - y) / (DIMENSION / 2);

    float cRe = -0.8f, cIm = 0.156f;
    float zRe = jx, zIm = jy;
    const int MAX_ITER = 200;

    for (int n = 0; n < MAX_ITER; ++n)
    {
        float tRe = zRe * zRe - zIm * zIm + cRe;
        float tIm = 2.0f * zRe * zIm + cIm;
        zRe = tRe;
        zIm = tIm;

        float mag2 = zRe * zRe + zIm * zIm;
        if (mag2 > 4.0f)
        {
            // Smooth iteration count
            // log(log|z|) / log(2), using |z| = sqrt(mag2)
            // log|z| = 0.5 * log(mag2)
            float smooth = (float)(n + 1) - logf(0.5f * logf(mag2)) / logf(2.0f);
            return smooth;
        }
    }
    return -1.0f; // inside the set (never escaped)
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
__global__ void kernel(unsigned char *ptr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= DIMENSION || y >= DIMENSION)
        return;

    int offset = (y * DIMENSION + x) * 4;

    float nu = juliaGPU(x, y);

    if (nu < 0.0f)
    {
        // Interior: black
        ptr[offset + 0] = 0;
        ptr[offset + 1] = 0;
        ptr[offset + 2] = 0;
    }
    else
    {
        // Exterior: map smooth value to a palette
        // Normalize to [0,1] and apply a cosine palette
        float t = nu / 200.0f; // 200 = MAX_ITER
        float r = 0.5f + 0.5f * cosf(6.28318f * (t + 0.0f));
        float g = 0.5f + 0.5f * cosf(6.28318f * (t + 0.33f));
        float b = 0.5f + 0.5f * cosf(6.28318f * (t + 0.67f));

        ptr[offset + 0] = (unsigned char)(255.0f * r);
        ptr[offset + 1] = (unsigned char)(255.0f * g);
        ptr[offset + 2] = (unsigned char)(255.0f * b);
    }
    ptr[offset + 3] = 255;
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
