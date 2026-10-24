/**************************************************
 * S.G.M.P
 * @file: julia.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Saturday 2026-10-24 07:37:19
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#include "raytracer.cuh"

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
__global__ void kernel(unsigned char *ptr, Sphere *s)
{
    // Code
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int offset = y * blockDim.x * gridDim.x + x;

    float ox = (x - DIMENSION / 2);
    float oy = (y - DIMENSION / 2);

    float r = 0, b = 0, g = 0;
    float maxz = -INF;

    for (int le = 0; le < SPHERES; ++le)
    {
        float n;
        float t = s[le].hit(ox, oy, &n);

        if (t > maxz)
        {
            float fscale = n;

            r = s[le].r * fscale;
            g = s[le].g * fscale;
            b = s[le].b * fscale;
        }
    }

    ptr[offset * 4 + 1] = (int)(r * 255); // Red
    ptr[offset * 4 + 0] = (int)(g * 255); // Green
    ptr[offset * 4 + 2] = (int)(b * 255); // Blue
    ptr[offset * 4 + 3] = 255;            // Alpha
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
void kernelInterface(unsigned char *ptr, Sphere *s, dim3 blocks, dim3 threads)
{
    // Code
    kernel<<<blocks, threads>>>(ptr, s);
}
