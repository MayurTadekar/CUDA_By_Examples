/**************************************************
 * S.G.M.P
 * @file: julia.h
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Saturday 2026-10-24 07:37:26
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#define DIMENSION 1024
#define INF 2e10f
#define SPHERES 20

struct Sphere
{
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float *n)
    {
        float dx = ox - x;
        float dy = oy - y;
        if (dx * dx + dy * dy < radius * radius)
        {
            float dz = sqrtf(radius * radius - dx * dx - dy * dy);
            *n = dz / sqrtf(radius * radius);
            return (dz + z);
        }

        return (-INF);
    }
};

void kernelInterface(unsigned char *ptr, Sphere *s, dim3, dim3);
