/**************************************************
 * S.G.M.P
 * @file: hello.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Friday 2026-10-23 20:38:53
 **************************************************/

#include <cuda.h>
#include <stdio.h>

__global__ void kernel(void)
{
    printf("Hello, World From GPU!!!\n");
}

int main(void)
{
    // Code

    kernel<<<1, 1>>>();

    printf("Hello, World!!!\n");

    return (0);
}
