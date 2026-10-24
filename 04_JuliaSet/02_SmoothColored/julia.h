/**************************************************
 * S.G.M.P
 * @file: julia.h
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Saturday 2026-10-24 07:37:26
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>

#define DIMENSION   1024 * 10

void kernelInterface(unsigned char *ptr, dim3);
