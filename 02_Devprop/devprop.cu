/**************************************************
 * S.G.M.P
 * @file: devprop.cu
 * @brief:
 * @author: Mayur Tadekar(mrtadekar@gmail.com)
 * @date: Friday 2026-10-23 20:52:53
 **************************************************/

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cstdio>
#include <cuda_runtime.h>

#define CHECKCUDAERROR(caller, functionCall)                                             \
    {                                                                                    \
        if (cudaSuccess != functionCall)                                                 \
        {                                                                                \
            fprintf(stderr, "%s -> CUDA Error %s", caller, cudaGetErrorString(cuError)); \
            exit(1);                                                                     \
        }                                                                                \
    }

/***********************************************************
 * @brief Prints the size of bytes in bytes, kb, mb, gb
 *
 * @param bytes
 * @return void
 ***********************************************************/
void printSizes(size_t bytes)
{
    // Code
    size_t _bytes = bytes;
    double _kb = (double)_bytes / 1024.0;
    double _mb = _kb / 1024.0;
    double _gb = _mb / 1024.0;

    printf("%zu bytes - ", _bytes);
    if (0.0 != _kb)
        printf("%0.2lf KB - ", _kb);
    if (0.0 != _mb)
        printf("%0.2lf MB - ", _mb);
    if (0.0 != _gb)
        printf("%0.2lf GB\n", _gb);
}

/***********************************************************
 * @brief
 *
 * @param
 * @return
 ***********************************************************/
int getCoresPerSM(int major, int minor)
{
    // Code
    switch (major)
    {
    case 2:
        return (32); // Fermi
    case 3:
        return (192); // Kepler
    case 5:
        return (128); // Maxwell
    case 6:
        return (minor == 0 ? 64 : 128); // Pascal
    case 7:
        return (64); // Volta / Turing
    case 8:
        return (minor == 0 ? 64 : 128); // Ampere
    case 9:
        return (128); // Hopper
    case 10:
        return (128); // Blackwell (B100/B200)
    case 12:
        return (128); // Blackwell (RTX 50xx)
    default:
        return (-1);
    }
}

/***********************************************************
 * @brief Prints all the DeviceProperties of a CUDA Device
 *
 * @param cudaDeviceProp
 * @return void
 ***********************************************************/
void printAllDevProp(const cudaDeviceProp &d)
{
    printf("\n=== Device Properties ===\n");

    // Identity
    printf("\n=== Identity ===\n");
    printf("name:                          %s\n", d.name);
    printf("uuid:                          ");
    for (int i = 0; i < 16; i++)
        printf("%02x", d.uuid.bytes[i]);
    printf("\n");
    printf("luid:                          ");
    for (int i = 0; i < 8; i++)
        printf("%02x", d.luid[i]);
    printf("\n");
    printf("luidDeviceNodeMask:            0x%08x\n", d.luidDeviceNodeMask);

    // Memory
    printf("\n=== Memory ===\n");
    printf("totalGlobalMem:                ");
    printSizes(d.totalGlobalMem);
    printf("sharedMemPerBlock:             ");
    printSizes(d.sharedMemPerBlock);
    printf("sharedMemPerMultiprocessor:    ");
    printSizes(d.sharedMemPerMultiprocessor);
    printf("sharedMemPerBlockOptin:        ");
    printSizes(d.sharedMemPerBlockOptin);
    printf("reservedSharedMemPerBlock:     ");
    printSizes(d.reservedSharedMemPerBlock);
    printf("totalConstMem:                 ");
    printSizes(d.totalConstMem);
    printf("memPitch:                      ");
    printSizes(d.memPitch);
    printf("memoryBusWidth:                %d bits\n", d.memoryBusWidth);
    printf("l2CacheSize:                   %d bytes\n", d.l2CacheSize);
    printf("persistingL2CacheMaxSize:      %d bytes\n", d.persistingL2CacheMaxSize);

    // Compute
    int coresPerSM = getCoresPerSM(d.major, d.minor);
    int totalCores = d.multiProcessorCount * coresPerSM;
    printf("\n=== Compute ===\n");
    printf("compute capability:            %d.%d\n", d.major, d.minor);
    printf("SMs: %d, Cores/SM: %d, Total CUDA Cores: %d\n", d.multiProcessorCount, coresPerSM, totalCores);
    printf("multiProcessorCount:           %d\n", d.multiProcessorCount);
    printf("maxThreadsPerBlock:            %d\n", d.maxThreadsPerBlock);
    printf("maxThreadsPerMultiProcessor:   %d\n", d.maxThreadsPerMultiProcessor);
    printf("maxBlocksPerMultiProcessor:    %d\n", d.maxBlocksPerMultiProcessor);
    printf("warpSize:                      %d\n", d.warpSize);
    printf("regsPerBlock:                  %d\n", d.regsPerBlock);
    printf("regsPerMultiprocessor:         %d\n", d.regsPerMultiprocessor);
    printf("asyncEngineCount:              %d\n", d.asyncEngineCount);

    // Grid/Block dimensions
    printf("\n=== Grid/Block dimensions ===\n");
    printf("maxThreadsDim:                 [%d, %d, %d]\n", d.maxThreadsDim[0], d.maxThreadsDim[1], d.maxThreadsDim[2]);
    printf("maxGridSize:                   [%d, %d, %d]\n", d.maxGridSize[0], d.maxGridSize[1], d.maxGridSize[2]);

    // Texture limits
    printf("\n=== Texture limits ===\n");
    printf("maxTexture1D:                  %d\n", d.maxTexture1D);
    printf("maxTexture1DMipmap:            %d\n", d.maxTexture1DMipmap);
    printf("maxTexture2D:                  [%d, %d]\n", d.maxTexture2D[0], d.maxTexture2D[1]);
    printf("maxTexture2DMipmap:            [%d, %d]\n", d.maxTexture2DMipmap[0], d.maxTexture2DMipmap[1]);
    printf("maxTexture2DLinear:            [%d, %d, %d]\n", d.maxTexture2DLinear[0], d.maxTexture2DLinear[1], d.maxTexture2DLinear[2]);
    printf("maxTexture2DGather:            [%d, %d]\n", d.maxTexture2DGather[0], d.maxTexture2DGather[1]);
    printf("maxTexture3D:                  [%d, %d, %d]\n", d.maxTexture3D[0], d.maxTexture3D[1], d.maxTexture3D[2]);
    printf("maxTexture3DAlt:               [%d, %d, %d]\n", d.maxTexture3DAlt[0], d.maxTexture3DAlt[1], d.maxTexture3DAlt[2]);
    printf("maxTextureCubemap:             %d\n", d.maxTextureCubemap);
    printf("maxTexture1DLayered:           [%d, %d]\n", d.maxTexture1DLayered[0], d.maxTexture1DLayered[1]);
    printf("maxTexture2DLayered:           [%d, %d, %d]\n", d.maxTexture2DLayered[0], d.maxTexture2DLayered[1], d.maxTexture2DLayered[2]);
    printf("maxTextureCubemapLayered:      [%d, %d]\n", d.maxTextureCubemapLayered[0], d.maxTextureCubemapLayered[1]);
    printf("textureAlignment:              %zu\n", d.textureAlignment);
    printf("texturePitchAlignment:         %zu\n", d.texturePitchAlignment);

    // Surface limits
    printf("\n=== Surface limits ===\n");
    printf("maxSurface1D:                  %d\n", d.maxSurface1D);
    printf("maxSurface2D:                  [%d, %d]\n", d.maxSurface2D[0], d.maxSurface2D[1]);
    printf("maxSurface3D:                  [%d, %d, %d]\n", d.maxSurface3D[0], d.maxSurface3D[1], d.maxSurface3D[2]);
    printf("maxSurface1DLayered:           [%d, %d]\n", d.maxSurface1DLayered[0], d.maxSurface1DLayered[1]);
    printf("maxSurface2DLayered:           [%d, %d, %d]\n", d.maxSurface2DLayered[0], d.maxSurface2DLayered[1], d.maxSurface2DLayered[2]);
    printf("maxSurfaceCubemap:             %d\n", d.maxSurfaceCubemap);
    printf("maxSurfaceCubemapLayered:      [%d, %d]\n", d.maxSurfaceCubemapLayered[0], d.maxSurfaceCubemapLayered[1]);
    printf("surfaceAlignment:              %zu\n", d.surfaceAlignment);

    // PCI / Topology
    printf("\n=== PCI / Topology ===\n");
    printf("pciBusID:                      %d\n", d.pciBusID);
    printf("pciDeviceID:                   %d\n", d.pciDeviceID);
    printf("pciDomainID:                   %d\n", d.pciDomainID);
    printf("gpuPciDeviceID:                0x%08x\n", d.gpuPciDeviceID);
    printf("gpuPciSubsystemID:             0x%08x\n", d.gpuPciSubsystemID);
    printf("tccDriver:                     %d\n", d.tccDriver);
    printf("integrated:                    %d\n", d.integrated);
    printf("isMultiGpuBoard:               %d\n", d.isMultiGpuBoard);
    printf("multiGpuBoardGroupID:          %d\n", d.multiGpuBoardGroupID);

    // NUMA
    printf("\n=== NUMA ===\n");
    printf("deviceNumaConfig:              %d\n", d.deviceNumaConfig);
    printf("deviceNumaId:                  %d\n", d.deviceNumaId);
    printf("hostNumaId:                    %d\n", d.hostNumaId);
    printf("hostNumaMultinodeIpcSupported: %d\n", d.hostNumaMultinodeIpcSupported);

    // Features (0/1 flags)
    printf("\n=== Features (0/1 flags) ===\n");
    printf("ECCEnabled:                    %d\n", d.ECCEnabled);
    printf("unifiedAddressing:             %d\n", d.unifiedAddressing);
    printf("canMapHostMemory:              %d\n", d.canMapHostMemory);
    printf("concurrentKernels:             %d\n", d.concurrentKernels);
    printf("streamPrioritiesSupported:     %d\n", d.streamPrioritiesSupported);
    printf("globalL1CacheSupported:        %d\n", d.globalL1CacheSupported);
    printf("localL1CacheSupported:         %d\n", d.localL1CacheSupported);
    printf("managedMemory:                 %d\n", d.managedMemory);
    printf("hostNativeAtomicSupported:     %d\n", d.hostNativeAtomicSupported);
    printf("pageableMemoryAccess:          %d\n", d.pageableMemoryAccess);
    printf("pageableMemoryAccessUsesHostPageTables: %d\n", d.pageableMemoryAccessUsesHostPageTables);
    printf("concurrentManagedAccess:       %d\n", d.concurrentManagedAccess);
    printf("computePreemptionSupported:    %d\n", d.computePreemptionSupported);
    printf("canUseHostPointerForRegisteredMem: %d\n", d.canUseHostPointerForRegisteredMem);
    printf("cooperativeLaunch:             %d\n", d.cooperativeLaunch);
    printf("directManagedMemAccessFromHost: %d\n", d.directManagedMemAccessFromHost);
    printf("hostRegisterSupported:         %d\n", d.hostRegisterSupported);
    printf("hostRegisterReadOnlySupported: %d\n", d.hostRegisterReadOnlySupported);
    printf("sparseCudaArraySupported:      %d\n", d.sparseCudaArraySupported);
    printf("deferredMappingCudaArraySupported: %d\n", d.deferredMappingCudaArraySupported);
    printf("timelineSemaphoreInteropSupported: %d\n", d.timelineSemaphoreInteropSupported);
    printf("memoryPoolsSupported:          %d\n", d.memoryPoolsSupported);
    printf("gpuDirectRDMASupported:        %d\n", d.gpuDirectRDMASupported);
    printf("gpuDirectRDMAWritesOrdering:   %d\n", d.gpuDirectRDMAWritesOrdering);
    printf("gpuDirectRDMAFlushWritesOptions: 0x%08x\n", d.gpuDirectRDMAFlushWritesOptions);
    printf("memoryPoolSupportedHandleTypes: 0x%08x\n", d.memoryPoolSupportedHandleTypes);
    printf("ipcEventSupported:             %d\n", d.ipcEventSupported);
    printf("clusterLaunch:                 %d\n", d.clusterLaunch);
    printf("unifiedFunctionPointers:       %d\n", d.unifiedFunctionPointers);
    printf("mpsEnabled:                    %d\n", d.mpsEnabled);
    printf("accessPolicyMaxWindowSize:     %d bytes\n", d.accessPolicyMaxWindowSize);
}

/***********************************************************
 * @brief Checks for CUDA capable Device
 *
 * @param majorNumber
 * @param minorNumber
 * @return 0/1 status if the device is capable of CUDA or not
 ***********************************************************/
int checkForCUDACapability(int major, int minor)
{
    // Code
    if (major >= 1)
        return (1);
    else if (1 == major && 3 < minor)
        return (1);
    else
        return (0);
}

// Entry Point Function
int main(void)
{
    // Code
    cudaError_t cuError;
    int devCount = 0;
    cudaDeviceProp cuDevProp;
    int devID;

    cuError = cudaGetDeviceCount(&devCount);
    if (cudaSuccess != cuError)
    {
        fprintf(stderr, "main() -> CUDA Error %s", cudaGetErrorString(cuError));
        exit(1);
    }

    fprintf(stdout, "Devices Found = %d\n", devCount);

    for (int le = 0; le < devCount; ++le)
    {
        cuError = cudaGetDeviceProperties(&cuDevProp, le);
        if (cudaSuccess != cuError)
        {
            fprintf(stderr, "main() -> CUDA Error %s", cudaGetErrorString(cuError));
            exit(1);
        }

        printf("    === General Information For Device Found at %d ===  \n", le);

        printAllDevProp(cuDevProp);

        printf("    === === === ===\n\n");

        if (1 == checkForCUDACapability(cuDevProp.major, cuDevProp.minor))
        {

            CHECKCUDAERROR("main()", cudaGetDevice(&devID));

            printf("ID of CUDA Capable Device = %d\n", devID);

            CHECKCUDAERROR("main()", cudaChooseDevice(&devID, &cuDevProp));
        }
    }

    return (0);
}