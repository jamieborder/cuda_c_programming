#include <stdio.h>

int main(void)
{
    int dev = 0;
    cudaSetDevice(dev);

    int driverVersion = 0, runtimeVersion = 0;

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties( &deviceProp, dev );

    printf("Device %d; \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion( &driverVersion );
    cudaRuntimeGetVersion( &runtimeVersion );

    printf("  CUDA Driver Version / Runtime Version:            %d.%d / %d.%d\n",
            driverVersion/1000, (driverVersion%100)/10, 
            runtimeVersion/1000, (runtimeVersion%100)/10);
    printf("  CUDA Capability Major/Minor version number:       %d.%d\n",
            deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                    %.2f GBytes (%llu bytes)\n",
            (float)deviceProp.totalGlobalMem/(pow(1024.0,3)),
            (unsigned long long)deviceProp.totalGlobalMem);
    printf("  GPU Clock rate:                                   %.0f MHz (%0.2f GHz)\n",
            deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
    printf("  Memory Clock rate:                                %.0f Mhz\n",
            deviceProp.memoryClockRate * 1e-3f);
    printf("  Total amount of shared memory per block:          %lu bytes\n",
            deviceProp.sharedMemPerBlock);
    printf("  Total numer of registers available per block:     %d\n",
            deviceProp.regsPerBlock);
    printf("  Warp size:                                        %d\n",
            deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:     %d\n",
            deviceProp.maxThreadsPerMultiProcessor);
    printf("  maximum number of threads per block:              %d\n",
            deviceProp.maxThreadsPerBlock);
    printf("  Maximum sizes of each dimension of a block:       %d x %d x %d\n",
            deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
            deviceProp.maxThreadsDim[2]);
    printf("  Maximum sizes of each dimension of a grid:        %d x %d x %d\n",
            deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
            deviceProp.maxGridSize[2]);

    return 0;
}
