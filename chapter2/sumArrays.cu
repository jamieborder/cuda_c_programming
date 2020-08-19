#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

void initialData( float *ip, int size )
{
    // generate different seed for random number
    time_t t;
    srand( (unsigned int) time (&t) );

    for (int i=0; i<size; i++) {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }
}

void sumArraysOnHost( float *A, float *B, float *C, const int N )
{
    for (int idx=0; idx<N; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void sumArraysOnDevice( float *A, float *B, float *C, const int N)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= N) return;
    C[tid] = A[tid] + B[tid];
}

double cpuSecond()
{
    struct timeval tp;
    gettimeofday( &tp, NULL );
    return ( (double)tp.tv_sec + (double)tp.tv_usec*1e-6 );
}

int main( int argc, char **argv )
{
    // timing...
    double startTime, cpuElapsed, gpuElapsed;

    // data
    int nElem = 1<<24;
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData( h_A, nElem );
    initialData( h_B, nElem );

    // set up device
    int dev = 0;
    cudaSetDevice(dev);

    float *d_A, *d_B, *d_C, *gpuRes;
    cudaMalloc( (float **)&d_A, nBytes );
    cudaMalloc( (float **)&d_B, nBytes );
    cudaMalloc( (float **)&d_C, nBytes );
    gpuRes = (float *)malloc(nBytes);

    cudaMemcpy( d_A, h_A, nBytes, cudaMemcpyHostToDevice );
    cudaMemcpy( d_B, h_B, nBytes, cudaMemcpyHostToDevice );

    startTime = cpuSecond();
    sumArraysOnHost( h_A, h_B, h_C, nElem );
    cpuElapsed = cpuSecond() - startTime;

    dim3 block = 256;    // number of warps per block
    dim3 grid = (nElem + block.x - 1) / block.x;    // number of blocks in grid
    printf("block = %i\n", block.x);
    printf("grid  = %i\n", grid.x);
    printf("total launched = %i\n", block.x * grid.x);
    printf("total needed   = %i\n", nElem);

    startTime = cpuSecond();
    sumArraysOnDevice<<< grid, block >>>( d_A, d_B, d_C, nElem );
    cudaDeviceSynchronize();
    gpuElapsed = cpuSecond() - startTime;

    cudaError_t error = cudaMemcpy( gpuRes, d_C, nBytes,
            cudaMemcpyDeviceToHost );

    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }

    double err = 0.0;
    for (int i=0; i<nElem; i++) {
        err += abs(h_C[i] - gpuRes[i]);
    }

    printf("Total error is %f\n", err);
    printf("Time on CPU is %f\n", cpuElapsed);
    printf("Time on GPU is %f\n", gpuElapsed);
    printf("GPU speed-up over CPU is %.2f x\n", cpuElapsed / gpuElapsed);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(gpuRes);

    return(0);
}
