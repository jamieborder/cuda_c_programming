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

void sumMatrixOnHost_1( float *A, float *B, float *C, const int nx,
        const int ny )
{
    float *ia = A;
    float *ib = B;
    float *ic = C;
    for (int iy=0; iy<ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

void sumMatrixOnHost_2( float *A, float *B, float *C, const int nx,
        const int ny )
{
    for (int iy=0; iy<ny; iy++) {
        for (int ix=0; ix<nx; ix++) {
            C[ix + iy * nx] = A[ix + iy * nx] + B[ix + iy * nx];
        }
    }
}

void sumMatrixOnHost_3( float *A, float *B, float *C, const int nx,
        const int ny )
{
    int nxy = nx * ny;
    for (int i=0; i<nxy; i++) {
            C[i] = A[i] + B[i];
    }
}

__global__ void sumMatrixOnDevice_1( float *A, float *B, float *C,
        const int nx, const int ny )
{
    size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny) {
        C[ix + iy * nx] = A[ix + iy * nx] + B[ix + iy * nx];
    }
    return;
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
    double startTime, cpuElapsed_1, cpuElapsed_2, cpuElapsed_3, gpuElapsed;

    // data
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    size_t nBytes = nxy * sizeof(float);
    printf("matrix dimensions  = (%i, %i)\n", nx, ny);
    printf("number of elements = %i\n", nxy);
    printf("number of bytes in each matrix = %lu\n", nBytes);

    float *h_A, *h_B, *h_C;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);

    initialData( h_A, nxy );
    initialData( h_B, nxy );

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

    memset( h_C, 0, nBytes );
    memset( gpuRes, 0, nBytes );

    startTime = cpuSecond();
    sumMatrixOnHost_1( h_A, h_B, h_C, nx, ny );
    cpuElapsed_1 = cpuSecond() - startTime;

    startTime = cpuSecond();
    sumMatrixOnHost_2( h_A, h_B, h_C, nx, ny );
    cpuElapsed_2 = cpuSecond() - startTime;

    startTime = cpuSecond();
    sumMatrixOnHost_3( h_A, h_B, h_C, nx, ny );
    cpuElapsed_3 = cpuSecond() - startTime;

    dim3 block(32, 32);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printf("block = (%i, %i, %i)\n", block.x, block.y, block.z);
    printf("grid  = (%i, %i, %i)\n", grid.x, grid.y, grid.z);
    printf("number of threads / block = %i\n", block.x * block.y * block.z);
    if (block.x * block.y * block.z > 1024) {
        printf("Can only launch 1024 threads per block, not %i\n",
                block.x * block.y * block.z);
        exit(1);
    }
    printf("total launched = %i\n", block.x * grid.x * block.y * grid.y);
    printf("total needed   = %i\n", nxy);

    startTime = cpuSecond();
    sumMatrixOnDevice_1<<< grid, block >>>( d_A, d_B, d_C, nx, ny );
    cudaDeviceSynchronize();
    printf("Launched with blockDim = (%i, %i, %i)\n", block.x, block.y, block.z);
    printf("Launched with  gridDim = (%i, %i, %i)\n", grid.x, grid.y, grid.z);
    gpuElapsed = cpuSecond() - startTime;

    cudaError_t error = cudaMemcpy( gpuRes, d_C, nBytes,
            cudaMemcpyDeviceToHost );

    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }

    double err = 0.0;
    for (int i=0; i<nxy; i++) {
        err += abs(h_C[i] - gpuRes[i]);
        // printf("%3i:: %7.3f %7.3f %7.3f %7.3f\n", i, h_C[i], gpuRes[i], h_A[i], h_B[i]);
    }

    printf("Total error is %f\n", err);
    printf("Time on CPU_v1 is %f\n", cpuElapsed_1);
    printf("Time on CPU_v2 is %f\n", cpuElapsed_2);
    printf("Time on CPU_v3 is %f\n", cpuElapsed_3);
    printf("Time on GPU    is %f\n", gpuElapsed);
    printf("GPU speed-up over CPU is %.2f x\n",
            (cpuElapsed_1 < cpuElapsed_2 ? 
               (cpuElapsed_1 < cpuElapsed_3 ? cpuElapsed_1 : cpuElapsed_3) 
             : (cpuElapsed_2 < cpuElapsed_3 ? cpuElapsed_2 : cpuElapsed_3)) / gpuElapsed);

    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(gpuRes);

    return 0;
}

int comp(const void * elem1, const void * elem2) 
{
    int f = *((int*)elem1);
    int s = *((int*)elem2);
    if (f > s) return  1;
    if (f < s) return -1;
    return 0;
}

void checkSol()
{
    int indices[] = { 96, 97 , 98 , 99 , 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115,
                     116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 32 , 33 , 34 , 35 , 36 , 37 , 38 , 39 ,
                     40 , 41 , 42 , 43 , 44 , 45 , 46 , 47 , 48 , 49 , 50 , 51 , 52 , 53 , 54 , 55 , 56 , 57 , 58 , 59 ,
                     60 , 61 , 62 , 63 , 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
                     144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163,
                     164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183,
                     184, 185, 186, 187, 188, 189, 190, 191, 64 , 65 , 66 , 67 , 68 , 69 , 70 , 71 , 72 , 73 , 74 , 75 ,
                     76 , 77 , 78 , 79 , 80 , 81 , 82 , 83 , 84 , 85 , 86 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 ,
                     0  , 1  , 2  , 3  , 4  , 5  , 6  , 7  , 8  , 9  , 10 , 11 , 12 , 13 , 14 , 15 , 16 , 17 , 18 , 19 ,
                     20 , 21 , 22 , 23 , 24 , 25 , 26 , 27 , 28 , 29 , 30 , 31 , 192, 193, 194, 195, 196, 197, 198, 199,
                     200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219,
                     220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
                     240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255 };

    qsort(indices, sizeof(indices)/sizeof(*indices), sizeof(*indices), comp);

    for (int i = 0 ; i < sizeof(indices)/sizeof(*indices) ; i++) {
        // printf ("%d ", indices[i]);
        if (i != indices[i]) {
            printf("failed:: %d:%d\n", i, indices[i]);
            break;
        }
    }
    printf("success:: all matched\n");
    return;
}
