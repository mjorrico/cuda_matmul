#ifndef MATMUL_CACHEBLOCK_H
#define MATMUL_CACHEBLOCK_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void mmul_cacheblock(float* a, float* b, float* c, int N) {
    __shared__ float shared_a[32 * 32];
    __shared__ float shared_b[32 * 32];

    int cRow = blockIdx.x;
    int cCol = blockIdx.y;

    int threadRow = threadIdx.x / 32;
    int threadCol = threadIdx.x % 32;

    a += cRow * 32 * N;
    b += cCol * 32;
    c += cRow * 32 * N + cCol * 32;

    float temp = 0;
    for (int blkIdx = 0; blkIdx < N; blkIdx += 32) {
        shared_a[threadRow * 32 + threadCol] = a[threadRow * N + threadCol];
        shared_b[threadRow * 32 + threadCol] = b[threadRow * N + threadCol];
        __syncthreads();

        a += 32;
        b += 32 * N;

        for (int dotIdx = 0; dotIdx < 32; dotIdx++) {
            temp += shared_a[threadRow * 32 + dotIdx] * shared_b[dotIdx * 32 + threadCol];
        }
        __syncthreads();
    }

    c[threadRow * N + threadCol] = temp;
}

#endif