#ifndef MATMUL_BLOCKTILING2D_H
#define MATMUL_BLOCKTILING2D_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

template <const uint BM, const uint BN, const uint BK, const uint TM, const uint TN> __global__ void mmul_blocktiling2d(float* a, float* b, float* c, int N) {
    __shared__ float shared_a[BM * BK];
    __shared__ float shared_b[BK * BN];

    int row = blockIdx.y * BM + (threadIdx.x / (BN / TN)) * TM;
    int col = blockIdx.x * BN + (threadIdx.x % (BN / TN)) * TN;

    int blkRowA = threadIdx.x / BK;
    int blkColA = threadIdx.x % BK;
    int blkRowB = threadIdx.x / BN;
    int blkColB = threadIdx.x % BN;

    float temp[TM * TN] = { 0 };
    float bThread;
    for (int blkIdx = 0; blkIdx < N; blkIdx += BK) {
        shared_a[blkRowA * BK + blkColA] = a[blockIdx.y * BM * N + blkRowA * N + blkColA + blkIdx];
        shared_b[blkRowB * BN + blkColB] = b[blockIdx.x * BN + blkColB + blkRowB * N + blkIdx * N];
        __syncthreads();

        for (int bk = 0; bk < BK; bk++) {
            for (int tn = 0; tn < TN; tn++) {
                bThread = shared_b[bk * BN + tn + (threadIdx.x * TN) % BN];
                for (int tileY = 0; tileY < TM; tileY++) {
                    temp[tileY * TN + tn] += shared_a[tileY * BK + bk + (threadIdx.x * TN) / BN * BK * TM] * bThread;
                }
            }
        }
        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++) {
        for (int tn = 0; tn < TN; tn++) {
            c[(row + tm) * N + col + tn] = temp[tm * TN + tn];
        }
    }
}

#endif