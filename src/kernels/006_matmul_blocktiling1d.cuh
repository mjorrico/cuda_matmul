#ifndef MATMUL_BLOCKTILING1D_H
#define MATMUL_BLOCKTILING1D_H

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

template <const uint BM, const uint BN, const uint BK, const uint TM> __global__ void mmul_blocktiling1d(float* a, float* b, float* c, int N) {
    __shared__ float shared_a[BM * BK];
    __shared__ float shared_b[BK * BN];

    int row = blockIdx.y * BM + (threadIdx.x / BN) * TM;
    int col = blockIdx.x * BN + (threadIdx.x % BN);

    int blkRowA = threadIdx.x / BK;
    int blkColA = threadIdx.x % BK;
    int blkRowB = threadIdx.x / BN;
    int blkColB = threadIdx.x % BN;

    float temp[TM] = { 0 };
    float bThread;
    for (int blkIdx = 0; blkIdx < N; blkIdx += BK) {
        shared_a[blkRowA * BK + blkColA] = a[blockIdx.y * BM * N + blkRowA * N + blkColA + blkIdx];
        shared_b[blkRowB * BN + blkColB] = b[blockIdx.x * BN + blkColB + blkRowB * N + blkIdx * N];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            bThread = shared_b[dotIdx * BN + blkColB];
            for (int thrMulIdx = 0; thrMulIdx < TM; thrMulIdx++) {
                temp[thrMulIdx] += shared_a[thrMulIdx * BK + dotIdx + blkRowB * BK * TM] * bThread;
            }
        }
        __syncthreads();
    }

    for (int tm = 0; tm < TM; tm++) {
        c[(row + tm) * N + col] = temp[tm];
    }
}

#endif
