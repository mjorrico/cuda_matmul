#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void mmul_cacheblock_v2(float* a, float* b, float* c, int N) {
    __shared__ float shared_a[32 * 32];
    __shared__ float shared_b[32 * 32];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp = 0;
    for (int blkIdx = 0; blkIdx < N; blkIdx += 32) {
        shared_a[threadIdx.y * 32 + threadIdx.x] = a[row * N + blkIdx + threadIdx.x];
        shared_b[threadIdx.y * 32 + threadIdx.x] = b[col + threadIdx.y * N + blkIdx * N];
        __syncthreads();

        for (int dotIdx = 0; dotIdx < 32; dotIdx++) {
            temp += shared_a[threadIdx.y * 32 + dotIdx] * shared_b[dotIdx * 32 + threadIdx.x];
        }
        __syncthreads();
    }

    c[row * N + col] = temp;
}