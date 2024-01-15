#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void mmul_coalesced(float* a, float* b, float* c, int N) {
    const uint row = blockIdx.x * 32 + (threadIdx.x / 32);
    const uint col = blockIdx.y * 32 + (threadIdx.x % 32);

    if (row < N && col < N) {
        float temp = 0;
        for (int i = 0; i < N; i++) {
            temp += a[row * N + i] * b[i * N + col];
        }

        c[row * N + col] = temp;
    }
}