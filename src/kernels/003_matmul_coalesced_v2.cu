#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

__global__ void mmul_coalesced_v2(float* a, float* b, float* c, int N) {
    // swapped so that memory access to A is coalesced
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        float temp = 0;
        for (int i = 0; i < N; i++) {
            temp += a[row * N + i] * b[i * N + col];
        }

        c[row * N + col] = temp;
    }
}