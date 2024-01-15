#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "matrixfunctions.hpp"
#include "run_kernel.cuh"

void mmul_benchmark(mmulFunc mmul, float* dev_a, float* dev_b, float* dev_c, float* c, int N, double gflop, double memoryio) {
    float elapsed_time, best_time = 1e9;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < 10; i++) {
        cudaEventRecord(start);

        mmul(dev_a, dev_b, dev_c, N);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        best_time = imin(best_time, elapsed_time);
    }

    std::cout << "Throughput: " << 1e3 * gflop / best_time << " GFLOPs." << std::endl; // GLOP / ms
    std::cout << "Bandwidth: " << memoryio / best_time << " GB/s." << std::endl << std::endl; // MB / ms

    cudaMemcpy(c, dev_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

void run_mmul_naive(float* a, float* b, float* c, int N){
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_naive<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_coalesced(float* a, float* b, float* c, int N){
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32 * 32);

    mmul_coalesced<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_coalesced_v2(float* a, float* b, float* c, int N){
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_coalesced_v2<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_cacheblock(float* a, float* b, float* c, int N){
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32 * 32);

    mmul_cacheblock<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_cacheblock_v2(float* a, float* b, float* c, int N){
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_cacheblock_v2<<<gridsize, blocksize>>>(a, b, c, N);
}