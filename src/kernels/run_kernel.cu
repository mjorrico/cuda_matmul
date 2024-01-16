#include <iostream>
#include <assert.h>

#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#include "matrixfunctions.hpp"
#include "run_kernel.cuh"

#include "kernels/001_matmul_naive.cuh"
#include "kernels/002_matmul_coalesced.cuh"
#include "kernels/003_matmul_coalesced_v2.cuh"
#include "kernels/004_matmul_cacheblock.cuh"
#include "kernels/005_matmul_cacheblock_v2.cuh"
#include "kernels/006_matmul_blocktiling1d.cuh"
#include "kernels/007_matmul_blocktiling2d.cuh"


void mmul_benchmark(mmulFunc mmul, float* dev_a, float* dev_b, float* dev_c, float* c, int N, double gflop, double memoryio, const int is_cublas) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    float elapsed_time, best_time;

    if (is_cublas) {
        std::cout << "With cuBLAS" << std::endl;
    }

    for (int i = 0; i < 10; i++) {
        best_time = 1e9;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        if (is_cublas) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, dev_b, N, dev_a, N, &beta, dev_c, N);
        }
        else {
            mmul(dev_a, dev_b, dev_c, N);
        }

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed_time, start, stop);
        best_time = imin(best_time, elapsed_time);

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    std::cout << "Throughput: " << 1e3 * gflop / best_time << " GFLOPs." << std::endl; // GLOP / ms
    std::cout << "Bandwidth: " << memoryio / best_time << " GB/s." << std::endl << std::endl; // MB / ms

    cudaMemcpy(c, dev_c, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
}

void run_mmul_cublas(float* a, float* b, float* c, int N) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0;
    const float beta = 0.0;

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, b, N, a, N, &beta, c, N);

    cublasDestroy(handle);
}

void run_mmul_naive(float* a, float* b, float* c, int N) {
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_naive<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_coalesced(float* a, float* b, float* c, int N) {
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32 * 32);

    mmul_coalesced<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_coalesced_v2(float* a, float* b, float* c, int N) {
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_coalesced_v2<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_cacheblock(float* a, float* b, float* c, int N) {
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32 * 32);

    mmul_cacheblock<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_cacheblock_v2(float* a, float* b, float* c, int N) {
    dim3 gridsize((N + 31) / 32, (N + 31) / 32);
    dim3 blocksize(32, 32);

    mmul_cacheblock_v2<<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_blocktiling1d(float* a, float* b, float* c, int N) {
    const uint BM = 64;
    const uint BK = 8;
    const uint BN = 64;
    const uint TM = 8;

    dim3 gridsize((N + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blocksize(BM * BN / TM);

    assert(BM == BN);
    assert(BK == BM / TM);

    mmul_blocktiling1d<BM, BN, BK, TM><<<gridsize, blocksize>>>(a, b, c, N);
}

void run_mmul_blocktiling2d(float* a, float* b, float* c, int N) {
    const uint BM = 128;
    const uint BK = 8;
    const uint BN = 128;
    const uint TM = 4;
    const uint TN = 4;

    dim3 gridsize((N + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 blocksize((BM / TM) * (BN / TN));

    assert(BM == BN);
    assert(BM == BK * TM * TN);

    mmul_blocktiling2d<BM, BN, BK, TM, TN><<<gridsize, blocksize>>>(a, b, c, N);
}