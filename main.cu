#include <iostream>
#include <cstdlib>

#include "matrixfunctions.hpp"
#include "run_kernel.cuh"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Matrix size is required." << std::endl;
        exit(1);
    }

    int N = 1 << atoi(argv[1]);
    int bytes = N * N * sizeof(float);
    double gflop = (2 * (double)N - 1) * N * N * 1e-9;
    double memoryio = 3 * N * N * sizeof(float) * 1e-6;
    std::cout << "Matrix size: " << N << " x " << N << std::endl;
    std::cout << "Total FLOP: " << gflop << " GFLOP" << std::endl;
    std::cout << "Memory I/O: " << memoryio << " MB" << std::endl << std::endl;

    float* a = (float*)malloc(bytes);
    float* b = (float*)malloc(bytes);
    float* c = (float*)malloc(bytes);

    float* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, bytes);
    cudaMalloc((void**)&dev_b, bytes);
    cudaMalloc((void**)&dev_c, bytes);

    generate_matrix(a, N);
    generate_matrix(b, N);
    cudaMemcpy(dev_a, a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, bytes, cudaMemcpyHostToDevice);

    // mmul_cpu(a, b, c, N);
    // mmul_benchmark(run_mmul_naive, dev_a, dev_b, dev_c, c, N, gflop, memoryio);
    mmul_benchmark(run_mmul_coalesced, dev_a, dev_b, dev_c, c, N, gflop, memoryio);
    validate(a, b, c, N);
    mmul_benchmark(run_mmul_coalesced_v2, dev_a, dev_b, dev_c, c, N, gflop, memoryio);
    validate(a, b, c, N);
    mmul_benchmark(run_mmul_cacheblock, dev_a, dev_b, dev_c, c, N, gflop, memoryio);
    validate(a, b, c, N);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);
}