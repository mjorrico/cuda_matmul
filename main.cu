#include <iostream>
#include <cstdlib>

#include "matrixfunctions.hpp"

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

    generate_matrix(a, N);
    generate_matrix(b, N);

    mmul_cpu(a, b, c, N);
    validate(a, b, c, N);

    free(a);
    free(b);
    free(c);
}