#include <iostream>
#include <iomanip>
#include <time.h>
#include <math.h>

void print_matrix(float* matrix, int N, int precision) {
    if (N > 16) {
        std::cout << "Matrix is too big. Can't be printed." << std::endl;
        return;
    }

    std::cout << std::fixed << std::setprecision(precision);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << std::setw(precision + 3) << matrix[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    std::cout << std::fixed << std::setprecision(7);
}

void generate_matrix(float* arr, int N) {
    srand((clock_t)std::clock());
    float c = sqrt((float)4 / N);
    for (int i = 0; i < N * N; i++) {
        arr[i] = (float)rand() * c / RAND_MAX;
    }
}

float imax(float x, float y) {
    return (x > y) ? x : y;
}

float imin(float x, float y) {
    return (x < y) ? x : y;
}

float iabs(float x) {
    return (x > 0) ? x : -x;
}

void transpose(float* source, float* dest, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dest[j * N + i] = source[i * N + j];
        }
    }
}

void validate(float* a, float* b, float* c, int N) {
    float res, total = 0, max_res = 0;
    int n = imin(256, N);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            res = 0;
            for (int k = 0; k < N; k++) {
                res += a[i * N + k] * b[k * N + j];
            }
            max_res = imax(max_res, iabs(res - c[i * N + j]));
            total += c[i * N + j];
        }
    }
    std::cout << "Max diff: " << max_res << std::endl;
    std::cout << "Avg element: " << total / (n * n) << std::endl << std::endl;
}

void mmul_cpu(float* a, float* b, float* c, int N) {
    float temp;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            temp = 0;
            for (int k = 0; k < N; k++) {
                temp += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] = temp;
        }
    }
}