#ifndef RUN_KERNEL_H
#define RUN_KERNEL_H

typedef void (*mmulFunc)(float*, float*, float*, int);

void mmul_benchmark(mmulFunc mmul, float* dev_a, float* dev_b, float* dev_c, float* c, int N, double gflop, double memoryio);

void run_mmul_naive(float* a, float* b, float* c, int N);
void run_mmul_coalesced(float* a, float* b, float* c, int N);
void run_mmul_coalesced_v2(float* a, float* b, float* c, int N);

__global__ void mmul_naive(float* a, float* b, float* c, int N);
__global__ void mmul_coalesced(float* a, float* b, float* c, int N);
__global__ void mmul_coalesced_v2(float* a, float* b, float* c, int N);

#endif