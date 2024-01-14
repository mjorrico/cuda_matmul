#ifndef MYFUNCTIONS_H
#define MYFUNCTIONS_H

void print_matrix(float* matrix, int N, int precision);
void generate_matrix(float* arr, int N);
float imax(float x, float y);
float imin(float x, float y);
float iabs(float x);
void transpose(float* source, float* dest, int N);
void validate(float* a, float* b, float* c, int N);
void mmul_cpu(float* a, float* b, float* c, int N);

#endif