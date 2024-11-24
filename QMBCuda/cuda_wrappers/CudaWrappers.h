#pragma once
#include <cstddef>
#include <cuComplex.h>
#include "device_launch_parameters.h"



// This header contains some wrapper functions for some of the  most useful cuBLAS ad cuSOLVER functions

// Vector norm for the float and double:
float VecNorm_cuBLAS_float_wrapper(float* __restrict d_A, int N);
double VecNorm_cuBLAS_double_wrapper(double* __restrict d_A, int N);
float VecNorm_cuBLAS_complex_float_wrapper(cuFloatComplex* __restrict d_A, int N);

template <typename T>
float VecNorm_cuBLAS_generic_float_wrapper(T* __restrict d_A, int N);

// Dot products
float DotProd_cuBLAS_float_wrapper(float* __restrict d_A, float* __restrict d_B, int N);
double DotProd_cuBLAS_double_wrapper(double* __restrict d_A, double* __restrict d_B, int N);
cuFloatComplex DotProd_cuBLAS_complex_wrapper(cuFloatComplex* __restrict d_A, cuFloatComplex* __restrict d_B, int N);

template <typename T>
T DotProd_cuBLAS_generic_float_wrapper(T* __restrict d_A, T* __restrict d_B, int N);


// Dense eigensolver for symmetric matrix:
template <typename T> void SymmDiag_cuSOLVER_wrapper(T* __restrict H_mat, T* __restrict W, T* __restrict V, const int N);

