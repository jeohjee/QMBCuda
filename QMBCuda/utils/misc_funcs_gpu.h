#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/complex.h"

__device__ double NChoosek2_dev(double num, double den);

template <typename T> 
struct mul_thrust_vec
{
    const float a;
    mul_thrust_vec(float _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x) const {
        return a * x;
    }
};

template <typename T>
struct saxpy_functor
{
    const float a;
    saxpy_functor(float _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x, const T& y) const {
        return a * x + y;
    }
};


template <typename T>
struct saxpy_functor2
{
    const float a;

    saxpy_functor2(float _a) : a(_a) {}

    __host__ __device__
        T operator()(const T& x, const T& y) const {
        return x + a * y;
    }
};

template <typename S, typename T> __global__ void init_random_vec(long long seed, S* state, int nobv, T* __restrict target_vec);

template <typename T> __device__ void GetComplexConjugate(T* orig_num);

//__device__ void GetComplexConjugate(thrust::complex<float>* orig_num);

template <typename T> T GetComplexConjugateHost(T orig_num);

template <typename T> float ExtReal(T x);
