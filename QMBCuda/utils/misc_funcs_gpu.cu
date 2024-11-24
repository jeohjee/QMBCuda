#include "misc_funcs_gpu.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/complex.h"
#include "curand_kernel.h"

using complex_th = thrust::complex<float>;

__device__ double NChoosek2_dev(double num, double den) {

    if (den <= 0) return 1;
    if (num <= 0) return 1;

    if (den <= 1) {
        return num;
    }
    return (num / den) * NChoosek2_dev(num - 1, den - 1);
}


template struct mul_thrust_vec<float>;
template struct mul_thrust_vec<complex_th>;

template struct saxpy_functor<float>;
template struct saxpy_functor<complex_th>;

template struct saxpy_functor2<float>;
template struct saxpy_functor2<complex_th>;


template <typename S, typename T> __global__ void init_random_vec(long long seed, S* state, int nobv, T* __restrict target_vec)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id > nobv) {
        return;
    }
    curand_init(seed + id, 0, 0, &state[id]);
    curandState newState = state[id];
    float tmpVal = curand_uniform(&newState);
    state[id] = newState;

    if constexpr (::cuda::std::is_same_v<T, float>) {
        target_vec[id] = tmpVal;
    }
    else {
        curandState newState = state[id];
        float tmpVal2 = curand_uniform(&newState);
        state[id] = newState;

        target_vec[id] = complex_th(tmpVal, tmpVal2);
    }
}

template __global__ void init_random_vec<curandState,float>(long long seed, curandState* state, int nobv, float* __restrict target_vec);
template __global__ void init_random_vec<curandState,complex_th>(long long seed, curandState* state, int nobv, complex_th* __restrict target_vec);


template <typename T>
__device__ void GetComplexConjugate(T* orig_num) {

    if constexpr (::cuda::std::is_same_v<T, complex_th>) {
        *orig_num = thrust::conj(*orig_num);
        return;
    }
    else {
        return;
    }
};
template __device__ void GetComplexConjugate<float>(float*);
template __device__ void GetComplexConjugate<complex_th>(complex_th*);



template <typename T>
T GetComplexConjugateHost(T orig_num) {
    if constexpr (::cuda::std::is_same_v<T, complex_th>) return thrust::conj(orig_num);
    else return orig_num;
}
template float GetComplexConjugateHost<float>(float orig_num);
template complex_th GetComplexConjugateHost<complex_th>(complex_th orig_num);

template <typename T>
float ExtReal(T x) {
    if constexpr (::cuda::std::is_same_v<T, float>) return x;
    else return x.real();
}
template float ExtReal<float>(float x);
template float ExtReal<complex_th>(complex_th x);