#include "CudaWrappers.h"
#include "thrust/complex.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "../cublas_utils.h"
#include "../cusolver_utils.h"
#include <cusolverDn.h>

using complex_th = thrust::complex<float>;

float VecNorm_cuBLAS_float_wrapper(float* __restrict d_A, int N) {

    cublasHandle_t cublasH = NULL;
    //cudaStream_t stream = NULL;

    const int incx = 1;
    float result = 0.0;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: compute */
    CUBLAS_CHECK(cublasSnrm2(cublasH, N, d_A, incx, &result));

    //CUDA_CHECK(cudaStreamSynchronize(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    //CUDA_CHECK(cudaStreamDestroy(stream));

    return result;
}

double VecNorm_cuBLAS_double_wrapper(double* __restrict d_A, int N) {

    cublasHandle_t cublasH = NULL;
    //cudaStream_t stream = NULL;

    const int incx = 1;
    double result = 0.0;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: compute */
    CUBLAS_CHECK(cublasDnrm2(cublasH, N, d_A, incx, &result));

    //CUDA_CHECK(cudaStreamSynchronize(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    //CUDA_CHECK(cudaStreamDestroy(stream));

    return result;
}

float VecNorm_cuBLAS_complex_float_wrapper(cuFloatComplex* __restrict d_A, int N) {

    cublasHandle_t cublasH = NULL;

    const int incx = 1;
    float result = 0.0;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));
    //CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    //CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: compute */
    CUBLAS_CHECK(cublasScnrm2(cublasH, N, d_A, incx, &result));

    //CUDA_CHECK(cudaStreamSynchronize(stream));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    //CUDA_CHECK(cudaStreamDestroy(stream));

    return result;
}


template <typename T>
float VecNorm_cuBLAS_generic_float_wrapper(T* __restrict d_A, int N) {

    float result = 0.0;
    if constexpr (::cuda::std::is_same_v<T, float>) {
        result = VecNorm_cuBLAS_float_wrapper(d_A, N);
        return result;
    }
    else {
        result = VecNorm_cuBLAS_complex_float_wrapper(reinterpret_cast<cuFloatComplex*>(d_A), N);
        return result;
    }

}
template float VecNorm_cuBLAS_generic_float_wrapper<float>(float* __restrict d_A, int N);
template float VecNorm_cuBLAS_generic_float_wrapper<complex_th>(complex_th* __restrict d_A, int N);

// Dot products
float DotProd_cuBLAS_float_wrapper(float* __restrict d_A, float* __restrict d_B, int N) {

    cublasHandle_t cublasH = NULL;

    const int incx = 1;
    const int incy = 1;

    float result = 0.0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasSdot(cublasH, N, d_A, incx, d_B, incy, &result));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    return result;
}

double DotProd_cuBLAS_double_wrapper(double* __restrict d_A, double* __restrict d_B, int N) {

    cublasHandle_t cublasH = NULL;

    const int incx = 1;
    const int incy = 1;

    double result = 0.0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasDdot(cublasH, N, d_A, incx, d_B, incy, &result));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    return result;
}

cuFloatComplex DotProd_cuBLAS_complex_wrapper(cuFloatComplex* __restrict d_A, cuFloatComplex* __restrict d_B, int N) {

    cublasHandle_t cublasH = NULL;

    const int incx = 1;
    const int incy = 1;

    cuFloatComplex result;
    result.x = 0.0;
    result.y = 0.0;

    CUBLAS_CHECK(cublasCreate(&cublasH));
    CUBLAS_CHECK(cublasCdotc(cublasH, N, d_A, incx, d_B, incy, &result));
    CUBLAS_CHECK(cublasDestroy(cublasH));

    return result;
}

template <typename T>
T DotProd_cuBLAS_generic_float_wrapper(T* __restrict d_A, T* __restrict d_B, int N) {

    if constexpr (::cuda::std::is_same_v<T, float>) {
        float result;
        result = DotProd_cuBLAS_float_wrapper(d_A, d_B, N);
        return result;
    }
    else {
        cuFloatComplex result;
        result = DotProd_cuBLAS_complex_wrapper(
            reinterpret_cast<cuFloatComplex*>(d_A),
            reinterpret_cast<cuFloatComplex*>(d_B),
            N);
        return complex_th(result.x, result.y);
    }

}
template float DotProd_cuBLAS_generic_float_wrapper(float* __restrict d_A, float* __restrict d_B, int N);
template complex_th DotProd_cuBLAS_generic_float_wrapper(complex_th* __restrict d_A, complex_th* __restrict d_B, int N);

// Dense eigensolver for symmetric matrix:

template <typename T> 
void SymmDiag_cuSOLVER_wrapper(T* __restrict H_mat, T* __restrict W, T* __restrict V, const int N) {

    /* Parameters:
    * H_mat: a NxN matrix to be diagonalized
    * W: a vector of size N that will store the eigenvalues
    * V: NxN matrix for storing the eigenvectors
    */

    const int m = N;
    const int lda = N;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;
    cusolverDnParams_t params = NULL;


    T* d_A = nullptr;
    T* d_W = nullptr;
    int* d_info = nullptr;

    int info = 0;

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void* d_work = nullptr;              /* device workspace */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void* h_work = nullptr;              /* host workspace for */

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_A), sizeof(T) * N * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_W), sizeof(T) * N));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_info), sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_A, &H_mat[0], sizeof(T) * N * N, cudaMemcpyHostToDevice,
        stream));


    // step 3: query working space of syevd
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

    CUSOLVER_CHECK(cusolverDnXsyevd_bufferSize(
        cusolverH, params, jobz, uplo, m, traits<T>::cuda_data_type, d_A, lda,
        traits<T>::cuda_data_type, d_W, traits<T>::cuda_data_type, &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_work), workspaceInBytesOnDevice));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void*>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }

    // step 4: compute spectrum
    CUSOLVER_CHECK(cusolverDnXsyevd(
        cusolverH, params, jobz, uplo, m, traits<T>::cuda_data_type, d_A, lda,
        traits<T>::cuda_data_type, d_W, traits<T>::cuda_data_type, d_work, workspaceInBytesOnDevice,
        h_work, workspaceInBytesOnHost, d_info));

    CUDA_CHECK(cudaMemcpyAsync(&V[0], d_A, sizeof(T) * N * N, cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(&W[0], d_W, sizeof(T) * N, cudaMemcpyDeviceToHost,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::printf("after Xsyevd: info = %d\n", info);
    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_W));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));
    CUDA_CHECK(cudaStreamDestroy(stream));
    //CUDA_CHECK(cudaDeviceReset());

}
template void SymmDiag_cuSOLVER_wrapper<float>(float* __restrict H_mat, float* __restrict W, float* __restrict V, const int N);