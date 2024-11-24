#include "lanczos.h"
#include "../cuda_wrappers/CudaWrappers.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/complex.h"
#include "thrust/transform.h"
#include "curand_kernel.h"
#include "../utils/misc_funcs_gpu.h"
#include "../utils/misc_funcs.h"
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../utils/print_funcs.h"

using complex_th = thrust::complex<float>;

template <typename T> void FormTriMat(T* __restrict H_tri, float* __restrict a_vec, float* __restrict b_vec, int N) {

    for (int i = 0; i < N; i++) {
        H_tri[N * i + i] = static_cast<T>(a_vec[i]);
        if (i < N - 1) {
            H_tri[N * i + i + 1] = static_cast<T>(b_vec[i + 1]);
        }
        if (i > 0) {
            H_tri[N * i + i - 1] = static_cast<T>(b_vec[i]);
        }
    }
}
template void FormTriMat<float>(float* __restrict H_tri, float* __restrict a_vec, float* __restrict b_vec, int N);

template <typename T>
void LanczosEigenStateFromSeed(long long seed, 
    T* __restrict final_eig_vec_ptr,
    std::vector<float> eig_vec_Krylov, 
    int nobv, int NIter, 
    int max_terms,
    uint32_t* __restrict index_mat, 
    T* __restrict vals_mat,
    short int* __restrict track_vec,
    int threads) 
{

    float a_tmp;
    int blocks = (nobv + threads - 1) / threads;

    thrust::device_vector<T> phi_n_m1(nobv, 0.0);
    thrust::device_vector<T> phi_n(nobv, 1.0);
    thrust::device_vector<T> phi_n_p1(nobv, 0.0);

    // Set up a random quess for phi_n:
    thrust::device_vector<curandState> rand_state(nobv);
    init_random_vec << <blocks, threads >> > (seed, rand_state.data().get(), nobv, phi_n.data().get());
    cudaDeviceSynchronize();

    //printf("From seed: last few elements of phi: \n");
    //thrust::copy(phi_n.end()-15, phi_n.end() -1, std::ostream_iterator<float>(std::cout, "\n"));
    ///////////////////////////////////////////////////

    /////////////////////////////////
    // Normalize phi_n:
    float tmp_sum;
    float phi_n_norm_tmp = 0;

    tmp_sum = VecNorm_cuBLAS_generic_float_wrapper(
        thrust::raw_pointer_cast(phi_n.data()),
        nobv);

    float phi_n_norm = tmp_sum;
    printf("phi norm: %f with seed: %lli and (blocks, threads, nobv) = (%d,%d,%d)\n", phi_n_norm, seed, blocks, threads, nobv);
    thrust::transform(phi_n.begin(), phi_n.end(), phi_n.begin(), mul_thrust_vec<T>(1.0 / phi_n_norm));
    /////////////////////////////////

    thrust::device_ptr<T> final_eig_vec(final_eig_vec_ptr);

    for (int n = 0; n < NIter; n++) {
        if (n % 25 == 0) printf("Lanzcos loop: %d \n", n);

        thrust::transform(final_eig_vec, final_eig_vec + nobv, phi_n.begin(), final_eig_vec, saxpy_functor2<T>(eig_vec_Krylov[n]));

        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), phi_n_p1.begin(), mul_thrust_vec<T>(0.0));
        SparseMatDenseVec_prod_EEL << <blocks, threads >> > (index_mat, vals_mat, track_vec,
            phi_n.data().get(), phi_n_p1.data().get(), nobv, max_terms);
        cudaDeviceSynchronize();

        T a_tmp2 = DotProd_cuBLAS_generic_float_wrapper(
            thrust::raw_pointer_cast(phi_n.data()),
            thrust::raw_pointer_cast(phi_n_p1.data()),
            nobv);

        // We know that a_tmp is actually a real-valued number:
        if constexpr (::cuda::std::is_same_v<T, float>) a_tmp = a_tmp2;
        else a_tmp = a_tmp2.real();

        // use thrust's built-in functions to build the final phi_n+1 vec:
        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), phi_n_m1.begin(), phi_n_p1.begin(), saxpy_functor2<T>(-1.0 * phi_n_norm_tmp));
        thrust::copy(phi_n.begin(), phi_n.end(), phi_n_m1.begin()); // phi_{n-1} <- phi_{n} as phi_{n-1} is not needed in this iteration step anymore
        thrust::transform(phi_n.begin(), phi_n.end(), phi_n_p1.begin(), phi_n.begin(), saxpy_functor<T>(-1.0 * a_tmp)); // -a_n phi_n + phi_{n+1}. This will be phi_n in the next iteration

        phi_n_norm_tmp = VecNorm_cuBLAS_generic_float_wrapper(
            thrust::raw_pointer_cast(phi_n.data()),
            nobv);
        thrust::transform(phi_n.begin(), phi_n.end(), phi_n.begin(), mul_thrust_vec<T>(1.0 / phi_n_norm_tmp));
    }
}

template void LanczosEigenStateFromSeed<float>(long long seed,
    float* __restrict final_eig_vec_ptr,
    std::vector<float> eig_vec_Krylov,
    int nobv, int NIter,
    int max_terms,
    uint32_t* __restrict index_mat,
    float* __restrict vals_mat,
    short int* __restrict track_vec,
    int threads); // Keep threads in some high number such as 512 

template void LanczosEigenStateFromSeed<complex_th>(long long seed,
    complex_th* __restrict final_eig_vec_ptr,
    std::vector<float> eig_vec_Krylov,
    int nobv, int NIter,
    int max_terms,
    uint32_t* __restrict index_mat,
    complex_th* __restrict vals_mat,
    short int* __restrict track_vec,
    int threads); // Keep threads in some high number such as 512 


// Lanczos
template <typename T>
float LanczosEigVals_CUDA(
    float* E_tmp, 
    int NStates, 
    const int NIter, 
    int nobv, 
    int max_terms, 
    long long seed,
    uint32_t* __restrict index_mat,
    T* __restrict vals_mat,
    short int* __restrict track_vec,
    float* eig_vec_Krylov_out, 
    bool exportKrylovVec, 
    bool checkEigState) 
{
    // NStates should be always kept as NStates = 1 as higher states do not make sense in Lanczos. Excited states
    // can be addressed by adding lower eigenstate outer products to the Hamiltonian

    //int blocks = 64;
    //int threads = (nobv + blocks - 1) / blocks;

    int threads = 512;
    int blocks = (nobv + threads - 1) / threads;

    int blocks2 = pow2ceil((uint32_t)blocks);

    printf("threads per block: %d\n", threads);
    printf("number of blocks: %d, rounding to the next power of 2: %d \n", blocks, blocks2);

    // First we need to create three phi states for our iteration to create a_n and b_n coefficients
    thrust::device_vector<T> phi_n_m1(nobv, 0.0);
    thrust::device_vector<T> phi_n(nobv, 1.0);
    thrust::device_vector<T> phi_n_p1(nobv, 0.0);

    ///////////////////////////////////////////////////
    // 
    // Set up a random quess for phi_n:
    thrust::device_vector<curandState> rand_state(nobv);
    init_random_vec << <blocks, threads >> > (seed, rand_state.data().get(), nobv, phi_n.data().get());
    cudaDeviceSynchronize();

    printf("Last few elements of phi: \n");
    thrust::copy(phi_n.end() - 3, phi_n.end() - 1, std::ostream_iterator<T>(std::cout, "\n"));
    ///////////////////////////////////////////////////
    // Normalize phi_n:

    float tmp_sum;
    float phi_n_m1_norm = 0.0;
    float phi_n_norm_tmp = 0;

    tmp_sum = VecNorm_cuBLAS_generic_float_wrapper(
        thrust::raw_pointer_cast(phi_n.data()),
        nobv);

    float phi_n_norm = tmp_sum;
    printf("phi norm: %f with seed: %lli and (blocks, threads, nobv) = (%d,%d,%d)\n", phi_n_norm, seed, blocks, threads, nobv);

    thrust::transform(phi_n.begin(), phi_n.end(), phi_n.begin(), mul_thrust_vec<T>(1.0 / phi_n_norm));
    thrust::device_vector<T> phi_n0(nobv);
    thrust::copy(phi_n.begin(), phi_n.end(), phi_n0.begin());
    ///////////////////////////////////////////////////

    float* a_vec = (float*)malloc(sizeof(float) * NIter);
    float* b_vec = (float*)malloc(sizeof(float) * NIter);

    b_vec[0] = 0.0;

    float a_den;
    float a_num;
    float b2_den;
    float b2_num;

    // From this line onwards, phi vectors are actually normalized. To solve a_n and b_n correctly, we need to keep track of the vector norms..

    // This loop creates the a_n and b_n coefficients
    for (int n = 0; n < NIter; n++) {
        if (n % 25 == 0) printf("Lanzcos loop: %d \n", n);

        if (n > 0) {
            //b_vec[n] = phi_n_norm/phi_n_m1_norm;
            b_vec[n] = phi_n_norm_tmp;
        }

        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), phi_n_p1.begin(), mul_thrust_vec<T>(0.0));
        SparseMatDenseVec_prod_EEL << <blocks, threads >> > (index_mat, vals_mat, track_vec,
            phi_n.data().get(), phi_n_p1.data().get(), nobv, max_terms);
        cudaDeviceSynchronize();

        T a_tmp = DotProd_cuBLAS_generic_float_wrapper(
            thrust::raw_pointer_cast(phi_n.data()),
            thrust::raw_pointer_cast(phi_n_p1.data()),
            nobv);
        // We know that a_tmp is actually a real-valued number:
        if constexpr (::cuda::std::is_same_v<T, float>)  a_vec[n] = a_tmp;
        else a_vec[n] = a_tmp.real();

        // use thrust's built-in functions to build the final phi_n+1 vec:
        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), phi_n_m1.begin(), phi_n_p1.begin(), saxpy_functor2<T>(-1.0 * phi_n_norm_tmp));
        thrust::copy(phi_n.begin(), phi_n.end(), phi_n_m1.begin()); // phi_{n-1} <- phi_{n} as phi_{n-1} is not needed in this iteration step anymore
        thrust::transform(phi_n.begin(), phi_n.end(), phi_n_p1.begin(), phi_n.begin(), saxpy_functor<T>(-1.0 * a_vec[n])); // -a_n phi_n + phi_{n+1}. This will be phi_n in the next iteration

        phi_n_norm_tmp = VecNorm_cuBLAS_generic_float_wrapper(
            thrust::raw_pointer_cast(phi_n.data()),
            nobv);

        thrust::transform(phi_n.begin(), phi_n.end(), phi_n.begin(), mul_thrust_vec<T>(1.0 / phi_n_norm_tmp));

    }

    // test a_n and b_n
    for (int i = 0; i < NIter; i++) {
        if (i % 25) continue;
        printf("(i=%d,%f, %f): \n", i, a_vec[i], b_vec[i]);
    }

    ////////////////////////////////////////////
    // Next we need to solve the remaining tridiagonal matrix
    using data_type = float;

    // First form the tridiagonal Hamiltonian:
    data_type* H_tri = (data_type*)malloc(sizeof(data_type) * NIter * NIter);

    memset(H_tri, 0.0, sizeof(data_type) * NIter * NIter);
    FormTriMat<data_type>(H_tri, a_vec, b_vec, NIter);

    printMatrixRowMajor<data_type>(H_tri, NIter, 10);

    const int m = NIter;
    data_type* V = (data_type*)malloc(sizeof(data_type) * NIter * NIter); // eigenvectors
    data_type* W = (data_type*)malloc(sizeof(data_type) * NIter); // eigenvalues

    SymmDiag_cuSOLVER_wrapper<float>(H_tri, W, V, m); // Diagonalization

    std::printf("eigenvalue = (matlab base-1), ascending order\n");
    int idx = 1;
    for (int i = 0; i < 10; i++) {
        std::printf("W[%i] = %f\n", idx, W[i]);
        idx++;
    }

    float E_val = W[0];
    *E_tmp = W[0];

    float* eig_vec_Krylov = (float*)malloc(sizeof(float) * NIter);
    for (int i = 0; i < NIter; i++) eig_vec_Krylov[i] = V[i];

    std::vector<float> eig_vec_Krylov_std_vector(NIter);
    for (int i = 0; i < NIter; i++) eig_vec_Krylov_std_vector[i] = V[i];

    if (exportKrylovVec) {
        for (int i = 0; i < NIter; i++) eig_vec_Krylov_out[i] = eig_vec_Krylov[i];
    }

    if (checkEigState) {
        thrust::device_vector<T> final_eig_vec(nobv, 0.0);

        LanczosEigenStateFromSeed(seed, thrust::raw_pointer_cast(&final_eig_vec[0]),
            eig_vec_Krylov_std_vector, nobv, NIter, max_terms,
            index_mat, vals_mat, track_vec, threads);

        //thrust::copy(final_eig_vec.begin(), final_eig_vec.begin()+50, std::ostream_iterator<float>(std::cout, "\n"));

        //////////////
        // We know the eigenvalue is correct so we can directly check the accuracy of the obtained eigenvector:

        //Let us, for the sake of saving device memory, save H|n> to phi_n_p1
        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), phi_n_p1.begin(), mul_thrust_vec<T>(0.0));
        SparseMatDenseVec_prod_EEL << <blocks, threads >> > (index_mat, vals_mat, track_vec,
            final_eig_vec.data().get(), phi_n_p1.data().get(), nobv, max_terms);
        cudaDeviceSynchronize();
        thrust::transform(phi_n_p1.begin(), phi_n_p1.end(), final_eig_vec.begin(), phi_n_p1.begin(), saxpy_functor2<T>(-1.0 * E_val));
        phi_n_norm_tmp = VecNorm_cuBLAS_generic_float_wrapper(
            thrust::raw_pointer_cast(phi_n_p1.data()),
            nobv);

        phi_n_norm_tmp = phi_n_norm_tmp / nobv;

        printf("(H|n> - E|n>)/nobv = %.6f", phi_n_norm_tmp);

    }
    //////////////
    free(V);
    free(W);

    free(a_vec);
    free(b_vec);
    free(H_tri);
    free(eig_vec_Krylov);
    return phi_n_norm_tmp; // returns the eigenvalue error
}

template float LanczosEigVals_CUDA<float>(
    float* E_tmp,
    int NStates,
    const int NIter,
    int nobv,
    int max_terms,
    long long seed,
    uint32_t* __restrict index_mat,
    float* __restrict vals_mat,
    short int* __restrict track_vec,
    float* eig_vec_Krylov_out,
    bool exportKrylovVec,
    bool checkEigState);

template float LanczosEigVals_CUDA<complex_th>(
    float* E_tmp,
    int NStates,
    const int NIter,
    int nobv,
    int max_terms,
    long long seed,
    uint32_t* __restrict index_mat,
    complex_th* __restrict vals_mat,
    short int* __restrict track_vec,
    float* eig_vec_Krylov_out,
    bool exportKrylovVec,
    bool checkEigState);
// eig_vec_Krylov_out = nullptr, exportKrylovVec = true, checkEigState = true