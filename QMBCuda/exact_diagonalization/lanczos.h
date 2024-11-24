#pragma once
#include <vector>

// This function forms the tridiagonal Hamiltonian at the end of the Lanczos iteration. The format is row major
template <typename T> void FormTriMat(T* __restrict H_tri, float* __restrict a_vec, float* __restrict b_vec, int N);

template <typename T>
void LanczosEigenStateFromSeed(long long seed, 
    T* __restrict final_eig_vec_ptr,
    std::vector<float> eig_vec_Krylov, 
    int nobv, int NIter, 
    int max_terms,
    uint32_t* __restrict index_mat, 
    T* __restrict vals_mat, 
    short int* __restrict track_vec, 
    int threads);


// Lanczos
template <typename T>
float LanczosEigVals_CUDA(float* E_tmp, 
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
    bool checkEigState);

