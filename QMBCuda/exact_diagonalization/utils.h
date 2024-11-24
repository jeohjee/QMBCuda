#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdint>
#include "thrust/complex.h"

__device__ int GetIndexInHilbertSubspace(uint32_t target_state, int GS_sector);

__global__ void SetUpSRS(uint32_t* __restrict bas_states, uint32_t* __restrict SRS_states,
    uint32_t* __restrict group_el, int nobv, int GSize, int LS, int GS_sector);

__global__ void SetUpSRS_from_indices(uint32_t* __restrict indices, uint32_t* __restrict SRS_states,
    uint32_t* __restrict group_el, int nobv, int GSize, int LS, int GS_sector, uint32_t* __restrict bas_states);

__global__ void SolveSRS_min(uint32_t* __restrict SRS_states, uint32_t* SRS_states_min, int nobv, int GSize);

__global__ void ComputeOrbitIndices(uint32_t* __restrict orbit_indices, int SRS_unique_count, int GSize, uint32_t* __restrict SRS_states,
    int GS_sector);

template <typename T>
__global__ void BuildHamiltonianForAbelianGroup(
    float Jxy_w,
    float Jz_w,
    int GS_sector,
    uint32_t* __restrict basis_states,
    uint32_t* __restrict ind_mat,
    T* __restrict val_mat,
    short int* __restrict  track_vec,
    int max_terms,
    int SRS_unique_count,
    int* __restrict T_ind1_dev,
    int* __restrict T_ind2_dev,
    T* __restrict T_val_dev,
    int T_size,
    uint32_t* __restrict SRS_states_inds,
    int GSize,
    uint32_t* __restrict orbit_indices,
    T* __restrict char_mat,
    int alpha_ind,
    float* __restrict norm_vecs,
    int NIr,
    float tol_norm
);

__global__ void BuildHoppingHam_v2(
    float Jxy_w,
    float Jz_w,
    int GS_sector,
    uint32_t* __restrict basis_states,
    uint32_t* __restrict ind_mat,
    float* __restrict val_mat,
    short int* __restrict  track_vec,
    int max_terms,
    int nobv,
    int* __restrict T_ind1_dev,
    int* __restrict T_ind2_dev,
    float* __restrict T_val_dev,
    int T_size
);

template<typename S>
__global__  void ExpecValArbOperatorAbelianIrrepProjHeisenbergZeroT(
    thrust::complex<float>* __restrict A_scalar_table_dev,
    int* __restrict A_decode_table_dev,
    int* __restrict A_site_table_dev,
    int* __restrict A_NOTerms_dev,
    int max_terms,
    int SRS_unique_count,
    uint32_t* __restrict SRS_states_inds,
    int GSize, int GS_sector,
    uint32_t* __restrict orbit_indices,
    thrust::complex<float>* __restrict char_mat,
    int alpha_ind,
    float* __restrict norm_vecs,
    int NIr,
    float tol_norm,
    int A_size,
    int A_col_size,
    uint32_t* __restrict basis_states,
    S* __restrict phi_state,
    thrust::complex<float>* __restrict A_exp
);

template <typename T>
__global__ void SparseMatDenseVec_prod_EEL(
    uint32_t* __restrict ind_mat,
    T* __restrict val_mat,
    short int* __restrict  track_vec,
    T* __restrict in_vec,
    T* __restrict out_vec,
    int nobv,
    int max_terms
);

void SparseMatDenseVec_prod_EEL_host(
    uint32_t* __restrict ind_mat,
    float* __restrict val_mat,
    short int* __restrict  track_vec,
    float* __restrict in_vec,
    float* __restrict out_vec,
    int nobv,
    int max_terms
);

__global__ void DenseVecSparseMatDenseVec_prod_EEL(
    uint32_t* __restrict ind_mat,
    float* __restrict val_mat,
    short int* __restrict  track_vec,
    float* __restrict in_vec,
    float* __restrict in_vec2,
    float* __restrict out_vec,
    int nobv,
    int max_terms
);

template <typename T>
__global__ void ComputeSRSNorms(
    float* __restrict norm_vecs,
    uint32_t* __restrict SRS_states,
    T* __restrict char_mat,
    int SRS_unique_count,
    int NIr,
    int GSize,
    uint32_t* __restrict bas_states
);