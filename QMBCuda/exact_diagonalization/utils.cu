#include "./utils.h"
#include "../utils/misc_funcs_gpu.h"
#include "thrust/complex.h"
#include "cooperative_groups.h"

using complex_th = thrust::complex<float>;

__device__ int GetIndexInHilbertSubspace(uint32_t target_state, int GS_sector) {

    int delta_x;
    int x_coord;

    int tmp_alpha = 0;
    double bin_coef;

    int curr_ind = 0;
    int tot_count = 32;
    for (long unsigned int i = 1 << 31; i > 0; i = i / 2) {
        if (target_state & i) {

            delta_x = tot_count - (GS_sector - curr_ind);
            x_coord = tot_count;
            curr_ind = curr_ind + 1;

            if (delta_x > 0) {
                bin_coef = NChoosek2_dev(static_cast<double>(x_coord - 1), static_cast<double>(delta_x - 1));
                tmp_alpha = tmp_alpha + static_cast<int>(bin_coef + 0.5);
            }

        }
        tot_count = tot_count - 1;

    }

    //printf("target_state: %d, final index: %d, GS sector: %d\n", (int)target_state, tmp_alpha, GS_sector);
    return tmp_alpha;
}


__global__ void SetUpSRS(uint32_t* __restrict bas_states, uint32_t* __restrict SRS_states,
    uint32_t* __restrict group_el, int nobv, int GSize, int LS, int GS_sector) {

    // Each thread is identified by the combined index of the basis state and the symmetry operator
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nobv * GSize) return;

    uint32_t col_ind = id % GSize;
    uint32_t bas_ind = id / GSize;

    uint32_t bas_state = bas_states[bas_ind];
    uint32_t final_state = 0;

    int site_ind = LS - 1;
    for (long unsigned int i = 1 << (LS - 1); i > 0; i = i / 2) {
        if (bas_state & i) {
            final_state = final_state + (1 << group_el[col_ind * LS + site_ind]);
        }
        site_ind -= 1;

    }
    int target_ind = GetIndexInHilbertSubspace(final_state, GS_sector);
    SRS_states[id] = static_cast<uint32_t>(target_ind);

    //printf("col_ind: %d, bas_ind: %d, bas_state: %d, final state: %d, ind: %d\n", (int)col_ind, (int)bas_ind, (int)bas_state, (int)final_state, target_ind);
}

__global__ void ComputeOrbitIndices(uint32_t* __restrict orbit_indices, int SRS_unique_count, int GSize, uint32_t* __restrict SRS_states,
    int GS_sector) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= SRS_unique_count * GSize) return;

    int orbit_ind = id / GSize;
    int intra_orbit_ind = id % GSize;

    int target_ind = (int)SRS_states[id];
    orbit_indices[target_ind * 2] = (uint32_t)orbit_ind;
    orbit_indices[target_ind * 2 + 1] = (uint32_t)intra_orbit_ind;

    //printf("id= %d, SRS_states[id] = %u, orbit_ind = %d, intra_orbit_ind = %d, target_ind = %d \n", id, SRS_states[id], orbit_ind, intra_orbit_ind, target_ind);
}

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
) {

    // TO DO: to move the input parameters to appropriate structs
    
    // Note that here the logic is slightly different than in simpler BuildHoppingHam_v2 in a sense that SRS_states_inds
    // labels the indices of the SRS states WITHIN the chosen Hilbert subspace

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= SRS_unique_count) return;

    // The first index in each row is kept for the on-site accumulation
    ind_mat[max_terms * id] = id;
    int track_ind = 1;

    float curr_norm = norm_vecs[id * NIr + alpha_ind];
    if (curr_norm < tol_norm) return; // in case of a null state

    T Jz_szsz_acc = 0;
    uint32_t source_state = basis_states[SRS_states_inds[id * GSize]];


    for (int ti2 = 0; ti2 < T_size; ti2++) {

        int i_ind = T_ind1_dev[ti2];
        int j_ind = T_ind2_dev[ti2];

        T J_z = T_val_dev[ti2] * Jz_w;
        T J_xy = T_val_dev[ti2] * Jxy_w;

        // First the Sz Sz term:
        bool i_spin = source_state & (1 << i_ind);
        bool j_spin = source_state & (1 << j_ind);

        Jz_szsz_acc += J_z * 0.25 * (-1.0 + 2.0 * (T)((float)i_spin)) * (-1.0 + 2.0 * (T)((float)j_spin));

        // check j_ind and i_ind bits
        if (!(source_state & (1 << i_ind))) continue;
        if (source_state & (1 << j_ind)) continue;

        // Write also possible J_xy terms:
        uint32_t target_state = source_state | (uint32_t)(1 << j_ind);
        target_state = target_state & ~((uint32_t)1 << i_ind);
        int target_ind = GetIndexInHilbertSubspace(target_state, GS_sector);

        // One needs to solve now to which orbit the target_state belongs to
        uint32_t orbit_ind = orbit_indices[target_ind * 2];
        int Gm_ind = (int)orbit_indices[target_ind * 2 + 1]; // index within the orbit

        if (norm_vecs[orbit_ind * NIr + alpha_ind] < tol_norm) continue; // in case target orbit is a null state

        int id_write = max_terms * id + track_ind;

        ind_mat[id_write] = orbit_ind;
        //printf("ind_mat[id_write] = %u, orbit_ind = %u \n", ind_mat[id_write], orbit_ind);

        float norm_term = norm_vecs[orbit_ind * NIr + alpha_ind] / norm_vecs[id * NIr + alpha_ind];
        //val_mat[id_write] = J_xy*char_mat[GSize*alpha_ind + Gm_ind]/(((float)GSize) * norm_term);
        val_mat[id_write] = J_xy * char_mat[GSize * alpha_ind + Gm_ind] * norm_term;

        //printf("off-d term is (%f,%f), norm_term is: %f, 1st norm is: %f, 2nd norm is %f, char term: (%f,%f)\n", ((complex_th)val_mat[id_write]).real(),
        //    ((complex_th)val_mat[id_write]).imag(), norm_term, norm_vecs[orbit_ind * NIr + alpha_ind], norm_vecs[id * NIr + alpha_ind],
        //    ((complex_th)char_mat[GSize * alpha_ind + Gm_ind]).real(),
        //    ((complex_th)char_mat[GSize * alpha_ind + Gm_ind]).imag());
        track_ind += 1; //update the current index
    }

    val_mat[max_terms * id] = Jz_szsz_acc;
    track_vec[id] = track_ind;
    //printf("norm of the current term is: %f, acc. Jz is %f and the diagonal term is %f \n", norm_vecs[id * NIr + alpha_ind], Jz_szsz_acc, val_mat[max_terms * id]);

}

template
__global__ void BuildHamiltonianForAbelianGroup<float>(
    float Jxy_w,
    float Jz_w,
    int GS_sector,
    uint32_t* __restrict basis_states,
    uint32_t* __restrict ind_mat,
    float* __restrict val_mat,
    short int* __restrict  track_vec,
    int max_terms,
    int SRS_unique_count,
    int* __restrict T_ind1_dev,
    int* __restrict T_ind2_dev,
    float* __restrict T_val_dev,
    int T_size,
    uint32_t* __restrict SRS_states_inds,
    int GSize,
    uint32_t* __restrict orbit_indices,
    float* __restrict char_mat,
    int alpha_ind,
    float* __restrict norm_vecs,
    int NIr,
    float tol_norm
);

template
__global__ void BuildHamiltonianForAbelianGroup<complex_th>(
    float Jxy_w,
    float Jz_w,
    int GS_sector,
    uint32_t* __restrict basis_states,
    uint32_t* __restrict ind_mat,
    complex_th* __restrict val_mat,
    short int* __restrict  track_vec,
    int max_terms,
    int SRS_unique_count,
    int* __restrict T_ind1_dev,
    int* __restrict T_ind2_dev,
    complex_th* __restrict T_val_dev,
    int T_size,
    uint32_t* __restrict SRS_states_inds,
    int GSize,
    uint32_t* __restrict orbit_indices,
    complex_th* __restrict char_mat,
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
) {
    // Brute force Heisenberg Hamiltonian without using any symmetry groups.

    // TO DO: to move the input parameters to appropriate structs

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nobv) return;

    // The first index in each row is kept for the on-site accumulation
    ind_mat[max_terms * id] = id;
    int track_ind = 1;

    float Jz_szsz_acc = 0;

    for (int ti2 = 0; ti2 < T_size; ti2++) {

        int i_ind = T_ind1_dev[ti2];
        int j_ind = T_ind2_dev[ti2];

        float J_z = T_val_dev[ti2] * Jxy_w;
        float J_xy = T_val_dev[ti2] * Jz_w;

        // Wed need to first add the SzSz term:

        bool i_spin = basis_states[id] & (1 << i_ind);
        bool j_spin = basis_states[id] & (1 << j_ind);

        Jz_szsz_acc += J_z * 0.25 * (-1.0 + 2.0 * static_cast<float>(i_spin)) * (-1.0 + 2.0 * static_cast<float>(j_spin));

        // check j_ind and i_ind bits
        if (!(basis_states[id] & (1 << i_ind))) continue;
        if (basis_states[id] & (1 << j_ind)) continue;

        // Write also possible J_xy terms:
        uint32_t target_state = basis_states[id] | (uint32_t)(1 << j_ind);
        target_state = target_state & ~((uint32_t)1 << i_ind);

        //int target_ind = GetIndexInHilbertSubspace(target_state, GS_sector);
        int target_ind = GetIndexInHilbertSubspace(target_state, GS_sector);

        int id_write = max_terms * id + track_ind;

        //ind_mat[id_write] = target_ind;
        ind_mat[id_write] = static_cast<uint32_t>(target_ind);
        val_mat[id_write] = J_xy;

        track_ind += 1; //update the current index
    }
    val_mat[max_terms * id] = Jz_szsz_acc;
    track_vec[id] = track_ind;
}


template<typename S>
__global__  void ExpecValArbOperatorAbelianIrrepProjHeisenbergZeroT(
    complex_th* __restrict A_scalar_table_dev, 
    int* __restrict A_decode_table_dev, 
    int* __restrict A_site_table_dev, 
    int* __restrict A_NOTerms_dev,
    int max_terms, 
    int SRS_unique_count, 
    uint32_t* __restrict SRS_states_inds, 
    int GSize, int GS_sector, 
    uint32_t* __restrict orbit_indices,
    complex_th* __restrict char_mat, 
    int alpha_ind,
    float* __restrict norm_vecs, 
    int NIr, 
    float tol_norm, 
    int A_size, 
    int A_col_size,
    uint32_t* __restrict basis_states, 
    S* __restrict phi_state, 
    complex_th* __restrict A_exp
) {
    // TO DO: to move the input parameters to appropriate structs

    // SRS_states_inds labels the indices of the SRS states WITHIN the chosen Hilbert subspace.
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= SRS_unique_count) return;

    float curr_norm = norm_vecs[id * NIr + alpha_ind];
    if (curr_norm < tol_norm) return; // in case of a null state

    complex_th val_tmp;
    complex_th A_tmp_val = complex_th(0.0, 0.0);

    bool interr_bool;
    for (int ai = 0; ai < A_size; ai++) {

        uint32_t target_state = basis_states[SRS_states_inds[id * GSize]]; // We start from the source state and operate one-by-one with the operators of the ai term

        interr_bool = false;
        for (int gi = 0; gi < GSize; gi++) {
            // For loop over gi accounts for different Fock states of the original orbit
            // Operate with gi to the target_state:
            target_state = basis_states[SRS_states_inds[id * GSize + gi]];
            complex_th alpha_coef = A_scalar_table_dev[ai]; // possible scalar, NOTE that final coefficient depends on gi (as we want "normalized" operator)!

            for (int ai2 = 0; ai2 < A_NOTerms_dev[ai]; ai2++) {

                int tmp_site = A_site_table_dev[ai * A_col_size + ai2];
                int tmp_action = A_decode_table_dev[ai * A_col_size + ai2];
                bool tmp_spin = target_state & (1 << tmp_site); // Get the current spin for the enquired site

                // This implementation is specific for the spin (Heisenberg) Hamiltonians:

                // DOES NOT SUPPORT IDENTITY OPERATOR -> FIX AT SOME POINT

                // First the lowering operator:
                if (tmp_action < 0) {
                    if (!(target_state & (1 << tmp_site))) {
                        interr_bool = true;
                        break;
                    }
                    target_state = target_state & ~((uint32_t)1 << tmp_site);
                }
                // Raising operator:
                if (tmp_action > 0) {
                    if ((target_state & (1 << tmp_site))) {
                        interr_bool = true;
                        break;
                    }
                    target_state = target_state | (uint32_t)(1 << tmp_site);
                }
                // Sz operator
                if (tmp_action == 0) {
                    // For Sz, we need to multiply coef by -1 for spin-down:
                    alpha_coef = alpha_coef * (0.5) * (-1.0 + 2.0 * (complex_th)(tmp_spin));
                }
            }
            if (interr_bool) break;

            int target_ind = GetIndexInHilbertSubspace(target_state, GS_sector);
            // One needs to solve now to which orbit the target_state belongs to
            uint32_t orbit_ind = orbit_indices[target_ind * 2];
            int Gm_ind = (int)orbit_indices[target_ind * 2 + 1]; // index within the orbit

            if (norm_vecs[orbit_ind * NIr + alpha_ind] < tol_norm) continue; // in case target orbit is a null state

            float norm_term = norm_vecs[orbit_ind * NIr + alpha_ind] / curr_norm;

            thrust::complex<float> char_term = thrust::complex<float>(char_mat[GSize * alpha_ind + gi]);
            GetComplexConjugate(&char_term);

            val_tmp = alpha_coef * norm_term * char_term * char_mat[GSize * alpha_ind + Gm_ind];
            GetComplexConjugate(&val_tmp);

            // We complex-conjugate in the previous line as we implicity flip the order of the rows and columns, i.e. take the transpose. 
            // As it is more useful to take take the hermitian conjugate, we performed the complex-conjugation.

            thrust::complex<float> phi_conj(phi_state[id]);
            GetComplexConjugate(&phi_conj);

            A_tmp_val = A_tmp_val + phi_conj * val_tmp * (complex_th(phi_state[orbit_ind]));

            //printf("val_tmp: (%f, %f), phi_conj: (%f, %f), id: %d \n", val_tmp.real(), val_tmp.imag(), phi_conj.real(), phi_conj.imag(), id);
        }
    }
    A_exp[id] = A_tmp_val;
    //printf("A_tmp_val: (%f, %f), id: %d / %d \n", A_tmp_val.real(), A_tmp_val.imag(), id, SRS_unique_count);
}


template
__global__  void ExpecValArbOperatorAbelianIrrepProjHeisenbergZeroT<float>(
    complex_th* __restrict A_scalar_table_dev,
    int* __restrict A_decode_table_dev,
    int* __restrict A_site_table_dev,
    int* __restrict A_NOTerms_dev,
    int max_terms,
    int SRS_unique_count,
    uint32_t* __restrict SRS_states_inds,
    int GSize, int GS_sector,
    uint32_t* __restrict orbit_indices,
    complex_th* __restrict char_mat,
    int alpha_ind,
    float* __restrict norm_vecs,
    int NIr,
    float tol_norm,
    int A_size,
    int A_col_size,
    uint32_t* __restrict basis_states,
    float* __restrict phi_state,
    complex_th* __restrict A_exp
);

template
__global__  void ExpecValArbOperatorAbelianIrrepProjHeisenbergZeroT<complex_th>(
    complex_th* __restrict A_scalar_table_dev,
    int* __restrict A_decode_table_dev,
    int* __restrict A_site_table_dev,
    int* __restrict A_NOTerms_dev,
    int max_terms,
    int SRS_unique_count,
    uint32_t* __restrict SRS_states_inds,
    int GSize, int GS_sector,
    uint32_t* __restrict orbit_indices,
    complex_th* __restrict char_mat,
    int alpha_ind,
    float* __restrict norm_vecs,
    int NIr,
    float tol_norm,
    int A_size,
    int A_col_size,
    uint32_t* __restrict basis_states,
    complex_th* __restrict phi_state,
    complex_th* __restrict A_exp
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
) {
    // Each thread corresponds to one of the elements of the output vector
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nobv) return;

    T res = (T)0.0;
    int NElems = static_cast<int>(track_vec[id]);

    for (int i = 0; i < NElems; i++) {
        int tmp_ind = static_cast<int>(ind_mat[max_terms * id + i]);
        res += val_mat[max_terms * id + i] * in_vec[tmp_ind];
    }
    out_vec[id] = res;
    //delete tmp_ind_vec;
}

template
__global__ void SparseMatDenseVec_prod_EEL<float>(
    uint32_t* __restrict ind_mat,
    float* __restrict val_mat,
    short int* __restrict  track_vec,
    float* __restrict in_vec,
    float* __restrict out_vec,
    int nobv,
    int max_terms
);

template
__global__ void SparseMatDenseVec_prod_EEL<complex_th>(
    uint32_t* __restrict ind_mat,
    complex_th* __restrict val_mat,
    short int* __restrict  track_vec,
    complex_th* __restrict in_vec,
    complex_th* __restrict out_vec,
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
) {

    for (int id = 0; id < nobv; id++) {

        float res = 0.0;
        int NElems = static_cast<int>(track_vec[id]);

        for (int i = 0; i < NElems; i++) {
            int tmp_ind = static_cast<int>(ind_mat[max_terms * id + i]);
            res += val_mat[max_terms * id + i] * in_vec[tmp_ind];
        }
        out_vec[id] = res;
    }
}


// The following function computes out_vec[id] = x[id] * (Hy)[id]. To complete the dot product, one needs to call a sum function 
__global__ void DenseVecSparseMatDenseVec_prod_EEL(
    uint32_t* __restrict ind_mat, 
    float* __restrict val_mat, 
    short int* __restrict  track_vec, 
    float* __restrict in_vec, 
    float* __restrict in_vec2, 
    float* __restrict out_vec, 
    int nobv, 
    int max_terms
) {

    // Each thread corresponds to one of the elements of the output vector
    auto grid = cooperative_groups::this_grid();
    auto block = cooperative_groups::this_thread_block();
    auto warp = cooperative_groups::tiled_partition<32>(block);

    float res = 0.0;

    for (int tid = grid.thread_rank(); tid < nobv; tid += grid.size()) {
        int NElems = static_cast<int>(track_vec[tid]);

        float res2 = 0.0;

        for (int i = 0; i < NElems; i++) {
            int tmp_ind = static_cast<int>(ind_mat[max_terms * tid + i]);
            res2 += val_mat[max_terms * tid + i] * in_vec[tmp_ind];
        }

        res += res2 * in_vec2[tid];
    }

    warp.sync(); // CHECK IF THIS IS FEASIBLE FOR A LARGE FOCK SPACE
    res += warp.shfl_down(res, 16);
    res += warp.shfl_down(res, 8);
    res += warp.shfl_down(res, 4);
    res += warp.shfl_down(res, 2);
    res += warp.shfl_down(res, 1);

    if (warp.thread_rank() == 0) atomicAdd(&out_vec[block.group_index().x], res);
}


__global__ void SetUpSRS_from_indices(uint32_t* __restrict indices, uint32_t* __restrict SRS_states,
    uint32_t* __restrict group_el, int nobv, int GSize, int LS, int GS_sector, uint32_t* __restrict bas_states) {

    // Each thread is identified by the combined index of the basis state and the symmetry operator
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nobv * GSize) return;

    uint32_t col_ind = id % GSize;
    uint32_t bas_ind = id / GSize;

    uint32_t bas_state = bas_states[indices[bas_ind]];
    uint32_t final_state = 0;

    int site_ind = LS - 1;
    for (long unsigned int i = 1 << (LS - 1); i > 0; i = i / 2) {
        if (bas_state & i) {
            //final_state = final_state + (1 << symm_op[site_ind]);
            final_state = final_state + (1 << group_el[col_ind * LS + site_ind]);
        }
        site_ind -= 1;

    }
    //printf("from indices:\n");
    int target_ind = GetIndexInHilbertSubspace(final_state, GS_sector);
    SRS_states[id] = static_cast<uint32_t>(target_ind);

    //printf("col_ind: %d, bas_ind: %d, bas_state: %d, final state: %d, ind: %d\n", (int)col_ind, (int)bas_ind, (int)bas_state, (int)final_state, (int)SRS_states[id]);
    //printf("col_ind: %d, bas_ind: %d, bas_state: %d, final state: %d, ind: %d\n", (int)col_ind, (int)bas_ind, (int)bas_state, (int)final_state, target_ind);
    //printf("col_ind: %d, bas_ind: %d, bas_state: %d, final state: %zu, ind: %d\n", (int)col_ind, (int)bas_ind, (int)bas_state, final_state, target_ind);
}




__global__ void SolveSRS_min(uint32_t* __restrict SRS_states, uint32_t* SRS_states_min, int nobv, int GSize) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= nobv) return;

    uint32_t min_state = SRS_states[id * GSize];
    uint32_t curr_comp;
    for (int i = 1; i < GSize; i++) {
        curr_comp = SRS_states[id * GSize + i];
        if (min_state > curr_comp) min_state = curr_comp;
    }
    SRS_states_min[id] = min_state;
}



// This function computes the norm of the SRS state for each orbit AND each irrep.
template <typename T>
__global__ void ComputeSRSNorms(
    float* __restrict norm_vecs, 
    uint32_t* __restrict SRS_states, 
    T* __restrict char_mat, 
    int SRS_unique_count, 
    int NIr, 
    int GSize,
    uint32_t* __restrict bas_states
) {

    // Character table is usually so small that it should fit the shared memory
    //extern __shared__ float char_mat_sh[];
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id >= SRS_unique_count * NIr) return;

    int alpha = id % NIr;
    int orbit = id / NIr;

    complex_th norm = complex_th(0.0, 0.0); // char mat might be complex

    for (int i = 0; i < GSize; i++) {
        uint32_t i_state = bas_states[SRS_states[orbit * GSize + i]];
        for (int j = 0; j < GSize; j++) {
            uint32_t j_state = bas_states[SRS_states[orbit * GSize + j]];
            if (i_state == j_state) {

                T tmp_char = char_mat[alpha * GSize + i];
                GetComplexConjugate(&tmp_char);
                norm = norm + tmp_char * char_mat[alpha * GSize + j];
            }
        }
    }
    float norm_real = sqrt(fabs(norm.real())) / ((float)GSize); // we know norm is real
    //printf("norm is: %f, states are: (%d,%d), indices are: (%d, %d), irrep is %d \n", norm, (int)bas_states[SRS_states[orbit * GSize + 0]], (int)bas_states[SRS_states[orbit * GSize + 1]],
    //    (int)SRS_states[orbit * GSize + 0], (int)SRS_states[orbit * GSize + 1],alpha);
    //printf("norm is: %f, original norm is: (%f,%f)\n", norm_real, norm.real(), norm.imag());
    norm_vecs[orbit * NIr + alpha] = norm_real;
}

template __global__ void ComputeSRSNorms<float>(
    float* __restrict norm_vecs,
    uint32_t* __restrict SRS_states,
    float* __restrict char_mat,
    int SRS_unique_count,
    int NIr,
    int GSize,
    uint32_t* __restrict bas_states
);

template __global__ void ComputeSRSNorms<complex_th>(
    float* __restrict norm_vecs,
    uint32_t* __restrict SRS_states,
    complex_th* __restrict char_mat,
    int SRS_unique_count,
    int NIr,
    int GSize,
    uint32_t* __restrict bas_states
);