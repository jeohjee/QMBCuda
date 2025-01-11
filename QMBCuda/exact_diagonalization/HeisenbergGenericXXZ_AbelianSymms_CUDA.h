#pragma once
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/complex.h"
#include "../lattice_models/T_standard.h"
#include "../symmetries/SymmetryClass.h"
#include "../quantum_operators/ArbitraryOperator.h"
#include "../lattice_models/Heisenberg.h"
#include "utils.h"


/* This ED solver supports arbitrary Heisenberg Hamiltonians that have the XXZ structure, i.e. Heisenberg Hamiltonians that conserve the total spin.
* Other than the limitation of having the XXZ structure, the spin-spin coupling terms can be arbitrary, also complex-valued for the in-plane coupling terms.
*/

template <typename T>
struct OperatorVectors {
	thrust::host_vector<OperatorType> H_decode_table;
	thrust::host_vector<T> H_scalar_table;
	thrust::host_vector<int> H_site_table; 
	thrust::host_vector<int> H_NOTerms;
	int H_size;
	int H_col_size;
};

template <typename T>
class HeisenbergXXZAbelianSymms_CUDA
{
public:

	HeisenbergXXZAbelianSymms_CUDA(
		Heisenberg<T> _lattice_model,
		SymmetryClass<T>* SymmGroup_in,
		int hop_f_in = -1,
		int GS_sector_in = -1,
		int NStates_in = 1, 
		long long seed = 2212456,
		const int NIter_in = 50, 
		float tol_norm_in = 0.00001);

	virtual ~HeisenbergXXZAbelianSymms_CUDA() {
		delete[] group_el_arr;
		delete[] char_mat_arr;
	};

	// To avoid unnecessary complications with the template functions, we just take the most general case, i.e. complex here:
	thrust::complex<float> ComputeStaticExpValZeroT(ManyBodyOperator<thrust::complex<float>> A, int max_terms = 0);

	std::vector<std::vector<float>> get_E_vecs() { return this->E_vecs; }
	std::vector<std::vector<long long>> get_seed_vecs() { return this->seed_vecs; }

protected:

	Heisenberg<T> lattice_model;
	int LS; //gives the size of the lattice. This is included in the members of T_vec but it's better to have a direct access to this
	int GS_sector; // the total spin-up sector (number of spin up sites) where the ground state is assumed to be found. Usually LS/2. This value is changed in the process of CPT to obtain the Green's functions
	int nobv_gs; // The size of the Hilbert space within the GS_sector
	int hop_f;
	int max_coupling_terms; // This tells the maximum number of Hamiltonian terms per Fock state
	int NIter;
	SymmetryClass<T>* SymmGroup;

	uint32_t* group_el_arr;
	T* char_mat_arr;

	int NStates; // Number of the lowest eigenstates per irrep.
	std::vector<std::vector<float>> E_vecs; // Eigenenergy vectors for each irrep.
	std::vector<std::vector<long long>> seed_vecs; // seed vectors for each irrep. With the seed one can in principle always obtain the eigenvectors again if needed without storing them in memory. 

	std::vector<std::vector<float>> KrylovGS_vecs;
	std::vector<float> KrylovGS_vec; // Krylov vector corresponding to the true ground state

	// As GS is so important, we list these params here explicitly:
	int GS_irrep;
	float E_gs;
	long long GS_seed;
	float tol_norm;

	void BuildBasisStates(thrust::device_vector<uint32_t>& bas_states_dev, unsigned long long int nobv);
	int BuildSRStates(SRSBuildInfo build_info,
		thrust::device_vector<uint32_t>& bas_states_dev,
		thrust::device_vector<uint32_t>& SRS_states,
		thrust::device_vector<uint32_t>& SRS_states_min,
		thrust::device_vector<uint32_t>& group_el_dev,
		thrust::device_vector<float>& norm_vecs,
		thrust::device_vector<uint32_t>& orbit_indices,
		thrust::device_vector<T>& char_mat_dev);
	OperatorVectors<T> BuildOperatorVecs(ManyBodyOperator<T> H);
};



template <typename T>
thrust::complex<float> HeisenbergXXZAbelianSymms_CUDA<T>::ComputeStaticExpValZeroT(
	ManyBodyOperator<thrust::complex<float>> A,
	int max_terms
);


