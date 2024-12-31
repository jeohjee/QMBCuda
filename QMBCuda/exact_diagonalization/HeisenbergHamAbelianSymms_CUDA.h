#pragma once
#include "thrust/device_vector.h"
#include "thrust/complex.h"
#include "../lattice_models/T_standard.h"
#include "../symmetries/SymmetryClass.h"
#include "../quantum_operators/ArbitraryOperator.h"
#include "utils.h"


/* At the moment, the class supports only XXZ model, so set J_weights accordingly when creating an instance.
* Generalization to XYZ model is straightforward after one goes trhough the calculation with pen and paper.
 This class supports also complex-valued Hamiltonians.
*/


// template variable T accounts for both the type of the hopping and character matrix. It can be either float or thrust::complex<float>
template <typename T>
class HeisenbergHamAbelianSymms_CUDA
{
public:

	HeisenbergHamAbelianSymms_CUDA(T_standard<T> T_mat_in, std::vector<float> B_field_in, std::vector<float> J_dim_weights, 
		int GS_sector_in, SymmetryClass<T> * SymmGroup_in, int hop_f_in, int NStates_in = 1, long long seed = 2212456,
		const int NIter_in = 50, float tol_norm_in = 0.00001);
	
	virtual ~HeisenbergHamAbelianSymms_CUDA() {
		delete[] group_el_arr;
		delete[] char_mat_arr;
	};

	// To avoid unnecessary complications with the template functions, we just take the most general case, i.e. complex here:
	thrust::complex<float> ComputeStaticExpValZeroT(ManyBodyOperator<thrust::complex<float>> A, int max_terms = 0);

	std::vector<std::vector<float>> get_E_vecs() { return this->E_vecs; }
	std::vector<std::vector<long long>> get_seed_vecs() { return this->seed_vecs; }

protected:

	T_standard<T> T_mat; //For the standard Heisenberg Hamiltonian, other kinds of Lattice components would not make sense
	std::vector<float> B_field; // Vector containing the on-site magnetic fields. This makes also the CPT possible.
	int LS; //gives the size of the lattice. This is included in the members of T_vec but it's better to have a direct access to this
	std::vector<float> J_weights; // relative J_x, J_y and J_z
	int GS_sector; // the total spin-up sector (number of spin up sites) where the ground state is assumed to be found. Usually LS/2. This value is changed in the process of CPT to obtain the Green's functions
	int nobv_gs; // The size of the Hilbert space within the GS_sector
	int hop_f;
	int NIter;
	SymmetryClass<T> * SymmGroup;
	
	uint32_t * group_el_arr;
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

	
	void BuildTmats(int T_size, thrust::device_vector<int>& T_ind1_dev,
		thrust::device_vector<int>& T_ind2_dev, thrust::device_vector<T>& T_val_dev);

	int BuildSRStates(SRSBuildInfo build_info,
		thrust::device_vector<uint32_t>& bas_states_dev,
		thrust::device_vector<uint32_t>& SRS_states,
		thrust::device_vector<uint32_t>& SRS_states_min,
		thrust::device_vector<uint32_t>& group_el_dev,
		thrust::device_vector<float>& norm_vecs,
		thrust::device_vector<uint32_t>& orbit_indices,
		thrust::device_vector<T>& char_mat_dev);

};

