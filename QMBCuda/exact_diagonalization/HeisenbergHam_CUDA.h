#pragma once
#include <vector>
#include "../lattice_models/T_standard.h"

// Brute force XXZ Heisenberg Hamiltonian with CUDA. Not optimized with symmetries but can be used as debugging tool.
// Supports only real-valued Hamiltonians.


class HeisenbergHam_CUDA
{
public:

	HeisenbergHam_CUDA(
		T_standard<float>T_mat_in, 
		std::vector<float> B_field_in, 
		std::vector<float> J_dim_weights, 
		int GS_sector_in, 
		int hop_f, 
		int NStates = 1,
		long long seed = 2212456,
		const int NIter = 50);
	virtual ~HeisenbergHam_CUDA() {};

protected:
	T_standard<float> T_mat; //For the standard Heisenberg Hamiltonian, other kinds of Lattice components would not make sense
	vector<float> B_field; // Vector containing the on-site magnetic fields. This makes also the CPT possible.
	int LS; //gives the size of the lattice. This is included in the members of T_vec but it's better to have a direct access to this
	vector<float> J_weights; // relative J_x, J_y and J_z
	int GS_sector; // the total spin-up sector (number of spin up sites) where the ground state is assumed to be found. Usually LS/2. This value is changed in the process of CPT to obtain the Green's functions

};


