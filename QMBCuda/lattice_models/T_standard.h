#pragma once
#include "Lattice.h"
#include <vector>
#include "../geometry/base.h"

using namespace std;

/*
This subclass of Lattice presents hopping Hamiltonian for which the hopping terms depend only on the spatial distance between
the lattice sites. The class can be used to present the usual one-particle hopping Hamiltonian or extended Heisenberg Hamiltonian

*/

// T is either float or complex
template <typename T>
class T_standard : public Lattice
{
public:

	T_standard() {};
	T_standard(LatticeGeometryInfo _geom_info, std::vector<T> J_vec, std::vector<float> Range_vec, int intra_c_bool_in, int inter_c_bool_in);
	
	vector<vector<T>> getTmat() { return this->T_mat; }
	int getTSize() { return this->T_size; }

protected:

	std::vector<std::vector<T>> T_mat;
	int intra_c_bool;
	int inter_c_bool;
	int T_size;
	LatticeGeometryInfo geom_info;

};


