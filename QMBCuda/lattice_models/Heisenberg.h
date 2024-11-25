#pragma once
#include "ArbitrarySpinLattice.h"
#include "../geometry/base.h"

// The following class represents a generic (real- or complex-valued) XYZ extended Heisenberg modelm
// where the spin-spin interaction depends on the distances between the sites.

template <typename T>
struct HeisenbergInfo {

	std::vector<std::vector<T>> J_vecs; 
	// first index corresponds to the Range_vec indices, 
	//second index is Jx, Jy, Jz, respectively 

	std::vector<float> Range_vec; // Gives the range of the interaction terms.
	bool intra_c_bool_in; 
	bool inter_c_bool_in;
};


template <typename T>
class Heisenberg : public SpinHamiltonian<T>
{
public:

	Heisenberg() {};
	Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<T> _heisenberg_info);

	void CreateHamiltonian(HeisenbergInfo<T> heisenberg_info);

protected:
	LatticeGeometryInfo geom_info;
	HeisenbergInfo<T> heisenberg_info;
};

