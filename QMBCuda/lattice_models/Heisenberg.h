#pragma once
#include "ArbitrarySpinLattice.h"
#include "../geometry/base.h"
#include <algorithm> 
#include <cmath>

// The following class represents a generic (real- or complex-valued) XXZ extended Heisenberg models
// where Jx and Jz can be arbitrary, with the limit of Jz being of course real.

/*
template <typename T>
struct HeisenbergInfo {

	std::vector<std::vector<T>> J_vecs; 
	// first index corresponds to the Range_vec indices, 
	//second index is Jx, Jy, Jz, respectively 

	std::vector<float> Range_vec; // Gives the range of the interaction terms.
	bool intra_c_bool_in; 
	bool inter_c_bool_in;
};*/

template <typename T>
struct HeisenbergInfo {
	
	std::vector<std::vector<int>> Jxy_terms; 
	std::vector<T> Jxy_couplings;

	std::vector<std::vector<int>> Jz_terms;
	std::vector<float> Jz_couplings;

	/* In both Jxy_terms and Jz_terms each row is a 5 element array with the struct (i, j, a, b).
	Each row represents a single out-going coupling (the hermitian congujate terms, i.e. the in-coming couplings,
	are created automatically.
	(i,j) refer to target unit cell indices and (a,b) refer to sublattice indices. a means the target sublattice,
	b is the sublattice where the hopping is originating. The corresponding coupling terms are in
	Jxy_couplings and Jz_couplings.
	*/

	bool intra_c_bool;
	bool inter_c_bool;
};


template <typename T>
class Heisenberg : public SpinHamiltonian<T>
{
public:

	Heisenberg() {};
	Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<T> _heisenberg_info);

	void CreateHamiltonian(HeisenbergInfo<T> heisenberg_info);

protected:
	HeisenbergInfo<T> heisenberg_info;
};



// NEXT: BUILD FUNCTIONS TO CREATE HeisenbergInfo (or maybe even better Heisenberg) for THE USUAL EXTENDED HEISENBERG MODELS

Heisenberg<float> CreateHeisenbergXXXSquare(int N1, int N2, std::vector<float> J_vec);


//Heisenberg<float> CreateHeisenbergXXXSquare(N1,N2,{J2,J3,J4...})
//Heisenberg<float> CreateHeisenbergXXXTriangular(N1,N2,{J2,J3,J4...})

//Heisenberg<float> CreateHeisenbergXXZSquare(N1,N2,{J1,J2,J3,J4...},{Jz2,Jz3,...})