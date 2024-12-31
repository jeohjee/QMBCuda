#pragma once
#include "ArbitrarySpinLattice.h"
#include "../geometry/base.h"


// The following class represents a generic (real- or complex-valued) XXZ extended Heisenberg models
// where Jx and Jz can be arbitrary, with the limit of Jz being of course real.


template <typename T>
struct HeisenbergInfo {
	
	std::vector<std::vector<int>> Jxy_terms; 
	std::vector<T> Jxy_couplings;

	std::vector<std::vector<int>> Jz_terms;
	std::vector<float> Jz_couplings;

	/* In both Jxy_terms and Jz_terms each row is a 5 element array with the struct (i, j, a, b).
	Each row represents a single out-going coupling (the hermitian congujate terms, i.e. the in-coming couplings,
	are created automatically.)
	(i,j) refer to target unit cell indices RELATIVE TO THE ORIGINATING UNIT CELL 
	and (a,b) refer to sublattice indices. a means the target sublattice,
	b is the sublattice from where the hopping is originating. The corresponding coupling terms are in
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

	void CreateHamiltonian();

protected:
	HeisenbergInfo<T> heisenberg_info;
};


bool distance_comparator(const std::pair<int, float>& a, const std::pair<int, float>& b);

Heisenberg<float> CreateExtendedHeisenbergXXX(LatticeGeometryInfo geom_info, std::vector<float> J_vec, bool intra_c_bool = true, bool inter_c_bool = true);

Heisenberg<float> CreateHeisenbergXXXSquare(int N1, int N2, std::vector<float> J_vec, bool intra_c_bool = true, bool inter_c_bool = true);

Heisenberg<float> CreateHeisenbergXXXTriangular(int N1, int N2, std::vector<float> J_vec, bool intra_c_bool = true, bool inter_c_bool = true);


//Heisenberg<float> CreateHeisenbergXXXSquare(N1,N2,{J2,J3,J4...})
//Heisenberg<float> CreateHeisenbergXXXTriangular(N1,N2,{J2,J3,J4...})

//Heisenberg<float> CreateHeisenbergXXZSquare(N1,N2,{J1,J2,J3,J4...},{Jz2,Jz3,...})