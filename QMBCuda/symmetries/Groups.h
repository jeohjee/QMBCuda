#pragma once
#include "../geometry/base.h"
#include "SymmetryClass.h"
#include "thrust/complex.h"
#include <vector>

typedef SymmetryClass<thrust::complex<float>> SymmetryClassC;

class Cn_group : public SymmetryClassC
{
public:

	Cn_group(int n_in, std::vector<float> R0_in);
	virtual ~Cn_group() {};

	int get_n();

protected:
	int n; // the degree of the rotational group, e.g. for C2 group n=2
	std::vector<float> R0; // The center of the rotation
};

// translational group:
class Tnm_group : public SymmetryClassC
{
public:

	Tnm_group(LatticeGeometryInfo _geom_info);
	virtual ~Tnm_group() {};

	int get_n();
	int get_m();

protected:
	int N;
	int M;
	LatticeGeometryInfo geom_info;
};