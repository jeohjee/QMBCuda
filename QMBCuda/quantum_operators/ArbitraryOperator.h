#pragma once
#include "SingleParticleOperators.h"
#include "../lattice_models/T_standard.h"
#include <vector>
#include "thrust/complex.h"

template <typename T>
class ManyBodyOperator
{
public:

	// This class represents arbitrary (many-body) operators built from single-particle operators
	ManyBodyOperator() {};
	ManyBodyOperator(std::vector<std::vector<Operator<T>>> _elements);
	virtual ~ManyBodyOperator() {};
	std::vector<std::vector<Operator<T>>> GetElems() const;
	int GetMaxTerms();
	void AddTerm(std::vector<Operator<T>> new_term);
	void AddSingleOperator(Operator<T> new_term);

	void LocalizeOperator(int new_site);
	void PrintOperator();

	ManyBodyOperator<T> operator+(ManyBodyOperator<T> const& A);
	ManyBodyOperator<T> operator*(ManyBodyOperator<T> const& A);
	ManyBodyOperator<T> operator*(float scalar);
	ManyBodyOperator<thrust::complex<float>> operator*(thrust::complex<float> scalar);


protected:
	std::vector<std::vector<Operator<T>>> elements; // Vector containing operators that are summed together to obtain the final operator
};

template <typename T>
ManyBodyOperator<T> operator*(float scalar, ManyBodyOperator<T>& A);

template <typename T>
ManyBodyOperator<thrust::complex<float>> operator*(thrust::complex<float> scalar, ManyBodyOperator<T>& A);

// SzSz correlator:
template <typename T>
class SzSz_correlator : public ManyBodyOperator<T>
{
public:

	// This class represents S^z_i S^z_j correlator operator

	SzSz_correlator(int i_in, int j_in); 
	virtual ~SzSz_correlator() {};

};


// Spin-z operator:
template <typename T>
class Sz : public ManyBodyOperator<T>
{
public:
	Sz(int i);
	virtual ~Sz() {};
};


// Spin-plus operator:
template <typename T>
class Sp : public ManyBodyOperator<T>
{
public:
	Sp(int i);
	virtual ~Sp() {};
};

// Spin-minus operator:
template <typename T>
class Sm : public ManyBodyOperator<T>
{
public:
	Sm(int i);
	virtual ~Sm() {};
};


// Fourier transform for any operator:
typedef ManyBodyOperator<thrust::complex<float>> ManyBodyOperatorC;

std::vector<ManyBodyOperatorC> FourierOperator(ManyBodyOperatorC A, std::vector<float> k, LatticeGeometryInfo geom_info);

std::vector< std::vector<std::vector<ManyBodyOperatorC>>>
FullFourierOperator(ManyBodyOperatorC A, LatticeGeometryInfo geom_info, bool neg_sqn);



