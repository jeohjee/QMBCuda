#pragma once
#include "Lattice.h"
#include "../quantum_operators/ArbitraryOperator.h"
#include "../geometry/base.h"

// The following class represents an arbitrary spin-1/2 lattice Hamiltonian with arbitrary
// terms. It is basically a Lattice wrapper around a ManyBodyOperator representing the actual Hamiltonian.

template <typename T>
class SpinHamiltonian : public Lattice
{
public:
	SpinHamiltonian() {};
	SpinHamiltonian(LatticeGeometryInfo _geom_info, ManyBodyOperator<T> _H);
	virtual ~SpinHamiltonian() {};

	void CreateGeometry();
	ManyBodyOperator<T> GetH();
protected:
	ManyBodyOperator<T> H; // This is the Hamiltonian of the lattice.
	LatticeGeometryInfo geom_info; // Hold the information about the geometry
};