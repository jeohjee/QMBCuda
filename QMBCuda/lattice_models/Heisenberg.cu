#include "Heisenberg.h"
#include "thrust/complex.h"
#include "../quantum_operators/ArbitraryOperator.h"

using complex_th = thrust::complex<float>;

template <typename T> Heisenberg<T>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<T> _heisenberg_info)
{
	geom_info = _geom_info;
	CreateGeometry();
	ComputeR_diffs();

	heisenberg_info = _heisenberg_info;
	CreateHamiltonian(heisenberg_info);
}
template Heisenberg<float>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<float> _heisenberg_info);
template Heisenberg<complex_th>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<complex_th> _heisenberg_info);


template <typename T>
void Heisenberg<T>::CreateHamiltonian(HeisenbergInfo<T> heisenberg_info) 
{
	// This function implements the creation of a generic XYZ Heisenberg lattice Hamiltonian.

	

}