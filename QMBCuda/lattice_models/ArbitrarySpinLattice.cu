#include "ArbitrarySpinLattice.h"
#include <vector>
#include "thrust/complex.h"

using complex_th = thrust::complex<float>;

template <typename T>
SpinHamiltonian<T>::SpinHamiltonian(LatticeGeometryInfo _geom_info, ManyBodyOperator<T> _H)
{
	geom_info = _geom_info;
	H = _H;

	CreateGeometry();
	ComputeR_diffs();
}
template SpinHamiltonian<float>::SpinHamiltonian(LatticeGeometryInfo _geom_info, ManyBodyOperator<float> _H);
template SpinHamiltonian<complex_th>::SpinHamiltonian(LatticeGeometryInfo _geom_info, ManyBodyOperator<complex_th> _H);

template <typename T>
void SpinHamiltonian<T>::CreateGeometry()
{
	vector<float> a1_orig = { geom_info.A_mat[0][0], geom_info.A_mat[1][0] };
	vector<float> a2_orig = { geom_info.A_mat[0][1], geom_info.A_mat[1][1] };

	for (int i = 0; i < geom_info.N1; i++) {
		for (int j = 0; j < geom_info.N2; j++) {
			for (int alpha = 0; alpha < geom_info.rx_alpha.size(); alpha++) {
				float r_x = i * a1_orig[0] + j * a2_orig[0] + geom_info.rx_alpha[alpha];
				float r_y = i * a1_orig[1] + j * a2_orig[1] + geom_info.ry_alpha[alpha];
				R1.emplace_back(r_x);
				R2.emplace_back(r_y);
			}
		}
	}
	LS = R1.size();
}
template void SpinHamiltonian<float>::CreateGeometry();
template void SpinHamiltonian<complex_th>::CreateGeometry();

template <typename T>
ManyBodyOperator<T> SpinHamiltonian<T>::GetH()
{
	return H;
}
template ManyBodyOperator<float> SpinHamiltonian<float>::GetH();
template ManyBodyOperator<complex_th> SpinHamiltonian<complex_th>::GetH();