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

	int M = geom_info.rx_alpha.size(); // number of sublattices
	int N1 = geom_info.N1;
	int N2 = geom_info.N2;
	int N = N1 * N2; // number of unit cells
	int LS = N1 * N2 * M; // number of sites

	bool intra_c_bool = heisenberg_info.intra_c_bool;
	bool inter_c_bool = heisenberg_info.inter_c_bool;

	if ((intra_c_bool == 0) && (inter_c_bool == 0)) {
		printf("ERROR: BOTH INTER AND INTRA-CLUSTER BOOLEANS ZERO, ABORTING");
		return;
	}

	std::vector<std::vector<int>> Jxy_terms = heisenberg_info.Jxy_terms;
	std::vector<std::vector<int>> Jz_terms = heisenberg_info.Jz_terms;
	std::vector<T> Jxy_couplings = heisenberg_info.Jxy_couplings;
	std::vector<float> Jz_couplings = heisenberg_info.Jz_couplings;

	for (int uc = 0; uc < N; uc++) {

		for (int ji = 0; ji < Jxy_terms.size(); ji++) {
			int j_site = uc * M + Jxy_terms[ji][3];
			int i_site = j_site + Jxy_terms[ji][2] + M * (N2 * Jxy_terms[ji][0] + Jxy_terms[ji][1]);
			
			if (~inter_c_bool && (i_site < 0 || i_site >= LS)) continue;
			if (~intra_c_bool && (i_site >= 0 && i_site < LS)) continue;

			i_site = i_site % LS;
			
			Sp<T> Sp_i = Sp<T>(i_site);
			Sm<T> Sm_j = Sm<T>(j_site);
			H = H + Jxy_couplings[ji]*(Sp_i * Sm_j);
		}
		for (int ji = 0; ji < Jz_terms.size(); ji++) {
			int j_site = uc * M + Jz_terms[ji][3];
			int i_site = j_site + Jz_terms[ji][2] + M * (N2 * Jz_terms[ji][0] + Jz_terms[ji][1]);

			if (~inter_c_bool && (i_site < 0 || i_site >= LS)) continue;
			if (~intra_c_bool && (i_site >= 0 && i_site < LS)) continue;

			i_site = i_site % LS;
			
			Sz<T> Sz_i = Sz<T>(i_site);
			Sz<T> Sz_j = Sz<T>(j_site);
			H = H + Jz_couplings[ji] * (Sz_i * Sz_j);
		}
	}

}
template void Heisenberg<float>::CreateHamiltonian(HeisenbergInfo<float> heisenberg_info);
template void Heisenberg<complex_th>::CreateHamiltonian(HeisenbergInfo<complex_th> heisenberg_info);

Heisenberg<float> CreateHeisenbergXXXSquare(int N1, int N2, std::vector<float> J_vec)
{
	LatticeGeometryInfo geom_info = create_square_lattice_info(N1, N2);
	std::vector<std::vector<float>> A_mat = geom_info.A_mat;
	HeisenbergInfo<float> heisenberg_info;


	std::vector<int> i_vec;
	std::vector<int> j_vec;
	std::vector<int> dist_vec;
	for (int ni = -N1; ni < N1; ni++) {
		for (int nj = -N2; nj < N2; nj++) {
			i_vec.push_back(ni);
			j_vec.push_back(nj);
			float tmp_dist = pow((float)ni * A_mat[0][0] + (float)nj * A_mat[0][1], 2) + pow((float)ni * A_mat[1][0] + (float)nj * A_mat[1][1], 2);
			dist_vec.push_back(tmp_dist);
		}
	}
	std::stable_sort(i_vec.begin(), i_vec.end(),
		[&dist_vec](size_t i1, size_t i2) {return dist_vec[i1] < dist_vec[i2]; });
	std::stable_sort(j_vec.begin(), j_vec.end(),
		[&dist_vec](size_t i1, size_t i2) {return dist_vec[i1] < dist_vec[i2]; });

	float J_curr = J_vec[0];
	int JSize = J_vec.size();
	int J_counter = -1;
	float dist_curr = dist_vec[0];
	float dist_thold = 0.001;

	for (int i = 0; i < N1 * N2; i++) {
		if (dist_vec[i] < dist_thold) continue;
		if (dist_vec[i] > dist_curr + dist_thold) {
			J_counter = J_counter + 1;
			dist_curr = dist_vec[i];
			if (J_counter >= JSize) continue;
			J_curr = J_vec[J_counter];
		}
		heisenberg_info.Jxy_terms.push_back({ i_vec[i],j_vec[i],0,0 });
		heisenberg_info.Jz_terms.push_back({ i_vec[i],j_vec[i],0,0 });
		heisenberg_info.Jxy_couplings.push_back(J_curr);
		heisenberg_info.Jz_couplings.push_back(J_curr);
	}

	return Heisenberg<float>(geom_info, heisenberg_info);

}