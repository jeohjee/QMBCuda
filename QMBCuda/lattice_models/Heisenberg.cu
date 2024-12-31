#include "Heisenberg.h"
#include "thrust/complex.h"
#include "../quantum_operators/ArbitraryOperator.h"
#include <algorithm> 
#include <cmath>

using complex_th = thrust::complex<float>;

template <typename T> Heisenberg<T>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<T> _heisenberg_info)
{
	geom_info = _geom_info;
	CreateGeometry();
	ComputeR_diffs();

	heisenberg_info = _heisenberg_info;
	CreateHamiltonian();
}
template Heisenberg<float>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<float> _heisenberg_info);
template Heisenberg<complex_th>::Heisenberg(LatticeGeometryInfo _geom_info, HeisenbergInfo<complex_th> _heisenberg_info);


template <typename T>
void Heisenberg<T>::CreateHamiltonian() 
{
	// This function implements the creation of a generic XXZ Heisenberg lattice Hamiltonian.

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

	std::vector<std::vector<std::vector<int>>> J_terms = { heisenberg_info.Jxy_terms, heisenberg_info.Jz_terms };

	std::vector<T> Jxy_couplings = heisenberg_info.Jxy_couplings;
	std::vector<float> Jz_couplings = heisenberg_info.Jz_couplings;

	for (int uc = 0; uc < N; uc++) {
		int j_site_1 = uc / M;
		int j_site_2 = uc % M;

		//_add_couplings(uc, Jxy_terms, Jxy_couplings, Sz_bool)
		for (int jii = 0; jii < 2; jii++) {
			for (int ji = 0; ji < J_terms[jii].size(); ji++) {
				int j_site = uc * M + J_terms[jii][ji][3];

				int i_site_1 = j_site_1 + J_terms[jii][ji][0];
				int i_site_2 = j_site_2 + J_terms[jii][ji][1];
				if (!intra_c_bool && ( (i_site_1 < 0 || i_site_1 >= N1) || (i_site_2 < 0 || i_site_2 >= N2) )) continue;
				if (!inter_c_bool && ((i_site_1 >= 0 || i_site_1 < N1) || (i_site_2 >= 0 || i_site_2 < N2) )) continue;

				i_site_1 = ((i_site_1 % N1) + N1) % N1;
				i_site_2 = ((i_site_2 % N2) + N2) % N2;

				int i_site = J_terms[jii][ji][2] + M * (N2 * i_site_1 + i_site_2);

				//i_site = ((i_site % LS) + LS) % LS;

				if (jii == 0) {
					Sp<T> Sp_i = Sp<T>(i_site);
					Sm<T> Sm_j = Sm<T>(j_site);
					H = H + Jxy_couplings[ji] * (Sp_i * Sm_j);
				}
				else {
					Sz<T> Sz_i = Sz<T>(i_site);
					Sz<T> Sz_j = Sz<T>(j_site);
					H = H + Jz_couplings[ji] * (Sz_i * Sz_j);
				}
			}
		}
	}

}
template void Heisenberg<float>::CreateHamiltonian();
template void Heisenberg<complex_th>::CreateHamiltonian();


bool distance_comparator(const std::pair<int, float>& a, const std::pair<int, float>& b) {
	return a.second < b.second;
}

Heisenberg<float> CreateHeisenbergXXXSquare(int N1, int N2, std::vector<float> J_vec, bool intra_c_bool, bool inter_c_bool)
{
	/*
	* This function creates Heisenberg<float> instance for a extended square lattice XXX Heisenberg model. 
	* Args:
	*	N1, N2: lattice size in x and y-directions
	*	J_vec: contains the spin-spin coupling strengths in the descending order. First element is the NN coupling, the
	*	next one is NNN coupling and so forth.
	*	intra_c_bool: determines whether the spin-spin couplings within the bulk are used or ignored (in most cases this should be true)
	*	inter_c_bool: determines whether periodic boundary conditions are used (true) or not (false).
	* This function should be easily generalizable for arbitrary lattices.  
	*/

	LatticeGeometryInfo geom_info = create_square_lattice_info(N1, N2);
	std::vector<std::vector<float>> A_mat = geom_info.A_mat;
	HeisenbergInfo<float> heisenberg_info;


	std::vector<int> i_vec;
	std::vector<int> j_vec;
	std::vector<float> dist_vec;
	std::vector<int> indices;
	int curr_ind = 0;
	for (int ni = -N1; ni < N1; ni++) {
		for (int nj = -N2; nj < N2; nj++) {
			i_vec.push_back(ni);
			j_vec.push_back(nj);
			float tmp_dist = sqrtf(pow((float)ni * A_mat[0][0] + (float)nj * A_mat[0][1], 2) + pow((float)ni * A_mat[1][0] + (float)nj * A_mat[1][1], 2));
			dist_vec.push_back(tmp_dist);
			indices.push_back(curr_ind);
			curr_ind = curr_ind + 1;
		}
	}
	int i_size = i_vec.size();

	// To sort i_vec, j_vec and dist_vec, we need to use the distance_comparator fuction:
	std::vector<std::pair<int, float>> distance_pairing;
	for (int ii = 0; ii < i_size; ii++) {
		distance_pairing.emplace_back(indices[ii], dist_vec[ii]);
	}
	std::stable_sort(distance_pairing.begin(), distance_pairing.end(), distance_comparator);
	for (int ii = 0; ii < distance_pairing.size(); ii++) {
		dist_vec[ii] = distance_pairing[ii].second; 
		i_vec[ii] = i_vec[distance_pairing[ii].first];
		j_vec[ii] = j_vec[distance_pairing[ii].first];
	}

	float J_curr = J_vec[0];
	int JSize = J_vec.size();
	int J_counter = -1;
	float dist_curr = dist_vec[0];
	float dist_thold = 0.001;
	

	// Over-complicated way to create the heisenberg_info struct:
	for (int i = 0; i < i_size; i++) {
		if (dist_vec[i] < dist_thold) continue;
		if (dist_vec[i] > dist_curr + dist_thold) {
			J_counter = J_counter + 1;
			if (J_counter >= JSize) break;
			J_curr = J_vec[J_counter];

			dist_curr = dist_vec[i];
		}
		heisenberg_info.Jxy_terms.push_back({ i_vec[i],j_vec[i],0,0 });
		heisenberg_info.Jz_terms.push_back({ i_vec[i],j_vec[i],0,0 });
		heisenberg_info.Jxy_couplings.push_back(J_curr);
		heisenberg_info.Jz_couplings.push_back(J_curr);
	}
	heisenberg_info.intra_c_bool = intra_c_bool;
	heisenberg_info.inter_c_bool = inter_c_bool;
	return Heisenberg<float>(geom_info, heisenberg_info);
}