//#include "../dependencies.h"
#include "base.h"
#define _USE_MATH_DEFINES
#include <math.h> 


std::vector<std::vector<float>> ReciprocalBasisVectors(LatticeGeometryInfo geom_info) {
	// Return the reciprocal basis vector for 2D lattice defined by geom_info
	std::vector<std::vector<float>> A_mat = geom_info.A_mat;
	std::vector<std::vector<float>> Ak_mat;
	float detA = A_mat[0][0] * A_mat[1][1] - A_mat[0][1] * A_mat[1][0];
	float mul_coef = 2 * (float)(M_PI) / detA;
	Ak_mat.push_back({ mul_coef * A_mat[1][1], (float)(-1.0) * mul_coef * A_mat[1][0] });
	Ak_mat.push_back({ (float)(-1.0) * mul_coef * A_mat[0][1] ,mul_coef * A_mat[0][0] });
	return Ak_mat;
}

arma::Mat<std::complex<float>> SimpleFourierTransform(arma::Mat<std::complex<float>> A_real_space, LatticeGeometryInfo geom_info) {

	// Extremely naive Fourier transform. Enough for our purposes

	std::vector<std::vector<float>> B_mat = ReciprocalBasisVectors(geom_info);
	int N1 = geom_info.N1;
	int N2 = geom_info.N2;

	arma::Mat<std::complex<float>> A_k_space(N1, N2, arma::fill::zeros);

	std::vector<std::vector<float>>A_mat = geom_info.A_mat;
	std::vector<float> a1 = { A_mat[0][0],A_mat[1][0] };
	std::vector<float> a2 = { A_mat[0][1],A_mat[1][1] };
	std::complex<float> im = std::complex<float>(0.0, 1.0);
	float div_coef = (float)1.0 / sqrt((float)(N1 * N2));

	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			float r_tmp_x = i * a1[0] + j * a2[0];
			float r_tmp_y = i * a1[1] + j * a2[1];

			for (int ki = 0; ki < N1; ki++) {
				for (int kj = 0; kj < N2; kj++) {
					std::vector<float> k_vec = {
						ki * B_mat[0][0] / (float)N1 + kj * B_mat[0][1] / (float)N2,
						ki * B_mat[1][0] / (float)N1 + kj * B_mat[1][1] / (float)N2
					};
					std::complex<float> exp_term = std::exp(im * (k_vec[0] * r_tmp_x + k_vec[1] * r_tmp_y));
					A_k_space(ki, kj) = A_k_space(ki, kj) + div_coef * exp_term * A_real_space(i, j);
				}
			}
		}
	}
	return A_k_space;
}

LatticeGeometryInfo create_square_lattice_info(int N1, int N2) {

	// The basis vectors:
	std::vector<std::vector<float>> A_mat;
	A_mat.push_back({ 1.0,0.0 });
	A_mat.push_back({ 0.0,1.0 });

	LatticeGeometryInfo square_lat_info = {};
	square_lat_info.A_mat = A_mat;
	square_lat_info.N1 = N1;
	square_lat_info.N2 = N2;
	square_lat_info.rx_alpha = { 0.0 };
	square_lat_info.ry_alpha = { 0.0 };

	return square_lat_info;
}

LatticeGeometryInfo create_triangular_lattice_info(int N1, int N2) {

	// The basis vectors:
	float sq3 = sqrtf(3.0) / 2.0;
	std::vector<std::vector<float>> A_mat;
	A_mat.push_back({ 1.0,0.5 });
	A_mat.push_back({ 0.0, sq3 });

	LatticeGeometryInfo triangular_lat_info = {};
	triangular_lat_info.A_mat = A_mat;
	triangular_lat_info.N1 = N1;
	triangular_lat_info.N2 = N2;
	triangular_lat_info.rx_alpha = { 0.0 };
	triangular_lat_info.ry_alpha = { 0.0 };

	return triangular_lat_info;
}