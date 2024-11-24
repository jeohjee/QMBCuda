#pragma once
#include <vector>
#include <armadillo>

//extern struct LatticeGeometryInfo;

struct LatticeGeometryInfo {
	std::vector<float> rx_alpha; // x-coordinates within the unit cell.
	std::vector<float> ry_alpha; // y-coordinates within the unit cell
	std::vector<std::vector<float>> A_mat;
	int N1; //number of unit cells within a1 direction
	int N2;
};

std::vector<std::vector<float>> ReciprocalBasisVectors(LatticeGeometryInfo);

arma::Mat<std::complex<float>> SimpleFourierTransform(arma::Mat<std::complex<float>> A_real_space, LatticeGeometryInfo geom_info);

LatticeGeometryInfo create_square_lattice_info(int N1, int N2);

LatticeGeometryInfo create_triangular_lattice_info(int N1, int N2);

