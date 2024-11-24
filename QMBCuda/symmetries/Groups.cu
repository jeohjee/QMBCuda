#include "Groups.h"
#include "SymmOperators.h"
#include <functional>
#include <complex>

#define _USE_MATH_DEFINES
#include <math.h>

using complex_std = std::complex<float>;

Cn_group::Cn_group(int n_in, std::vector<float> R0_in)
{
	this->n = n_in;
	this->Group_elems = {};
	this->R0 = R0_in;

	// Abelian operators are created by multiplying the most recent operator with Cn
	std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> C_n = std::bind(Cn, std::placeholders::_1, std::placeholders::_2, n, this->R0);
	std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> Cn_curr = Iden;
	for (int i = 0; i < n; i++) {
		this->Group_elems.push_back(Cn_curr);
		Cn_curr = std::bind(GetElemProd, std::placeholders::_1, std::placeholders::_2, C_n, Cn_curr);
	}

	float float_n = (float)n;
	complex_std im = complex_std(0.0, 1.0);
	float phi = (float)M_PI * 2 / float_n;

	this->char_table = {};
	complex_std eps = std::exp(im * phi);
	complex_std curr_eps = complex_std(1.0, 0.0);
	for (int i = 0; i < n; i++) {
		std::vector<thrust::complex<float>> char_row = {};
		char_row.push_back(thrust::complex<float>(1.0, 0.0));
		for (int j = 1; j < n; j++) {
			float j_f = (float)j;
			char_row.push_back(thrust::complex<float>(std::pow(curr_eps, j_f)));
		}
		curr_eps = curr_eps * eps;
		this->char_table.emplace_back(char_row);
	}

}

int Cn_group::get_n() { 
	return n; 
}

Tnm_group::Tnm_group(LatticeGeometryInfo _geom_info)
{

	this->Group_elems = {};
	this->char_table = {};
	this->geom_info = _geom_info;
	N = this->geom_info.N1;
	M = this->geom_info.N2;

	std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> Tx = std::bind(
		Tnm_general, std::placeholders::_1, std::placeholders::_2, 1, 0, this->geom_info
	);

	std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> Ty = std::bind(
		Tnm_general, std::placeholders::_1, std::placeholders::_2, 0, 1, this->geom_info
	);

	std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> Tnm_curr = Iden;
	for (int nx = 0; nx < N; nx++) {
		for (int ny = 0; ny < M; ny++) {
			this->Group_elems.push_back(Tnm_curr);
			Tnm_curr = std::bind(GetElemProd, std::placeholders::_1, std::placeholders::_2, Tnm_curr, Ty);
		}
		Tnm_curr = std::bind(GetElemProd, std::placeholders::_1, std::placeholders::_2, Tnm_curr, Tx);
	}

	float float_N = (float)N;
	float float_M = (float)M;
	complex_std im = complex_std(0.0, 1.0);
	float phi_N = (float)M_PI * 2 / float_N;
	float phi_M = (float)M_PI * 2 / float_M;
	complex_std eps1 = std::exp(im * phi_N);
	complex_std eps2 = std::exp(im * phi_M);

	complex_std curr_eps = complex_std(1.0, 0.0);

	complex_std start_coef_x;
	complex_std start_coef_y;


	for (int nx = 0; nx < N; nx++) {
		for (int ny = 0; ny < M; ny++) {
			start_coef_x = std::pow(eps1, nx);
			start_coef_y = std::pow(eps2, ny);

			std::vector<thrust::complex<float>> char_row = {};

			complex_std tmp_ch_x;
			complex_std tmp_ch_y;
			complex_std final_coef;
			for (int jx = 0; jx < N; jx++) {
				float jx_f = (float)jx;
				tmp_ch_x = std::pow(start_coef_x, jx_f);
				for (int jy = 0; jy < M; jy++) {
					float jy_f = (float)jy;
					tmp_ch_y = std::pow(start_coef_y, jy_f);
					final_coef = tmp_ch_x * tmp_ch_y;
					char_row.push_back(thrust::complex<float>(final_coef));
				}

			}
			this->char_table.emplace_back(char_row);
		}
	}
}

int Tnm_group::get_n() { 
	return N; 
}

int Tnm_group::get_m() { 
	return M; 
};