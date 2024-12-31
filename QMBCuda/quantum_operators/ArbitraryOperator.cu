#include "ArbitraryOperator.h"
#include <unordered_map>
#include <iostream>
#include <complex>

using complex_th = thrust::complex<float>;

// Constructor for the ManyBodyOperator:
template <typename T>
ManyBodyOperator<T>::ManyBodyOperator(std::vector<std::vector<Operator<T>>> _elements) {
	elements = _elements;
}
template ManyBodyOperator<float>::ManyBodyOperator(std::vector<std::vector<Operator<float>>> _elements);
template ManyBodyOperator<complex_th>::ManyBodyOperator(std::vector<std::vector<Operator<complex_th>>> _elements);


template <typename T>
std::vector<std::vector<Operator<T>>> ManyBodyOperator<T>::GetElems() const {
	return elements;
}
template std::vector<std::vector<Operator<float>>> ManyBodyOperator<float>::GetElems() const;
template std::vector<std::vector<Operator<complex_th>>> ManyBodyOperator<complex_th>::GetElems() const;


template <typename T> int ManyBodyOperator<T>::GetMaxTerms() {

	int curr_max = 0;
	for (int i = 0; i < this->elements.size(); i++) {
		if (curr_max < this->elements[i].size()) {
			curr_max = this->elements[i].size();
		}
	}
	return curr_max;
}
template int ManyBodyOperator<float>::GetMaxTerms();
template int ManyBodyOperator<complex_th>::GetMaxTerms();


template <typename T>
void ManyBodyOperator<T>::AddTerm(std::vector<Operator<T>> new_term) {
	elements.emplace_back(new_term);
}
template void ManyBodyOperator<float>::AddTerm(std::vector<Operator<float>> new_term);
template void ManyBodyOperator<complex_th>::AddTerm(std::vector<Operator<complex_th>> new_term);

template <typename T> void ManyBodyOperator<T>::AddSingleOperator(Operator<T> new_term) {
	std::vector<Operator<T>> tmp_term = { new_term };
	elements.emplace_back(tmp_term);
}
template void ManyBodyOperator<float>::AddSingleOperator(Operator<float> new_term);
template void ManyBodyOperator<complex_th>::AddSingleOperator(Operator<complex_th> new_term);

template <typename T> void ManyBodyOperator<T>::LocalizeOperator(int new_site) {
	for (int ai = 0; ai < elements.size(); ai++) {
		for (int bi = 0; bi < elements[ai].size(); bi++) {
			elements[ai][bi].SetSite(new_site);
		}
	}
}
template void ManyBodyOperator<float>::LocalizeOperator(int new_site);
template void ManyBodyOperator<complex_th>::LocalizeOperator(int new_site);


template <typename T>
void ManyBodyOperator<T>::PrintOperator() {

	std::unordered_map<OperatorType, std::string> op_dict;
	op_dict[OperatorType::Sz] = "Sz";
	op_dict[OperatorType::Sp] = "Sp";
	op_dict[OperatorType::Sm] = "Sm";

	int max_terms_per_line = 5;
	int term_counter = 0;

	for (int i = 0; i < elements.size(); i++) {
		T scalar = (T)1.0;
		std::vector<std::string> tmp_op_vec = {};
		std::vector<int> lattice_site_vec = {};
		for (int oi = 0; oi < elements[i].size(); oi++) {
			scalar = scalar * elements[i][oi].GetScalar();
			tmp_op_vec.push_back(op_dict[elements[i][oi].GetType()]);
			lattice_site_vec.push_back(elements[i][oi].GetSite());
		}
		if (scalar == (T)0.0) continue;

		if (scalar != (T)1.0) {
			if constexpr (::cuda::std::is_same_v<T, float>) {
				printf("%f ", scalar);
			}
			else {
				//if (scalar.imag() !=0) printf("(%f + %f i) ", scalar.real(), scalar.imag());
				//else printf("%f", scalar.real());

				if (scalar.imag() != 0) {
					if (scalar.imag() >= 0)  printf("(%f + %f i) ", scalar.real(), scalar.imag());
					else printf("(%f %f i) ", scalar.real(), scalar.imag());
				}
				else {
					if (scalar.real() >= 0) printf("%f ", scalar.real());
					else printf("(%f) ", scalar.real());
				}

			}
		}

		for (int oi = 0; oi < elements[i].size(); oi++) {
			std::cout << tmp_op_vec[oi] << "(r=" << lattice_site_vec[oi] << ")";
		}
		if (i < elements.size() - 1) printf(" + ");

		term_counter = term_counter + 1;
		if (term_counter >= max_terms_per_line) {
			term_counter = 0;
			printf("\n");
		}
	}
	printf("\n");
}
template void ManyBodyOperator<float>::PrintOperator();
template void ManyBodyOperator<complex_th>::PrintOperator();


// Necessary + and * operators for ManyBodyOperators:
template <typename T>
ManyBodyOperator<T> ManyBodyOperator<T>::operator+(ManyBodyOperator<T> const& A) {

	std::vector<std::vector<Operator<T>>> _elements = {};
	_elements = elements;
	for (int ai = 0; ai < A.GetElems().size(); ai++) _elements.push_back(A.GetElems()[ai]);
	return ManyBodyOperator<T>(_elements);

}
template ManyBodyOperator<float> ManyBodyOperator<float>::operator+(ManyBodyOperator<float> const& A);
template ManyBodyOperator<complex_th> ManyBodyOperator<complex_th>::operator+(ManyBodyOperator<complex_th> const& A);

template <typename T>
ManyBodyOperator<T> ManyBodyOperator<T>::operator*(ManyBodyOperator<T> const& A) {

	std::vector<std::vector<Operator<T>>> _elements = {};

	for (int ai = 0; ai < elements.size(); ai++) {
		for (int bi = 0; bi < A.GetElems().size(); bi++) {
			std::vector<Operator<T>> tmp_vec = elements[ai];
			for (int ai2 = 0; ai2 < A.GetElems()[bi].size(); ai2++) tmp_vec.push_back(A.GetElems()[bi][ai2]);
			_elements.push_back(tmp_vec);
		}
	}
	return ManyBodyOperator<T>(_elements);
}
template ManyBodyOperator<float> ManyBodyOperator<float>::operator*(ManyBodyOperator<float> const& A);
template ManyBodyOperator<complex_th> ManyBodyOperator<complex_th>::operator*(ManyBodyOperator<complex_th> const& A);

template <typename T>
ManyBodyOperator<T> ManyBodyOperator<T>::operator*(float scalar) {
	std::vector<std::vector<Operator<T>>> _elements = {};
	_elements = elements;
	for (int i = 0; i < elements.size(); i++) {
		T new_scalar = elements[i][0].GetScalar();
		new_scalar = new_scalar * (T)scalar;
		_elements[i][0].SetScalar(new_scalar);
	}
	return ManyBodyOperator<T>(_elements);
}
template ManyBodyOperator<float> ManyBodyOperator<float>::operator*(float scalar);
template ManyBodyOperator<complex_th> ManyBodyOperator<complex_th>::operator*(float scalar);

template <typename T>
ManyBodyOperator<T> operator*(float scalar, ManyBodyOperator<T>& A) {
	return A * scalar;
}
template ManyBodyOperator<float> operator*<float>(float scalar, ManyBodyOperator<float>& A);
template ManyBodyOperator<complex_th> operator*<complex_th>(float scalar, ManyBodyOperator<complex_th>& A);

template <typename T>
ManyBodyOperator<complex_th> ManyBodyOperator<T>::operator*(complex_th scalar) {

	std::vector<std::vector<Operator<complex_th>>> _elements = {};
	for (int i = 0; i < elements.size(); i++) {
		std::vector<Operator<complex_th>> tmp_op_vec = {};
		for (int j = 0; j < elements[i].size(); j++) {
			int op_site = elements[i][j].GetSite();
			OperatorType op_type = elements[i][j].GetType();
			T op_scalar = elements[i][j].GetScalar();
			complex_th comp_scalar = (complex_th)op_scalar;
			if (j == 0) comp_scalar = comp_scalar * scalar;
			Operator<complex_th> tmp_op = Operator<complex_th>(op_site, op_type, comp_scalar);
			tmp_op_vec.push_back(tmp_op);
		}
		_elements.push_back(tmp_op_vec);
	}
	return ManyBodyOperator<complex_th>(_elements);
}
template ManyBodyOperator<complex_th> ManyBodyOperator<float>::operator*(complex_th scalar);
template ManyBodyOperator<complex_th> ManyBodyOperator<complex_th>::operator*(complex_th scalar);

template <typename T>
ManyBodyOperator<complex_th> operator*(complex_th scalar, ManyBodyOperator<T>& A) {
	return A * scalar;
}
template ManyBodyOperator<complex_th> operator*<float>(complex_th scalar, ManyBodyOperator<float>& A);
template ManyBodyOperator<complex_th> operator*<complex_th>(complex_th scalar, ManyBodyOperator<complex_th>& A);


// Constructor for SzSz:
template <typename T> SzSz_correlator<T>::SzSz_correlator(int i_in, int j_in) {

	std::vector<Operator<T>> Sz_ops;
	Operator<T> Sz_i(i_in, OperatorType::Sz, (T)1.0);
	Operator<T> Sz_j(j_in, OperatorType::Sz, (T)1.0);
	Sz_ops.emplace_back(Sz_i);
	Sz_ops.emplace_back(Sz_j);
	elements.emplace_back(Sz_ops);
}
template SzSz_correlator<float>::SzSz_correlator(int i_in, int j_in);
template SzSz_correlator<complex_th>::SzSz_correlator(int i_in, int j_in);

// Constructor for Sz:
template <typename T> Sz<T>::Sz(int i) {
	Operator<T> Sz_i(i, OperatorType::Sz, (T)1.0);
	std::vector<Operator<T>> Sz_ops = { Sz_i };
	elements.emplace_back(Sz_ops);
}
template Sz<float>::Sz(int i);
template Sz<complex_th>::Sz(int i);

// Constructor for S+:
template <typename T> Sp<T>::Sp(int i) {
	Operator<T> Sp_i(i, OperatorType::Sp, (T)1.0);
	std::vector<Operator<T>> Sp_ops = { Sp_i };
	elements.emplace_back(Sp_ops);
}
template Sp<float>::Sp(int i);
template Sp<complex_th>::Sp(int i);

// Constructor for S-:
template <typename T> Sm<T>::Sm(int i) {
	Operator<T> Sm_i(i, OperatorType::Sm, (T)1.0);
	std::vector<Operator<T>> Sm_ops = { Sm_i };
	elements.emplace_back(Sm_ops);
}
template Sm<float>::Sm(int i);
template Sm<complex_th>::Sm(int i);

// Fourier transform for any operator:

std::vector<ManyBodyOperatorC> FourierOperator(ManyBodyOperatorC A, std::vector<float> k, LatticeGeometryInfo geom_info) {

	std::vector<std::vector<float>>A_mat = geom_info.A_mat;
	int N1 = geom_info.N1;
	int N2 = geom_info.N2;
	int LS = geom_info.rx_alpha.size();

	std::vector<float> a1 = { A_mat[0][0],A_mat[1][0] };
	std::vector<float> a2 = { A_mat[0][1],A_mat[1][1] };

	std::vector<ManyBodyOperatorC> A_k_vec;
	for (int alpha = 0; alpha < LS; alpha++) {
		ManyBodyOperatorC A_k_tmp;
		A_k_vec.emplace_back(A_k_tmp);
	}

	std::complex<float> im = std::complex<float>(0.0, 1.0);
	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			for (int alpha = 0; alpha < LS; alpha++) {

				int unit_cell_ind = alpha + j * LS + i * N2 * LS;
				float r_tmp_x = i * a1[0] + j * a2[0];
				float r_tmp_y = i * a1[1] + j * a2[1];

				ManyBodyOperatorC A_ij_a = A;
				A_ij_a.LocalizeOperator(unit_cell_ind);

				complex_th exp_coef = complex_th(std::exp(im * (k[0] * r_tmp_x + k[1] * r_tmp_y)));
				A_k_vec[alpha] = A_k_vec[alpha] + exp_coef * A_ij_a;
			}
		}
	}
	for (int alpha = 0; alpha < LS; alpha++) {
		A_k_vec[alpha] = complex_th(1 / sqrt((float)(N1 * N2))) * A_k_vec[alpha];
	}
	return A_k_vec;
}

std::vector< std::vector<std::vector<ManyBodyOperatorC>>>
FullFourierOperator(ManyBodyOperatorC A, LatticeGeometryInfo geom_info, bool neg_sqn = false) {

	std::vector<std::vector<float>> B_mat = ReciprocalBasisVectors(geom_info);
	int N1 = geom_info.N1;
	int N2 = geom_info.N2;
	std::vector< std::vector<std::vector<ManyBodyOperatorC>>> Ak_mat(N1);

	for (int i = 0; i < N1; i++) {
		for (int j = 0; j < N2; j++) {
			std::vector<float> k_vec = {
				i * B_mat[0][0] / (float)N1 + j * B_mat[0][1] / (float)N2,
				i * B_mat[1][0] / (float)N1 + j * B_mat[1][1] / (float)N2
			};
			if (neg_sqn) {
				k_vec[0] = -1 * k_vec[0];
				k_vec[1] = -1 * k_vec[1];
			}
			std::vector<ManyBodyOperatorC> Ak_tmp = FourierOperator(A, k_vec, geom_info);
			Ak_mat[i].push_back(Ak_tmp);
		}
	}
	return Ak_mat;
}
