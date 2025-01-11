#include "SingleParticleOperators.h"
#include "thrust/complex.h"
#include <map>

using complex_th = thrust::complex<float>;

OperatorType GetConjugateOperatorType(OperatorType orig_op) {

	// First we define the mapping between the original and conjugate operators:
	std::map<OperatorType, OperatorType> conjugation_map;
	conjugation_map[OperatorType::I] = OperatorType::I;

	// Spin operators (i.e. hard-core bosons):
	conjugation_map[OperatorType::Sz] = OperatorType::Sz;
	conjugation_map[OperatorType::Sp] = OperatorType::Sm;
	conjugation_map[OperatorType::Sm] = OperatorType::Sp;

	// The usual bosons:
	conjugation_map[OperatorType::b] = OperatorType::b_dag;
	conjugation_map[OperatorType::b_dag] = OperatorType::b;

	// Femionic operators:
	conjugation_map[OperatorType::c_up] = OperatorType::c_dag_up;
	conjugation_map[OperatorType::c_dag_up] = OperatorType::c_up;
	conjugation_map[OperatorType::c_down] = OperatorType::c_dag_down;
	conjugation_map[OperatorType::c_dag_down] = OperatorType::c_down;

	return conjugation_map[orig_op];
}

std::string PrintOperatorType(OperatorType op) {
	// First we define the mapping between the operators and their text form:
	std::map<OperatorType, std::string> print_map;

	print_map[OperatorType::I] = "I";

	// Spin operators (i.e. hard-core bosons):
	print_map[OperatorType::Sz] = "Sz";
	print_map[OperatorType::Sp] = "Sp";
	print_map[OperatorType::Sm] = "Sm";

	// The usual bosons:
	print_map[OperatorType::b] = "b";
	print_map[OperatorType::b_dag] = "b_dag";

	// Femionic operators:
	print_map[OperatorType::c_up] = "c_up";
	print_map[OperatorType::c_dag_up] = "c_dag_up";
	print_map[OperatorType::c_down] = "c_dw";
	print_map[OperatorType::c_dag_down] = "c_dag_dw";

	return print_map[op];

}

template <typename T> Operator<T>::Operator(int site_in, OperatorType type_in, T scalar_in) {
	site = site_in;
	scalar = scalar_in;
	type = type_in;
}

template Operator<float>::Operator(int site_in, OperatorType type_in, float scalar_in = (float)1.0);
template Operator<complex_th>::Operator(int site_in, OperatorType type_in, complex_th scalar_in = (complex_th)1.0);

template <typename T>
OperatorType Operator<T>::GetType() { return type; }
template OperatorType Operator<float>::GetType();
template OperatorType Operator<complex_th>::GetType();

template <typename T>
int Operator<T>::GetSite() { return site; }
template int Operator<float>::GetSite();
template int Operator<complex_th>::GetSite();

template <typename T>
T Operator<T>::GetScalar() { return scalar; }
template float Operator<float>::GetScalar();
template complex_th Operator<complex_th>::GetScalar();


template <typename T>
void Operator<T>::SetScalar(T _scalar) { scalar = _scalar; };
template void Operator<float>::SetScalar(float _scalar);
template void Operator<complex_th>::SetScalar(complex_th _scalar);


template <typename T>
void Operator<T>::SetSite(int new_site) { site = new_site; };
template void Operator<float>::SetSite(int new_site);
template void Operator<complex_th>::SetSite(int new_site);