#include "SingleParticleOperators.h"
#include "thrust/complex.h"

using complex_th = thrust::complex<float>;

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