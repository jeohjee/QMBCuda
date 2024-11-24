#include "SymmetryClass.h"
#include "thrust/complex.h"
#include "../utils/misc_funcs.h"

using complex_th = thrust::complex<float>;

template <typename T> SymmetryClass<T>::SymmetryClass(
	std::vector<std::vector<T>> character_table,
	std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> GroupElems)
{
	Group_elems = GroupElems;
	char_table = character_table;

}

template
SymmetryClass<float>::SymmetryClass(
	std::vector<std::vector<float>> character_table,
	std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> GroupElems);

template
SymmetryClass<complex_th>::SymmetryClass(
	std::vector<std::vector<complex_th>> character_table,
	std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> GroupElems);

template <typename T> void SymmetryClass<T>::GetGroupElemArr(std::vector<float> R1_in, std::vector<float> R2_in, uint32_t* group_el_arr) {

	arma::Row<float> R1(R1_in);
	arma::Row<float> R2(R2_in);
	int LS = R1_in.size();

	int GSize = Group_elems.size();

	std::vector<arma::Mat<float>> transformed_coords;
	std::vector<arma::Row<arma::uword>> tmp_elem_arrs;

	for (int gi = 0; gi < GSize; gi++) {
		transformed_coords.push_back(Group_elems[gi](R1, R2));
		arma::Row<arma::uword> tmp_row(LS, arma::fill::zeros);
		tmp_elem_arrs.push_back(tmp_row);
	}
	// Now we need to create finally the matrices that correspond to the group elements in the given (R1,R2) basis.
	float coord_tol = 0.001;

	for (int lsi = 0; lsi < LS; lsi++) {
		for (int gi = 0; gi < GSize; gi++) {
			arma::Row<float> tmp_vec_d1 = arma::abs(R1 - transformed_coords[gi](0, lsi));
			arma::Row<float> tmp_vec_d2 = arma::abs(R2 - transformed_coords[gi](1, lsi));

			arma::uvec q1 = find(tmp_vec_d1 < coord_tol);
			arma::uvec q2 = find(tmp_vec_d2 < coord_tol);
			arma::uvec q3 = intersect(q1, q2);

			tmp_elem_arrs[gi](lsi) = q3[0] + 1;
		}
	}

	arma::Row<arma::uword> row_ind_vec = LS - arma::linspace<arma::Row<arma::uword>>(1, LS, LS);
	arma::Cube<float> group(LS, LS, GSize, arma::fill::zeros);

	for (int gi = 0; gi < GSize; gi++) {

		for (int ii = 0; ii < LS; ii++) {
			group(row_ind_vec(ii), LS - tmp_elem_arrs[gi](ii), gi) = 1;
		}
	}
	TranformGroupEls_ArmaToDev(&group, GSize, LS, group_el_arr);
	// The row index of group_el goes through the symmetry group elements. The column index indicates the original lattice site.
	// The value of group_el indicates the final lattice site destination

}

template void SymmetryClass<float>::GetGroupElemArr(std::vector<float> R1_in, std::vector<float> R2_in, uint32_t* group_el_arr);
template void SymmetryClass<complex_th>::GetGroupElemArr(std::vector<float> R1_in, std::vector<float> R2_in, uint32_t* group_el_arr);

template <typename T>
std::vector<std::vector<T>> SymmetryClass<T>::GetCharTable() { 
	return char_table; 
};

template std::vector<std::vector<float>> SymmetryClass<float>::GetCharTable();
template std::vector<std::vector<complex_th>> SymmetryClass<complex_th>::GetCharTable();

template <typename T>
std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> SymmetryClass<T>::GetGroupElemVec() {
	return Group_elems; 
};

template std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> SymmetryClass<float>::GetGroupElemVec();
template std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> SymmetryClass<complex_th>::GetGroupElemVec();