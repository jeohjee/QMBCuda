#pragma once
#include <vector>
#include <functional>
#include <armadillo>


// This is the most generic symmetry group class one can have that includes only the group elements (operators) and the character table.


template <typename T>
class SymmetryClass
{
public:

	SymmetryClass() {};
	SymmetryClass(std::vector<std::vector<T>> character_table, std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> GroupElems);

	virtual ~SymmetryClass() {};

	std::vector<std::vector<T>> GetCharTable();
	std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> GetGroupElemVec();
	void GetGroupElemArr(std::vector<float> R1_in, std::vector<float> R2_in, uint32_t* group_el_arr);


protected:
	std::vector<std::function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> Group_elems;
	std::vector<std::vector<T>> char_table;
};

