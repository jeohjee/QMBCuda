#pragma once
#include <cstdint>
#include <vector>
#include <armadillo>


int pow2ceil(uint32_t);

double NChoosek2(double num, double den);

template <typename T> void TransformArmaMatArr(T* char_mat_arr, arma::Mat<T>* arma_mat, int N1, int N2);

template <typename T> void TransformMatArr(T* char_mat_arr, std::vector<std::vector<T>>* orig_mat, int N1, int N2);

template <typename T> void StdMatToArray(std::vector<std::vector<T>> InMat, T* OutMat);

template <typename T> void TranformGroupEls_ArmaToDev(arma::Cube<T>* group, int GSize, int LS, uint32_t* group_el_arr);