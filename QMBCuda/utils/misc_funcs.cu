#include "misc_funcs.h"
#include "thrust/complex.h"

using complex_std = std::complex<float>;
using complex_th = thrust::complex<float>;

int pow2ceil(uint32_t n) {

    uint32_t n_tmp = n;
    unsigned r = 1;

    while (n_tmp >>= 1) {
        r++;
    }

    r = r - 1;

    int pow2 = 1 << r;
    if (n > pow2) pow2 = (pow2 << 1);
    return pow2;
}

double NChoosek2(double num, double den) {

    if (den <= 0) return 1;
    if (num <= 0) return 1;

    if (den <= 1) {
        return num;
    }
    return (num / den) * NChoosek2(num - 1, den - 1);
}

// Transform a Armadillo mat to array mat (row major ordering):
template <typename T> void TransformArmaMatArr(T* char_mat_arr, arma::Mat<T>* arma_mat, int N1, int N2) {

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            char_mat_arr[i * N2 + j] = (*arma_mat)(i, j);
        }
    }

}
template void TransformArmaMatArr<float>(float* char_mat_arr, arma::Mat<float>* arma_mat, int N1, int N2);
template void TransformArmaMatArr<complex_std>(complex_std* char_mat_arr, arma::Mat<complex_std>* arma_mat, int N1, int N2);


template <typename T> void TransformMatArr(T* char_mat_arr, std::vector<std::vector<T>>* orig_mat, int N1, int N2) {

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            char_mat_arr[i * N2 + j] = (*orig_mat)[i][j];
        }
    }

}
template void TransformMatArr<float>(float* char_mat_arr, std::vector<std::vector<float>>* orig_mat, int N1, int N2);
template void TransformMatArr<complex_std>(complex_std* char_mat_arr, std::vector<std::vector<complex_std>>* orig_mat, int N1, int N2);
template void TransformMatArr<complex_th>(complex_th* char_mat_arr, std::vector<std::vector<complex_th>>* orig_mat, int N1, int N2);


template <typename T> void StdMatToArray(std::vector<std::vector<T>> InMat, T* OutMat) {
    // Transfers a nested std::vector to a usual (row major) array:
    int size1 = InMat.size();
    int size2 = InMat[0].size();

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            OutMat[i * size2 + j] = InMat[i][j];
        }
    }
}
template void StdMatToArray<float>(std::vector<std::vector<float>> InMat, float* OutMat);
template void StdMatToArray<complex_std>(std::vector<std::vector<complex_std>> InMat, complex_std* OutMat);
template void StdMatToArray<complex_th>(std::vector<std::vector<complex_th>> InMat, complex_th* OutMat);


// The row index of group_el goes through the symmetry group elements. The column index indicates the original lattice site.
// The value of group_el indicates the final lattice site destination
template <typename T> void TranformGroupEls_ArmaToDev(arma::Cube<T>* group, int GSize, int LS, uint32_t* group_el_arr)
{
    for (int gi = 0; gi < GSize; gi++) {
        for (uint32_t i = 0; i < LS; i++) {
            for (uint32_t j = 0; j < LS; j++) {
                if ((*group)(i, j, gi) == 0) continue;
                group_el_arr[gi * LS + (LS - 1 - j)] = LS - 1 - i; // we flip the order of site indexing
            }
        }
    }
}

template void TranformGroupEls_ArmaToDev<float>(arma::Cube<float>* group, int GSize, int LS, uint32_t* group_el_arr);