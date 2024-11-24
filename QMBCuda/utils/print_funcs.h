#pragma once
#include "thrust/device_vector.h"
#include <string>
#include <vector>
#include "thrust/complex.h"
#include "thrust/copy.h"

template <typename T> void printMatrixRowMajor(T* InMat, int N, int Nprint);

template <typename T> void printMatrixRowMajor_NonSq(T* InMat, int N1, int N2);

//template <typename T> void printDeviceMatrix(thrust::device_vector<T> DevMat, int NRows, int NCols, std::string header, std::string ender);

template <typename T> void printMatrix(std::vector<std::vector<T>> InMat);

template <typename T> void printVector(std::vector<T> InVec);

void WriteArrayToFile(std::string file_name, float* array, int array_size);

// For some reason, thrust::copy cannot deal with templates very well if used in the source file
template <typename T>
void printDeviceMatrix(thrust::device_vector<T> DevMat, int NRows, int NCols, std::string header, std::string ender) {
    printf("%s:\n", header.c_str());
    int start_ind = 0;
    for (int i = 0; i < NRows; i++) {
        thrust::copy(DevMat.begin() + start_ind, DevMat.begin() + start_ind + NCols, std::ostream_iterator<T>(std::cout, ", "));
        start_ind = start_ind + NCols;
        printf("\n");
    }
    printf("%s:\n", ender.c_str());
}