#include "print_funcs.h"
#include "thrust/device_vector.h"
#include <fstream>
#include <iostream>
#include <cstdio>





using complex_std = std::complex<float>;
using complex_th = thrust::complex<float>;

// should work either for T=float or T=double
template <typename T> void printMatrixRowMajor(T* InMat, int N, int Nprint)
{
    for (int i = 0; i < Nprint; i++) {
        for (int j = 0; j < Nprint; j++) {
            printf("%f ", InMat[i * N + j]);
        }
        printf("\n");
    }
}
template void printMatrixRowMajor<float>(float* InMat, int N, int Nprint);
template void printMatrixRowMajor<double>(double* InMat, int N, int Nprint);


template <typename T> void printMatrixRowMajor_NonSq(T* InMat, int N1, int N2)
{
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            printf("%f ", (float)InMat[i * N2 + j]);
        }
        printf("\n");
    }
}
template void printMatrixRowMajor_NonSq<float>(float* InMat, int N1, int N2);
template void printMatrixRowMajor_NonSq<double>(double* InMat, int N1, int N2);
template void printMatrixRowMajor_NonSq<uint32_t>(uint32_t* InMat, int N1, int N2);

template <typename T> void printMatrix(std::vector<std::vector<T>> InMat) {

    int size1 = InMat.size();
    int size2 = InMat[0].size();

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            std::cout << InMat[i][j] << " ";
        }
        std::cout << "\n";
    }
}

template void printMatrix<float>(std::vector<std::vector<float>> InMat);
template void printMatrix<complex_th>(std::vector<std::vector<complex_th>> InMat);
template void printMatrix<complex_std>(std::vector<std::vector<complex_std>> InMat);


template <typename T> void printVector(std::vector<T> InVec) {
    int size1 = InVec.size();
    std::cout << std::endl;
    for (int i = 0; i < size1; i++) {
        std::cout << InVec[i] << " ";
    }
    std::cout << "\n";
}
template void printVector<float>(std::vector<float> InVec);
template void printVector<complex_th>(std::vector<complex_th> InVec);
template void printVector<complex_std>(std::vector<complex_std> InVec);



void WriteArrayToFile(std::string file_name, float* array, int array_size) {

    std::ofstream file(file_name, std::ios::binary);
    if (file.is_open()) {
        // Write the array to the file
        file.write(reinterpret_cast<char*>(&array[0]), array_size * sizeof(float));

        // Close the file
        file.close();

        // Print a message
        std::cout << "Array written to file successfully.\n";
    }
    else {
        // Print an error message
        std::cerr << "Error opening file.\n";
    }
}
