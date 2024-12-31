#include "examples.h"
#include "../exact_diagonalization/HeisenbergHam_CUDA.h"
#include "../exact_diagonalization/HeisenbergHamAbelianSymms_CUDA.h"
#include "cuComplex.h"
#include <vector>
#include <functional>
#include "thrust/complex.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"

#include "../cuda_wrappers/CudaWrappers.h"
#include "../symmetries/SymmetryClass.h"
#include "../symmetries/SymmOperators.h"
#include "../symmetries/Groups.h"
#include "../lattice_models/T_standard.h"
#include "../utils/print_funcs.h"
#include "../utils/misc_funcs.h"
#include "../lattice_models/Heisenberg.h"

#define _USE_MATH_DEFINES
#include <math.h> 

using complex_th = thrust::complex<float>;


int complex_example() {

    /*
    * Simple example with complex_th
    */

    float test = 2.0;
    thrust::device_vector<complex_th> complex_vec(3, complex_th(1.0, 2.0));
    float norm = 0.0;
    norm = VecNorm_cuBLAS_complex_float_wrapper(reinterpret_cast<cuFloatComplex*>(thrust::raw_pointer_cast(complex_vec.data())), 3);
    std::cout << "Norm is:" << norm << "\n";

    complex_th test_c(2.3, 0.0);
    float test_f = test_c.real();
    std::cout << "test_f is: " << test_f;
    return 0;
}

int operator_example() {

    /* 
    Few simple examples of operator types.
    */

    Operator<float> Heis_test(0, OperatorType::Sz,1.0);
    OperatorType Heis_type = Heis_test.GetType();
    if (Heis_type == OperatorType::Sz) {
        printf("Operator is Sz type\n");
    }

    float scalar = Heis_test.GetScalar();
    printf("scalar is : % f", scalar);

    SzSz_correlator<float> SzSz(0, 1);

    Sp<float> Sp_op(0);
    std::vector<std::vector<Operator<float>>> test_op_vec = Sp_op.GetElems();
    Operator<float> tmp_op = test_op_vec[0][0];
    OperatorType Heis_type2 = tmp_op.GetType();

    if (Heis_type2 == OperatorType::Sp) {
        printf("Operator is Sp type\n");
    }
    return 0;
}


int square_lattice_symmetries_example() {

    /* Square lattice Heisenberg exact diagonalization with CUDA by
     taking advantage of Abelian symmetries*/

    float J1 = 1.0;
    float J2 = 0.15;

     // Size of the lattice
    int N1 = 6;
    int N2 = 4;

    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0,1.0 });

    LatticeGeometryInfo geom_info;
    geom_info.A_mat = A_mat;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };

    int LS = N1*N2;

    
    // Create the lattice model:
    std::vector<float> J_vec = { J1,J2 };
    std::vector<float> Range_vec = { 1.0001,1.4143 };
    int intra_c_bool = 1;
    int inter_c_bool = 0; // should determine whether we have periodic or non-periodic BCs

    T_standard<float> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };

    // Create the C2 symmetry class:
    std::vector<float> R0 = TLat.GetCentralPoint(); // This extract the central point of the lattice which is needed for C2 group

    std::vector<std::vector<float>> char_table = {};
    char_table.push_back({ 1.0,1.0 });
    char_table.push_back({ 1.0,-1.0 });
    auto C2_func = bind(Cn, placeholders::_1, placeholders::_2, 2, R0); // Create the C2 operator

    std::vector<function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)>> Group_elems = { Iden,C2_func }; // C2 symmetry group elements
    SymmetryClass<float> C2_class_abelian(char_table, Group_elems); // C2 is an abelian group

    // Finally, create the Hamiltonian:
    auto Heisenberg_time = std::chrono::high_resolution_clock::now();

    HeisenbergHamAbelianSymms_CUDA<float> Ham(TLat, B_field_vec, J_dim_weights, LS / 2, &C2_class_abelian, 9);

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate in different irreps sectors: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    // Print the eigenvalues:
    std::vector<std::vector<float>> E_vecs = Ham.get_E_vecs();
    printMatrix<float>(E_vecs);
    float* E_vecs_array = (float*)malloc(sizeof(float) * E_vecs.size() * E_vecs[0].size());
    StdMatToArray<float>(E_vecs, E_vecs_array);


    Operator<complex_th> Sz0(0, OperatorType::Sz,1.0);
    Operator<complex_th> Sz1(3, OperatorType::Sz,1.0);
    std::vector< Operator<complex_th>> aux_vec = { Sz0, Sz1 };

    ManyBodyOperator<complex_th> Sz_mb_wrapper;
    //Sz_mb_wrapper.AddSingleOperator(Sz0);
    Sz_mb_wrapper.AddTerm(aux_vec);

    complex_th Sz0_exp;
    Sz0_exp = Ham.ComputeStaticExpValZeroT(Sz_mb_wrapper);

    std::cout << "\n Result: " << Sz0_exp.real() << ", 1i " << Sz0_exp.imag() << "\n";

    return 0;
}


int square_brute_force_example() {

    /* Square lattice Heisenberg exact diagonalization with CUDA without
    taking advantage of symmetries*/

    // Size of the lattice
    int N1 = 6;
    int N2 = 4;

    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0,1.0 });

    LatticeGeometryInfo geom_info;
    geom_info.A_mat = A_mat;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };

    int LS = N1 * N2;

    std::vector<float> J_vec = { 1.0,0.15 };
    std::vector<float> Range_vec = { 1.0001,1.4143 };
    int intra_c_bool = 1;
    int inter_c_bool = 0;

    T_standard<float> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };

    auto Heisenberg_time = std::chrono::high_resolution_clock::now();
    HeisenbergHam_CUDA(TLat, B_field_vec, J_dim_weights, LS / 2, 9); // This diagonalizes the Heisenberg Hamiltonian defined by the input arguments

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    return 0;
}

// Helper function for examples below:
void create_triangular(arma::Row<float>* R1, arma::Row<float>* R2, int N1, int N2) {

    float sq3 = sqrtf(3.0)/ 2.0;
    std::vector<float> a1 = { 1.0,0.0 };
    std::vector<float> a2 = { 0.5, sq3};

    int curr_ind = 0;
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            (*R1)(curr_ind) =i * a1[0] + j * a2[0];
            (*R2)(curr_ind) = i * a1[1] + j * a2[1];
            curr_ind++;
        }
    }
    printf("Triangular sites:\n");
    (*R1).print();
    (*R2).print();

}

// Helper function for examples below:
void create_square(arma::Row<float>* R1, arma::Row<float>* R2, int N1, int N2) {

    std::vector<float> a1 = { 1.0,0.0 };
    std::vector<float> a2 = { 0.0, 1.0 };

    int curr_ind = 0;
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            (*R1)(curr_ind) = i * a1[0] + j * a2[0];
            (*R2)(curr_ind) = i * a1[1] + j * a2[1];
            curr_ind++;
        }
    }
    printf("Square lattice sites:\n");
    (*R1).print();
    (*R2).print();

}

int Cn_group_example() {

    int n = 4;

    Cn_group Cx_group = Cn_group(n, {0.0,0.0});
    std::vector<std::vector<complex_th>> char_mat = Cx_group.GetCharTable();
    printMatrix(char_mat);

    int N1 = 3;
    int N2 = 3;
    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_triangular(&R1, &R2, N1, N2);

    for (int i = 0; i < n; i++) {
        arma::Mat<float> tmp_mat = Cx_group.GetGroupElemVec()[i](R1, R2);
        printf("after %d:th operator:\n",i);
        tmp_mat.print();
    }

    return 0;
}

int Tnm_example() {

    int N1 = 4;
    int N2 = 6;
    
    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_square(&R1, &R2, N1, N2);

    int nx = 1;
    int ny = 1;

    arma::Mat<float> tmp_mat = Tnm(R1, R2, N1, N2, nx, ny);
    tmp_mat.print();

    return 0;
}

int Tnm_group_example() {

    int N = 2;
    int M = 2;

    LatticeGeometryInfo geom_info = create_square_lattice_info(N,M);

    Tnm_group T_group = Tnm_group(geom_info);
    std::vector<std::vector<complex_th>> char_mat = T_group.GetCharTable();
    printMatrix(char_mat);


    int N_tot = N * M;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_square(&R1, &R2, N, M);

    for (int i = 0; i < N*M; i++) {
        arma::Mat<float> tmp_mat = T_group.GetGroupElemVec()[i](R1, R2);
        printf("after %d:th operator:\n", i);
        tmp_mat.print();
    }

    return 0;
}


int square_lattice_symmetries_example2() {

    /* Square lattice Heisenberg exact diagonalization with CUDA by
     taking advantage of Abelian symmetries by using pre-defined symmetry groups*/

    complex_th J1 = complex_th(1.0,0.0);
    complex_th J2 = complex_th(0.15, 0.0);

    // Size of the lattice
    int N1 = 4;
    int N2 = 4;

    LatticeGeometryInfo geom_info = create_square_lattice_info(N1,N2);

    int LS = N1 * N2;

    Tnm_group symm_group(geom_info);


    std::vector<complex_th> J_vec = { J1,J2 };
    std::vector<float> Range_vec = { 1.0001,1.4143 };
    int intra_c_bool = 1;
    int inter_c_bool = 1; // should determine whether we have periodic or non-periodic BCs

    T_standard<complex_th> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };


    // Next we create the Hamiltonian
    auto Heisenberg_time = std::chrono::high_resolution_clock::now();

    int max_term = 2 * TLat.getTSize() / (N1*N2);
    max_term = max_term;

    HeisenbergHamAbelianSymms_CUDA<complex_th> Ham(TLat, B_field_vec, J_dim_weights, LS / 2, &symm_group, max_term);

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate in different irreps sectors: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    // Print the eigenvalues:
    std::vector<std::vector<float>> E_vecs = Ham.get_E_vecs();
    printMatrix<float>(E_vecs);
    float* E_vecs_array = (float*)malloc(sizeof(float) * E_vecs.size() * E_vecs[0].size());
    StdMatToArray<float>(E_vecs, E_vecs_array);


    Operator<complex_th> Sz0(0, OperatorType::Sz,1.0);
    Operator<complex_th> Sz1(3, OperatorType::Sz,1.0);
    std::vector< Operator<complex_th>> aux_vec = { Sz0, Sz1 };

    ManyBodyOperator<complex_th> Sz_mb_wrapper;
    Sz_mb_wrapper.AddTerm(aux_vec);

    complex_th Sz0_exp;

    Sz0_exp = Ham.ComputeStaticExpValZeroT(Sz_mb_wrapper);

    std::cout << "\n Result: " << Sz0_exp.real() << ", 1i " << Sz0_exp.imag() << "\n";

    return 0;
}



int square_lattice_symmetries_example3() {

    /* Square lattice Heisenberg exact diagonalization with CUDA by
     taking advantage of Abelian symmetries by using pre-defined symmetry groups*/

    complex_th J1 = complex_th(1.0, 0.0);
    complex_th J2 = complex_th(0.15, 0.0);


     // Size of the lattice
    int N1 = 4;
    int N2 = 4;

    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0,1.0 });

    LatticeGeometryInfo geom_info;
    geom_info.A_mat = A_mat;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };

    int LS = N1 * N2;

    // Create the lattice geometry:
    std::vector<complex_th> J_vec = { J1,J2 };
    std::vector<float> Range_vec = { 1.0001,1.4143 };
    int intra_c_bool = 1;
    int inter_c_bool = 0; // should determine whether we have periodic or non-periodic BCs

    T_standard<complex_th> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };

    std::vector<float> R0 = TLat.GetCentralPoint();
    
    // Create the symmetry group C4:
    Cn_group symm_group(4, R0);

    // Next we create the Hamiltonian
    auto Heisenberg_time = std::chrono::high_resolution_clock::now();

    HeisenbergHamAbelianSymms_CUDA<complex_th> Ham(TLat, B_field_vec, J_dim_weights, LS / 2, &symm_group, 9);

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate in different irreps sectors: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    // Print the eigenvalues:
    std::vector<std::vector<float>> E_vecs = Ham.get_E_vecs();
    printMatrix<float>(E_vecs);
    float* E_vecs_array = (float*)malloc(sizeof(float) * E_vecs.size() * E_vecs[0].size());
    StdMatToArray<float>(E_vecs, E_vecs_array);


    Operator<complex_th> Sz0(0, OperatorType::Sz,1.0);
    Operator<complex_th> Sz1(3, OperatorType::Sz,1.0);
    std::vector< Operator<complex_th>> aux_vec = { Sz0, Sz1 };

    ManyBodyOperator<complex_th> Sz_mb_wrapper;
    //Sz_mb_wrapper.AddSingleOperator(Sz0);
    Sz_mb_wrapper.AddTerm(aux_vec);

    complex_th Sz0_exp;

    Sz0_exp = Ham.ComputeStaticExpValZeroT(Sz_mb_wrapper);

    std::cout << "\n Result: " << Sz0_exp.real() << ", 1i " << Sz0_exp.imag() << "\n";

    return 0;
}


int operator_arithmetic_example() {

    Operator<float> Heis_test(0, OperatorType::Sz, 1.0);

    OperatorType Heis_type = Heis_test.GetType();
    if (Heis_type == OperatorType::Sz) {
        printf("Operator is Sz type\n");
    }

    float scalar = Heis_test.GetScalar();
    printf("scalar is : % f", scalar);

    SzSz_correlator<float> SzSz(0, 1);

    Sp<float> Sp_op(0);
    Sm<float> Sm_op(1);
    SzSz_correlator<float> SzSz2(2, 4);

    std::vector<std::vector<Operator<float>>> test_op_vec = Sp_op.GetElems();
    Operator<float> tmp_op = test_op_vec[0][0];
    OperatorType Heis_type2 = tmp_op.GetType();

    if (Heis_type2 == OperatorType::Sp) {
        printf("Operator is Sp type\n");
    }

    printf("testing the plus operator:\n");

    printf("First case:\n");
    ManyBodyOperator<float> sum_op1 = Sp_op + Sm_op;
    sum_op1.PrintOperator();

    printf("\n2nd case:\n");
    ManyBodyOperator<float> prod_op1 = Sp_op * Sm_op;
    prod_op1.PrintOperator();

    printf("\n3rd case:\n");
    ManyBodyOperator<float> prod_op2 = prod_op1 * Sm_op * SzSz2 + SzSz * Sp_op;
    prod_op2.PrintOperator();

    printf("\n4th case:\n");

    SzSz_correlator<complex_th> SzSz3(1, 0);
    SzSz_correlator<complex_th> SzSz4(2, 0);

    ManyBodyOperator<complex_th> prod_op3 = SzSz3 + SzSz4;
    prod_op3.PrintOperator();

    printf("\n Test scaling by scalar:");
    printf("\n5th case:\n");
    ManyBodyOperator<complex_th> prod_op4 = prod_op3 * 2.0;
    prod_op4.PrintOperator();

    printf("\n6th case:\n");
    ManyBodyOperator<complex_th> prod_op5 = prod_op2 * complex_th(2.0,1.0);
    prod_op5.PrintOperator();

    printf("\n7th case (scalar first):\n");
    ManyBodyOperator<complex_th> prod_op6 =  2.0 * prod_op3;
    prod_op6.PrintOperator();

    printf("\n8th case (scalar first):\n");
    ManyBodyOperator<complex_th> prod_op7 = complex_th(2.0, 1.0) * prod_op2 ;
    prod_op7.PrintOperator();

    printf("\n////////\n");
    printf("\n Test Fourier transform:\n");

    int N1 = 2;
    int N2 = 2;
    LatticeGeometryInfo geom_info = create_square_lattice_info(N1, N2);

    Sz<complex_th> Sz_op_real_space(0);
    
    std::vector<float> k_vec = { (float)M_PI,0.0 };
    std::vector<ManyBodyOperator<complex_th>> S_k_vec = FourierOperator(
        Sz_op_real_space,
        k_vec,
        geom_info
        );
    S_k_vec[0].PrintOperator();

    printf("\n////////\n");
    printf("\n Test full Fourier transform:\n");

    std::vector<std::vector<std::vector<ManyBodyOperator<complex_th>>>> S_k_mat = FullFourierOperator(
        Sz_op_real_space,
        geom_info,
        false
    );

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            S_k_mat[i][j][0].PrintOperator();
        }
    }

    return 0;
}

int Tnm_general_example() {

    int N1 = 4;
    int N2 = 6;

    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_square(&R1, &R2, N1, N2);

    int nx = 1;
    int ny = 1;

    printf("\n With old method:\n");
    arma::Mat<float> tmp_mat = Tnm(R1, R2, N1, N2, nx, ny);
    tmp_mat.print();

    LatticeGeometryInfo geom_info;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };
    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0,1.0 });
    geom_info.A_mat = A_mat;

    printf("\n With new method:\n");
    arma::Mat<float> tmp_mat2 = Tnm_general(R1, R2, nx, ny, geom_info);
    tmp_mat2.print();

    return 0;
}


int Tnm_general_triangular_example() {

    int N1 = 4;
    int N2 = 6;

    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_triangular(&R1, &R2, N1, N2);
    

    arma::Mat<float> R_orig = arma::join_vert(R1,R2);
    R_orig.save("R_orig.txt", arma::raw_ascii);

    int nx = 1;
    int ny = 1;

    LatticeGeometryInfo geom_info;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };

    float sq3 = sqrtf(3.0) / 2.0;

    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.5 });
    A_mat.push_back({ 0.0, sq3 });
    geom_info.A_mat = A_mat;

    printf("\n With new method:\n");
    arma::Mat<float> tmp_mat2 = Tnm_general(R1, R2, nx, ny, geom_info);
    tmp_mat2.print();
    tmp_mat2.save("R_new.txt", arma::raw_ascii);

    return 0;
}

int Tnm_general_square_example() {

    int N1 = 6;
    int N2 = 4;

    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_square(&R1, &R2, N1, N2);


    arma::Mat<float> R_orig = arma::join_vert(R1, R2);
    R_orig.save("R_orig.txt", arma::raw_ascii);

    int nx = 1;
    int ny = 4;

    LatticeGeometryInfo geom_info;
    geom_info.N1 = N1;
    geom_info.N2 = N2;
    geom_info.rx_alpha = { 0.0 };
    geom_info.ry_alpha = { 0.0 };

  
    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0, 1.0 });
    geom_info.A_mat = A_mat;

    printf("\n With new method:\n");
    arma::Mat<float> tmp_mat2 = Tnm_general(R1, R2, nx, ny, geom_info);
    tmp_mat2.print();
    tmp_mat2.save("R_new.txt", arma::raw_ascii);

    return 0;
}


int Tn_group_example() {

    // Size of the lattice
    int N1 = 2;
    int N2 = 2;

    // The basis vectors:
    std::vector<std::vector<float>> A_mat;
    A_mat.push_back({ 1.0,0.0 });
    A_mat.push_back({ 0.0,1.0 });

    LatticeGeometryInfo geom_info = create_square_lattice_info(N1, N2);


    Tnm_group T_group = Tnm_group(geom_info);
    std::vector<std::vector<complex_th>> char_mat = T_group.GetCharTable();
    printMatrix(char_mat);

    int N_tot = N1 * N2;
    arma::Row<float> R1(N_tot);
    arma::Row<float> R2(N_tot);
    create_square(&R1, &R2, N1, N2);

    for (int i = 0; i < N_tot; i++) {
        arma::Mat<float> tmp_mat = T_group.GetGroupElemVec()[i](R1, R2);
        printf("after %d:th operator:\n", i);
        tmp_mat.print();
    }

    return 0;
}


int triangular_lattice_symmetry_example() {

    /* Triangular lattice Heisenberg exact diagonalization with CUDA by
     taking advantage of Abelian symmetries by using pre-defined symmetry groups*/

    complex_th J1 = complex_th(1.0, 0.0);
    complex_th J2 = complex_th(0.15, 0.0);
    //float J2 = 0.0;

    // Size of the lattice
    int N1 = 4;
    int N2 = 4;

    LatticeGeometryInfo geom_info = create_triangular_lattice_info(N1, N2);

    int LS = N1 * N2;

    Tnm_group symm_group(geom_info);

    std::vector<complex_th> J_vec = { J1,J2 };
    std::vector<float> Range_vec = { 1.0001,1.733 };
    int intra_c_bool = 1;
    int inter_c_bool = 1; // should determine whether we have periodic or non-periodic BCs

    T_standard<complex_th> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };

    // Next we create the Hamiltonian
    auto Heisenberg_time = std::chrono::high_resolution_clock::now();

    int max_term = 2 * TLat.getTSize() / (N1 * N2);
    max_term = max_term;

    HeisenbergHamAbelianSymms_CUDA<complex_th> Ham(TLat, B_field_vec, J_dim_weights, LS / 2, &symm_group, max_term);

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate in different irreps sectors: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    // Print the eigenvalues:
    std::vector<std::vector<float>> E_vecs = Ham.get_E_vecs();
    printMatrix<float>(E_vecs);
    float* E_vecs_array = (float*)malloc(sizeof(float) * E_vecs.size() * E_vecs[0].size());
    StdMatToArray<float>(E_vecs, E_vecs_array);


    Operator<complex_th> Sz0(0, OperatorType::Sz,1.0);
    Operator<complex_th> Sz1(3, OperatorType::Sz,1.0);
    std::vector< Operator<complex_th>> aux_vec = { Sz0, Sz1 };

    ManyBodyOperator<complex_th> Sz_mb_wrapper;
    //Sz_mb_wrapper.AddSingleOperator(Sz0);
    Sz_mb_wrapper.AddTerm(aux_vec);

    complex_th Sz0_exp;

    Sz0_exp = Ham.ComputeStaticExpValZeroT(Sz_mb_wrapper);

    std::cout << "\n Result: " << Sz0_exp.real() << ", 1i " << Sz0_exp.imag() << "\n";

    return 0;
}



int triangular_lattice_SzSz_correlator_example() {

    /* Triangular lattice Heisenberg exact diagonalization with CUDA by
     taking advantage of Abelian symmetries by using pre-defined symmetry groups*/

    complex_th J1 = complex_th(1.0, 0.0);
    complex_th J2 = complex_th(0.15, 0.0);

    // Size of the lattice
    int N1 = 6;
    int N2 = 4;

    LatticeGeometryInfo geom_info = create_triangular_lattice_info(N1, N2);

    int LS = N1 * N2;

    Tnm_group symm_group(geom_info);

    // Later on, we should write more generic functions to create automatically the symmetry classes, including the translational invariance

    std::vector<complex_th> J_vec = { J1,J2 };
    std::vector<float> Range_vec = { 1.0001,1.733 };
    int intra_c_bool = 1;
    int inter_c_bool = 1; // should determine whether we have periodic or non-periodic BCs

    T_standard<complex_th> TLat(geom_info, J_vec, Range_vec, intra_c_bool, inter_c_bool);
    std::vector<float> B_field_vec(LS, 0.0f);
    std::vector<float> J_dim_weights = { 1.0,1.0,1.0 };


    // Next we create the Hamiltonian
    auto Heisenberg_time = std::chrono::high_resolution_clock::now();

    int max_term = 2 * TLat.getTSize() / (N1 * N2);
    max_term = max_term;

    HeisenbergHamAbelianSymms_CUDA<complex_th> Ham(TLat, B_field_vec, J_dim_weights, LS / 2, &symm_group, max_term);

    auto Heisenberg_time_end = std::chrono::high_resolution_clock::now();
    auto duration_Heisenberg = std::chrono::duration_cast<std::chrono::milliseconds> (Heisenberg_time_end - Heisenberg_time);
    printf("\n Elapsed time of the creating the %d x %d Heisenberg Hamiltonian and solving its lowest eigenstate in different irreps sectors: ", N1, N2);
    std::cout << duration_Heisenberg.count() << " ms\n";

    // Print the eigenvalues:
    std::vector<std::vector<float>> E_vecs = Ham.get_E_vecs();
    printMatrix<float>(E_vecs);
    float* E_vecs_array = (float*)malloc(sizeof(float) * E_vecs.size() * E_vecs[0].size());
    StdMatToArray<float>(E_vecs, E_vecs_array);


    //////////////////////////
    // Solve here the correlator < Sz_k Sz_-k> for each k in two different ways:
    
    // First, directly in the k-space (should be more slow if we want all the k terms):
    Sz<complex_th> Sz_op(0);
    std::vector<std::vector<std::vector<ManyBodyOperator<complex_th>>>> Sz_k_mat = FullFourierOperator(
        Sz_op, geom_info, false
    );
    std::vector<std::vector<std::vector<ManyBodyOperator<complex_th>>>> Sz_k_mat_neg = FullFourierOperator(
        Sz_op, geom_info, true
    );
    using complex_std = std::complex<float>;

    arma::Mat<complex_std> SzSz_corr_mat(N1, N2, arma::fill::ones);
    arma::Mat<complex_std> SzSz_corr_mat_real_space(N1,N2, arma::fill::ones);

    auto time_method1_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            Sz_k_mat[i][j][0] = Sz_k_mat[i][j][0] * Sz_k_mat_neg[i][j][0];
            complex_th tmp_corr = Ham.ComputeStaticExpValZeroT(Sz_k_mat[i][j][0]);
            SzSz_corr_mat(i,j) = complex_std(tmp_corr.real(), tmp_corr.imag());
            std::cout << "(" << i << "," << j << ")" << "\n Result: " << SzSz_corr_mat(i, j).real() << ", 1i " << SzSz_corr_mat(i, j).imag() << "\n";
        }
    }
    auto time_method1_end = std::chrono::high_resolution_clock::now();
    auto duration_method1 = std::chrono::duration_cast<std::chrono::milliseconds> (time_method1_end - time_method1_start);
    
    SzSz_corr_mat.save("SzSz_corr.txt", arma::raw_ascii);


    // Do the same with the real-space correlations:
    auto time_method2_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N1 ; i++) {
        for (int j = 0; j < N2; j++) {
            Sz<complex_th> Sz_op1(0);
            Sz<complex_th> Sz_op2(j+N2*i);
            ManyBodyOperator<complex_th> SzSz = Sz_op2 * Sz_op1;
            complex_th tmp_corr = Ham.ComputeStaticExpValZeroT(SzSz);
            SzSz_corr_mat_real_space(i,j) = complex_std(tmp_corr.real(), tmp_corr.imag());
        }
    }
    auto time_method2_end = std::chrono::high_resolution_clock::now();
    auto duration_method2 = std::chrono::duration_cast<std::chrono::milliseconds> (time_method2_end - time_method2_start);

    arma::Mat<complex_std> SzSz_corr_mat2 = SimpleFourierTransform(SzSz_corr_mat_real_space, geom_info);
    SzSz_corr_mat2 = sqrt((float)(N1 * N2)) * SzSz_corr_mat2; // due to definition of the simple Fourier transform
    SzSz_corr_mat2.save("SzSz_corr2.txt", arma::raw_ascii);

    arma::Mat<complex_std> SzSz_corr_diff = SzSz_corr_mat - SzSz_corr_mat2;
    SzSz_corr_mat.print();
    printf("Difference:\n");
    SzSz_corr_diff.print();

    printf("Time comparison:\n");
    std::cout << "Method 1: " << duration_method1.count() << " ms, Method 2: " << duration_method2.count() << " ms." << std::endl;

    return 0;
}


int create_XXX_Heisenberg_square() {

    Heisenberg<float> H_model = CreateHeisenbergXXXSquare(2, 4, { 1.0,0.15 });
    ManyBodyOperator<float> Ham = H_model.GetH();
    Ham.PrintOperator();

    return 0;
}


int run_example() {
    int func_ind = 16; // The function we execute

    
    std::vector <std::function<int() >> func_list = {
        complex_example,
        operator_example,
        square_lattice_symmetries_example,
        square_brute_force_example,
        Cn_group_example, //5th
        Tnm_example,
        Tnm_group_example,
        square_lattice_symmetries_example2,
        square_lattice_symmetries_example3,
        operator_arithmetic_example, //10th
        Tnm_general_example,
        Tnm_general_triangular_example,
        Tnm_general_square_example,
        Tn_group_example,
        triangular_lattice_symmetry_example, //15th
        triangular_lattice_SzSz_correlator_example,
        create_XXX_Heisenberg_square
    };

    int out = func_list[func_ind]();
    
    //int out = 1;
    return out;
}