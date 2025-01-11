#pragma once
#include <armadillo>


int complex_example();

int operator_example();

int sqaure_lattice_symmetries_example();

int square_brute_force_example();

void create_triangular(arma::Row<float>* R1, arma::Row<float>* R2, int N1, int N2);

void create_square(arma::Row<float>* R1, arma::Row<float>* R2, int N1, int N2);

int Cn_group_example();

int Tnm_example();

int Tnm_group_example();

int square_lattice_symmetries_example2();

int square_lattice_symmetries_example3();

int operator_arithmetic_example();

int Tnm_general_example();

int Tnm_general_triangular_example();

int Tnm_general_square_example();

int Tn_group_example();

int triangular_lattice_symmetry_example();

int triangular_lattice_SzSz_correlator_example();

int create_XXX_Heisenberg();

int solve_XXX_Heisenberg_square_generic_formalism();

int run_example();