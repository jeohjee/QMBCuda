#pragma once
#include "../lattice_models/T_standard.h"
#include <vector>

//rotations:
arma::Mat<float> Cn(arma::Row<float> R1, arma::Row<float> R2, int n, std::vector<float> R0);

// reflections
arma::Mat<float> sigma_n_arb(arma::Row<float> R1, arma::Row<float> R2, float theta);

// identity operator:
arma::Mat<float> Iden(arma::Row<float> R1, arma::Row<float> R2);


// translational shift in square lattice:
arma::Mat<float> Tnm(arma::Row<float> R1, arma::Row<float> R2, int Nx, int Ny, int nx, int ny);

// more general trasnlation
arma::Mat<float> Tnm_general(arma::Row<float> R1, arma::Row<float> R2, int nx, int ny, LatticeGeometryInfo geom_info);

arma::Mat<float> GetElemProd(
    arma::Row<float> R1, arma::Row<float> R2,
    function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> G2,
    function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> G1);