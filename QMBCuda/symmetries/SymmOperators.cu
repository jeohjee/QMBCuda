#include "SymmOperators.h"
#define _USE_MATH_DEFINES
#include <math.h>

//rotations:
arma::Mat<float> Cn(arma::Row<float> R1, arma::Row<float> R2, int n, std::vector<float> R0)
{
    float theta = 2 * M_PI / (static_cast<float>(n));
    arma::Mat<float> Rot_mat = { {cos(theta), -sin(theta)}, {sin(theta), cos(theta)} };

    R1 = R1 - R0[0];
    R2 = R2 - R0[1];
    arma::Mat<float> R_mats = join_vert(R1, R2);
    arma::Mat<float> trans_coords = Rot_mat * R_mats;
    trans_coords.row(0) = trans_coords.row(0) + R0[0];
    trans_coords.row(1) = trans_coords.row(1) + R0[1];
    return trans_coords;
}

// reflections
arma::Mat<float> sigma_n_arb(arma::Row<float> R1, arma::Row<float> R2, float theta)
{
    //theta is the angle in degrees, describing the angle of the normal vector of the reflection plane
    arma::Mat<float> R_mats = join_vert(R1, R2);
    theta = theta * M_PI / (180.0);
    arma::Mat<float> Rot_mat = { {cos(theta), -sin(theta)}, {sin(theta), cos(theta)} };
    arma::Col<float> norm_vec = { 1.0, 0.0 };
    norm_vec = Rot_mat * norm_vec;

    arma::Row<float> r_perp = norm_vec.t() * R_mats;
    arma::Mat<float> trans_coords = R_mats - 2 * norm_vec * r_perp;

    return trans_coords;
}

// identity operator:
arma::Mat<float> Iden(arma::Row<float> R1, arma::Row<float> R2)
{
    arma::Mat<float> trans_coords = join_vert(R1, R2);
    return trans_coords;
}

// translational shift in square lattice:
arma::Mat<float> Tnm(arma::Row<float> R1, arma::Row<float> R2, int Nx, int Ny, int nx, int ny)
{
    /* Nx and Ny are the lattice size in x and y directions,
    nx and ny correspond to the actual translation */

    R1 = R1 + (float)nx;
    R2 = R2 + (float)ny;

    for (int i = 0; i < R1.n_cols; i++) {
        R1(i) = std::fmod(R1(i), (float)Nx);
        R2(i) = std::fmod(R2(i), (float)Ny);
    }

    arma::Mat<float> trans_coords = join_vert(R1, R2);
    return trans_coords;
}

// more general trasnlation
arma::Mat<float> Tnm_general(arma::Row<float> R1, arma::Row<float> R2, int nx, int ny, LatticeGeometryInfo geom_info)
{
    /* Nx and Ny are the lattice size in a1 and a2 directions,
    nx and ny correspond to the actual translation */

    /*printf("&&&&\n");
    R1.print();
    R2.print();
    printf("&&&&\n");*/
    //R2.print();

    float tol_coef = 0.001; // this is needed to compensate possible numerical inaccuracies

    std::vector<float> R_origin = { 0.0,0.0 };
    //R_origin[0] = R1(0) - geom_info.rx_alpha[0];
    //R_origin[1] = R2(0) - geom_info.ry_alpha[0];

    arma::Row<float> R1_new = R1 - R_origin[0];
    arma::Row<float> R2_new = R2 - R_origin[1];

    arma::Col<float> a1(2);
    a1(0) = geom_info.A_mat[0][0];
    a1(1) = geom_info.A_mat[1][0];
    arma::Col<float> a2(2);
    a2(0) = geom_info.A_mat[0][1];
    a2(1) = geom_info.A_mat[1][1];

    float W_12 = arma::dot(a1, a2);
    arma::Mat<float> W_mat(2, 2, arma::fill::ones);
    W_mat(0, 1) = W_12;
    W_mat(1, 0) = W_12;
    arma::Mat<float> W_mat_inv = arma::inv(W_mat);

    arma::Col<float> v_proj(2);
    arma::Col<float> alpha_vec(2);

    int N1 = geom_info.N1;
    int N2 = geom_info.N2;

    R1_new = R1_new + (float)nx * a1(0) + (float)ny * a2(0);
    R2_new = R2_new + (float)nx * a1(1) + (float)ny * a2(1);

    for (int i = 0; i < R1.n_cols; i++) {
        arma::Col<float> R_tmp = { {R1_new(i)},{R2_new(i)} };
        v_proj(0) = arma::dot(a1, R_tmp);
        v_proj(1) = arma::dot(a2, R_tmp);
        alpha_vec = W_mat_inv * v_proj;

        alpha_vec(0) = std::fmod(alpha_vec(0), (float)N1);
        alpha_vec(1) = std::fmod(alpha_vec(1), (float)N2);

        if (fabs(alpha_vec(0) - N1) < tol_coef) alpha_vec(0) = 0.0;
        if (fabs(alpha_vec(1) - N2) < tol_coef) alpha_vec(1) = 0.0;

        R1_new(i) = alpha_vec(0) * a1(0) + alpha_vec(1) * a2(0);
        R2_new(i) = alpha_vec(0) * a1(1) + alpha_vec(1) * a2(1);

    }

    R1_new = R1_new + R_origin[0];
    R2_new = R2_new + R_origin[1];

    arma::Mat<float> trans_coords = join_vert(R1_new, R2_new);
    return trans_coords;
}

arma::Mat<float> GetElemProd(
    arma::Row<float> R1, arma::Row<float> R2,
    function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> G2,
    function<arma::Mat<float>(arma::Row<float>, arma::Row<float>)> G1) {

    arma::Mat<float> res1 = G2(R1, R2);
    arma::Row<float> R1_new = res1.row(0);
    arma::Row<float> R2_new = res1.row(1);
    return G1(R1_new, R2_new);
}