#include "T_standard.h"
#include <iostream>
#include <map>
#include <utility>
#include <algorithm>
#include <cmath>
#include "thrust/complex.h"

using complex_th = thrust::complex<float>;
using namespace std;

template <typename T> 
T_standard<T>::T_standard(LatticeGeometryInfo _geom_info, vector<T> J_vec, vector<float> Range_vec, int intra_c_bool_in, int inter_c_bool_in)
{

	geom_info = _geom_info;
	vector<float> a1_orig = { geom_info.A_mat[0][0], geom_info.A_mat[1][0] };
	vector<float> a2_orig = { geom_info.A_mat[0][1], geom_info.A_mat[1][1] };
	float Rx_mean = 0.0;
	float Ry_mean = 0.0;

	for (int i = 0; i < geom_info.N1; i++) {
		for (int j = 0; j < geom_info.N2; j++) {
			for (int alpha = 0; alpha < geom_info.rx_alpha.size(); alpha++) {
				float r_x = i * a1_orig[0] + j * a2_orig[0] + geom_info.rx_alpha[alpha];
				float r_y = i * a1_orig[1] + j * a2_orig[1] + geom_info.ry_alpha[alpha];
				R1.emplace_back(r_x);
				R2.emplace_back(r_y);
				Rx_mean = Rx_mean + r_x;
				Ry_mean = Ry_mean + r_y;
			}
		}
	}

	LS = R1.size();

	ComputeR_diffs();

	vector<float> a1 = { a1_orig[0] * geom_info.N1, a1_orig[1] * geom_info.N1 };
	vector<float> a2 = { a2_orig[0] * geom_info.N2, a2_orig[1] * geom_info.N2 };

	float t_hold = 0.0001;

	intra_c_bool = intra_c_bool_in;
	inter_c_bool = inter_c_bool_in;

	if ((intra_c_bool == 0) && (inter_c_bool == 0)) {
		cout << "ERROR: BOTH INTER AND INTRA-CLUSTER BOOLEANS ZERO, ABORTING";
		return;
	}

	int J_size = J_vec.size();

	map < pair<int, int>, vector<float >> r_cells2_x;
	map < pair<int, int>, vector<float >> r_cells2_y;

	map < pair<int, int>, vector<vector<T>> > M_inter;
	//map < pair<int, int>, vector<vector<float>> > M_inter;

	for (int i = -1; i < 2; i++)
	{
		for (int j = -1; j < 2; j++) {

			float tmp_x = static_cast<float>(i) * a1[0] + static_cast<float>(j) * a2[0];
			float tmp_y = static_cast<float>(i) * a1[1] + static_cast<float>(j) * a2[1];

			vector<float> tmp_vec_R1;
			vector<float> tmp_vec_R2;
			for (int lsi = 0; lsi < LS; lsi++)
			{
				tmp_vec_R1.push_back(R1[lsi] + tmp_x);
				tmp_vec_R2.push_back(R2[lsi] + tmp_y);
			}

			r_cells2_x[{i, j}] = tmp_vec_R1;
			r_cells2_y[{i, j}] = tmp_vec_R2;

		}
	}

	for (int i1 = -1; i1 < 2; i1++)
	{
		for (int i2 = -1; i2 < 2; i2++)
		{
			vector<float> r_tmp_x = r_cells2_x[{i1, i2}];
			vector<float> r_tmp_y = r_cells2_y[{i1, i2}];

			vector<vector<T>> M_inter_tmp;

			for (int i = 0; i < LS; i++)
			{
				vector<float> ri_x(static_cast<size_t>(LS), R1[i]);
				vector<float> ri_y(static_cast<size_t>(LS), R2[i]);

				transform(ri_x.begin(), ri_x.end(), r_tmp_x.begin(), ri_x.begin(), minus<float>());
				transform(ri_y.begin(), ri_y.end(), r_tmp_y.begin(), ri_y.begin(), minus<float>());

				vector<float> r_diff(LS);

				for (int lsi = 0; lsi < LS; lsi++) {
					//r_diff[lsi] = sqrt(pow(ri_x[lsi],2) + pow(ri_y[lsi],2));
					float r_diff_tmp = sqrt(pow(ri_x[lsi], 2) + pow(ri_y[lsi], 2));

					if (r_diff_tmp < t_hold) {
						continue;
					}

					for (int j_ind = 0; j_ind < J_size; j_ind++) {
						if (r_diff_tmp > Range_vec[j_ind]) {
							continue;
						}
						else {
							M_inter_tmp.push_back({ static_cast<T>(lsi),static_cast<T>(i),J_vec[j_ind] });
							break;
						}
					}
				}
				M_inter[{i1, i2}] = M_inter_tmp;
			}
		}
	}

	for (int i = -1; i < 2; i++) {
		for (int j = -1; j < 2; j++) {

			if ((i == 0) && (j == 0)) {
				if (intra_c_bool == 0) {
					continue;
				}
			}
			else {
				if (inter_c_bool == 0) {
					continue;
				}
			}

			int T_size_tmp = M_inter[{i, j}].size();
			for (int ti = 0; ti < T_size_tmp; ti++) {
				T_mat.push_back(M_inter[{i, j}][ti]);
			}

		}
	}

	T_size = T_mat.size();

}

template
T_standard<float>::T_standard(LatticeGeometryInfo _geom_info, vector<float> J_vec, vector<float> Range_vec, int intra_c_bool_in, int inter_c_bool_in);

template
T_standard<complex_th>::T_standard(LatticeGeometryInfo _geom_info, vector<complex_th> J_vec, vector<float> Range_vec, int intra_c_bool_in, int inter_c_bool_in);

