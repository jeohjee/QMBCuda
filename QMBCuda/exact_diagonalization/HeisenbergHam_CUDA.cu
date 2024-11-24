#include "HeisenbergHam_CUDA.h"
#include <thrust/complex.h>
#include "../utils/misc_funcs.h"
#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include <thrust/copy.h>
#include "thrust/extrema.h"
#include <chrono>
#include <iostream>
#include <bitset>
#include "./utils.h"
#include "lanczos.h"

HeisenbergHam_CUDA::HeisenbergHam_CUDA(
	T_standard<float> T_mat_in, 
	std::vector<float> B_field_in, 
	std::vector<float> J_dim_weights,
	int GS_sector_in, 
	int hop_f, 
	int NStates, 
	long long seed,
	const int NIter
)
{

	T_mat = T_mat_in;
	B_field = B_field_in;
	J_weights = J_dim_weights;
	GS_sector = GS_sector_in;

	LS = B_field.size();

	std::cout << "Creating the Hilbert space: \n";
	auto start_time_CUDA_create_Fock = std::chrono::high_resolution_clock::now();

	unsigned long long int nobv = pow(2, LS);

	double nobv_Sz0_cuda_d = NChoosek2(static_cast<double>(LS), static_cast<double>(GS_sector));
	int nobv_Sz0_cuda = static_cast<int>(nobv_Sz0_cuda_d);

	thrust::host_vector<uint32_t> bas_states_host(nobv_Sz0_cuda);

	int i_count = 0;
	int ind_tmp = 0;

	for (int i = 0; i < nobv; i++) {

		int count_tmp = std::bitset<32>(i).count();
		if (count_tmp == GS_sector) {
			bas_states_host[ind_tmp] = i; // set to a thrust host vec
			ind_tmp = ind_tmp += 1;
		}
		i_count = i_count + 1;
		if (i_count == 80000000) {
			std::cout << static_cast<float>(i) / static_cast<float>(nobv) << "\n";
			i_count = 0;
		}
	}

	auto end_time_CUDA_create_Fock = std::chrono::high_resolution_clock::now();
	auto duration_time_Fock = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_CUDA_create_Fock - start_time_CUDA_create_Fock);
	std::cout << "\n Elapsed time of creating the Fock space: " << duration_time_Fock.count() << " ms" << std::endl;

	auto start_time_copy_Fock = std::chrono::high_resolution_clock::now();

	// copy the relevant Fock states to the device memory:
	thrust::device_vector<uint32_t> bas_states_dev(nobv_Sz0_cuda);
	thrust::copy(bas_states_host.begin(), bas_states_host.end(), bas_states_dev.begin());


	// Create the index and value matrices. We might want to use CUBLAS library for this
	//int hop_f = 9; // this value is set by hand by inspecting how many hopping terms one site possibly has (including also the site itself due to the on-site interaction).
	int max_terms = GS_sector * hop_f;

	//auto start_time_copy_Fock = std::chrono::high_resolution_clock::now();

	thrust::device_vector<uint32_t> index_mat_dev(nobv_Sz0_cuda * max_terms);
	thrust::device_vector<float> vals_mat_dev(nobv_Sz0_cuda * max_terms);
	thrust::device_vector<short int> track_non_zero_inds_dev(nobv_Sz0_cuda, 0);

	int threads_GS = 32;
	int blocks_GS = (nobv_Sz0_cuda + threads_GS - 1) / threads_GS;


	std::vector<std::vector<float>> T_mat0 = T_mat.getTmat();

	int T_size = T_mat0.size();
	std::vector<int> T_ind1_vec(T_size);
	std::vector<int> T_ind2_vec(T_size);
	std::vector<float> T_val_vec(T_size);

	for (int i = 0; i < T_size; i++) {
		T_ind1_vec[i] = static_cast<int>(T_mat0[i][0]);
		T_ind2_vec[i] = static_cast<int>(T_mat0[i][1]);

		T_val_vec[i] = T_mat0[i][2];
	}


	thrust::device_vector<int> T_ind1_dev(T_ind1_vec);
	thrust::device_vector<int> T_ind2_dev(T_ind2_vec);
	thrust::device_vector<float> T_val_dev(T_val_vec);

	auto end_time_copy_Fock = std::chrono::high_resolution_clock::now();
	auto duration_time_copy = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_copy_Fock - start_time_copy_Fock);
	std::cout << "\n Elapsed time of copying and setting various vectors: " << duration_time_copy.count() << " ms" << std::endl;

	// Build the Hamiltonian:
	auto start_time_CUDA_create_Ham3 = std::chrono::high_resolution_clock::now();
	BuildHoppingHam_v2 << <blocks_GS, threads_GS >> > (J_dim_weights[0], J_dim_weights[2], GS_sector, bas_states_dev.data().get(), index_mat_dev.data().get(),
		vals_mat_dev.data().get(), track_non_zero_inds_dev.data().get(), max_terms, nobv_Sz0_cuda,
		T_ind1_dev.data().get(), T_ind2_dev.data().get(), T_val_dev.data().get(), T_size);

	cudaDeviceSynchronize();
	auto end_time_CUDA_create_Ham3 = std::chrono::high_resolution_clock::now();
	auto duration_time_CUDA3 = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_CUDA_create_Ham3 - start_time_CUDA_create_Ham3);
	std::cout << "\n Elapsed time of creating the Hamiltonian: " << duration_time_CUDA3.count() << " ms" << std::endl;

	int max_term_real = *(thrust::max_element(track_non_zero_inds_dev.begin(), track_non_zero_inds_dev.end()));
	printf("max term per row is: %d\n", max_term_real);

	float* E_vec = (float*)malloc(sizeof(float) * NStates);
	E_vec[0] = 0;

	uint32_t* ind_mat_ptr = thrust::raw_pointer_cast(&index_mat_dev[0]);
	float* val_mat_ptr = thrust::raw_pointer_cast(&vals_mat_dev[0]);
	short int* track_ptr = thrust::raw_pointer_cast(&track_non_zero_inds_dev[0]);

	auto start_Lanczos = std::chrono::high_resolution_clock::now();
	//const int NIter = 50;

	LanczosEigVals_CUDA(E_vec, NStates, NIter, 
		nobv_Sz0_cuda, max_terms, seed, 
		ind_mat_ptr, val_mat_ptr, track_ptr,
		nullptr, true, true);


	auto end_Lanczos = std::chrono::high_resolution_clock::now();
	auto duration_Lanczos = std::chrono::duration_cast<std::chrono::milliseconds> (end_Lanczos - start_Lanczos);
	std::cout << "\n Elapsed time of the Lanczos CUDA: " << duration_Lanczos.count() << " ms" << std::endl;

}