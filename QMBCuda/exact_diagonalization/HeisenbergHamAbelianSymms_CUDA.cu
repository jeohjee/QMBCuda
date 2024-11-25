#include "HeisenbergHamAbelianSymms_CUDA.h"
#include <thrust/copy.h>
#include "thrust/sort.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"
#include "thrust/reduce.h"
#include <iostream>
#include <bitset>
#include "utils.h"
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "../utils/print_funcs.h"
#include "../utils/misc_funcs.h"
#include "../utils/misc_funcs_gpu.h"
#include <chrono>
#include "lanczos.h"

using complex_th = thrust::complex<float>;

template <typename T> HeisenbergHamAbelianSymms_CUDA<T>::HeisenbergHamAbelianSymms_CUDA(
	T_standard<T> T_mat_in, 
	std::vector<float> B_field_in, 
	std::vector<float> J_dim_weights,
	int GS_sector_in, 
	SymmetryClass<T>* SymmGroup_in, 
	int hop_f_in, 
	int NStates_in, 
	long long seed,
	const int NIter_in, 
	float tol_norm_in
)
{
	T_mat = T_mat_in;
	B_field = B_field_in; // TO DO: this variable is not used at the moment
	J_weights = J_dim_weights;
	GS_sector = GS_sector_in;
	SymmGroup = SymmGroup_in;
	LS = B_field.size();
	NStates = NStates_in;
	hop_f = hop_f_in;
	tol_norm = tol_norm_in;
	NIter = NIter_in;

	int threads_GS = 32;
	int blocks_GS;

	vector<vector<T>> char_mat = SymmGroup->GetCharTable(); // this assumes that char table in SymmGroup is form of T. Should be made more safe

	int GSize = SymmGroup->GetGroupElemVec().size();
	int NIr = SymmGroup->GetCharTable().size(); // number of irreducible representations

	// As the old code uses armadillo, one needs to transform the group elements to a format accessible to CUDA:
	group_el_arr = new uint32_t[LS * GSize];
	SymmGroup->GetGroupElemArr(T_mat.GetR1(), T_mat.GetR2(), group_el_arr); // Store the information on the group elements to array
	printMatrixRowMajor_NonSq(group_el_arr, GSize, LS);

	// Copy the group elements to the device memory:
	thrust::device_vector<uint32_t> group_el_dev(group_el_arr, group_el_arr + LS * GSize);

	// Copy the character matrix from Arma form:
	char_mat_arr = new T[GSize * GSize];
	TransformMatArr(char_mat_arr, &char_mat, GSize, GSize);
	thrust::device_vector<T> char_mat_dev(char_mat_arr, char_mat_arr + GSize * GSize);

	//printf("The character matrix is:\n");
	//printDeviceMatrix<T>(char_mat_dev, GSize, GSize);

	// Copy the content of T_mat to device memory:
	int T_size = T_mat.getTSize();
	thrust::device_vector<int> T_ind1_dev(T_size, 0);
	thrust::device_vector<int> T_ind2_dev(T_size, 0);
	thrust::device_vector<T> T_val_dev(T_size, 0);
	BuildTmats(T_size, T_ind1_dev, T_ind2_dev, T_val_dev);

	//printf("first column of T_mat:\n");
	//thrust::copy(T_ind1_dev.begin(), T_ind1_dev.end() - 1, std::ostream_iterator<int>(std::cout, "\n"));

	// From here onwards, we will build Hamiltonians. 
	unsigned long long int nobv = pow(2, LS);

	auto start_time_hs_loop = std::chrono::high_resolution_clock::now();

	// Set up the corresponding Hilbert space:
	int GS_fill = GS_sector;
	int max_terms = (GS_fill * hop_f) / 2;

	double nobv_Sz0_cuda_d = NChoosek2(static_cast<double>(LS), static_cast<double>(GS_fill));
	int nobv_Sz0_cuda = static_cast<int>(nobv_Sz0_cuda_d);
	nobv_gs = nobv_Sz0_cuda;

	thrust::device_vector<uint32_t> bas_states_dev(nobv_Sz0_cuda);
	BuildBasisStates(bas_states_dev, nobv);

	printf("Set up the super representative states(SRS)\n");
	thrust::device_vector<uint32_t> SRS_states(nobv_Sz0_cuda * GSize);
	thrust::device_vector<uint32_t> SRS_states_min(nobv_Sz0_cuda);
	thrust::device_vector<float> norm_vecs(1, 0.0);
	thrust::device_vector<uint32_t> orbit_indices(2 * nobv_Sz0_cuda, 0);

	SRSBuildInfo SRS_build_info;
	SRS_build_info.nobv_Sz0_cuda = nobv_Sz0_cuda;
	SRS_build_info.GSize = GSize;
	SRS_build_info.threads_GS = threads_GS;
	SRS_build_info.NIr = NIr;
	SRS_build_info.LS = LS;
	SRS_build_info.GS_fill = GS_fill;

	int SRS_unique_count;

	SRS_unique_count = BuildSRStates(SRS_build_info,
		bas_states_dev, SRS_states, SRS_states_min,
		group_el_dev, norm_vecs, orbit_indices, char_mat_dev);

	printf("Allocate the arrays needed to represnt the Hamiltonian\n");
	thrust::device_vector<uint32_t> index_mat_dev(SRS_unique_count * max_terms);
	thrust::device_vector<T> vals_mat_dev(SRS_unique_count * max_terms);
	thrust::device_vector<short int> track_non_zero_inds_dev(SRS_unique_count, 0);

	float E_gs;

	// Form the Hamiltonians for each irrep. by looping over the irreps.
	for (int alpha = 0; alpha < NIr; alpha++) {
		printf("irrep: %d\n", alpha);

		thrust::fill(index_mat_dev.begin(), index_mat_dev.end(), 0);
		thrust::fill(vals_mat_dev.begin(), vals_mat_dev.end(), 0.0);
		thrust::fill(track_non_zero_inds_dev.begin(), track_non_zero_inds_dev.end(), 0);

		// Build the Hamiltonian:
		auto start_time_CUDA_create_Ham3 = std::chrono::high_resolution_clock::now();

		blocks_GS = (SRS_unique_count + threads_GS - 1) / threads_GS;
		BuildHamiltonianForAbelianGroup << <blocks_GS, threads_GS >> > (J_dim_weights[0], J_dim_weights[2], GS_fill, bas_states_dev.data().get(), index_mat_dev.data().get(),
			vals_mat_dev.data().get(), track_non_zero_inds_dev.data().get(), max_terms, SRS_unique_count,
			T_ind1_dev.data().get(), T_ind2_dev.data().get(), T_val_dev.data().get(), T_size,
			SRS_states.data().get(), GSize, orbit_indices.data().get(), char_mat_dev.data().get(), alpha,
			norm_vecs.data().get(), NIr, tol_norm);

		cudaDeviceSynchronize();
		auto end_time_CUDA_create_Ham3 = std::chrono::high_resolution_clock::now();
		auto duration_time_CUDA3 = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_CUDA_create_Ham3 - start_time_CUDA_create_Ham3);
		std::cout << "\n Elapsed time of creating the Hamiltonian: " << duration_time_CUDA3.count() << " ms" << std::endl;

		//printDeviceMatrix<T>(vals_mat_dev, SRS_unique_count, max_terms);
		//printDeviceMatrix<uint32_t>(index_mat_dev, SRS_unique_count, max_terms);

		int max_term_real = *(thrust::max_element(track_non_zero_inds_dev.begin(), track_non_zero_inds_dev.end()));
		printf("max term per row is: %d and the max_terms parameter is %d\n", max_term_real, max_terms);

		/////////////// saving the val_mat and index_mat to disk:
		/*thrust::host_vector<float>vals_mat_host(SRS_unique_count * max_terms);
		thrust::host_vector<uint32_t>index_mat_host(SRS_unique_count * max_terms);
		thrust::copy(vals_mat_dev.begin(), vals_mat_dev.end(), vals_mat_host.begin());
		thrust::copy(index_mat_dev.begin(), index_mat_dev.end(), index_mat_host.begin());

		float* val_mat_ptr2 = (float*)malloc(sizeof(float)* SRS_unique_count * max_terms);
		float* ind_mat_ptr2 = (float*)malloc(sizeof(float) * SRS_unique_count * max_terms);

		for (int pi = 0; pi < SRS_unique_count * max_terms; pi++) val_mat_ptr2[pi] = vals_mat_host[pi];
		for (int pi = 0; pi < SRS_unique_count * max_terms; pi++) ind_mat_ptr2[pi] = (float)index_mat_host[pi];

		WriteArrayToFile("vals_mat.bin", val_mat_ptr2, SRS_unique_count * max_terms);
		WriteArrayToFile("ind_mat.bin", ind_mat_ptr2, SRS_unique_count * max_terms);*/
		/////////////////////////////////////////////////////////

		std::vector<float> E_vec_tmp(NStates, 0.0);
		std::vector<long long> seed_vec_tmp(NStates, 0);

		// Atm, one should keep NStates=1
		auto start_Lanczos = std::chrono::high_resolution_clock::now();

		float* krylov_cand_vec = (float*)malloc(sizeof(float) * NIter);

		for (int i_ex = 0; i_ex < NStates; i_ex++) {

			bool extract_Krylov = (i_ex == 0);
			bool checkEigVec = (i_ex == 0);

			// Next, one needs to solve the obtained Hamiltonian with the Lanzcos
			float E_tmp = 0;

			uint32_t* ind_mat_ptr = thrust::raw_pointer_cast(&index_mat_dev[0]);
			T* val_mat_ptr = thrust::raw_pointer_cast(&vals_mat_dev[0]);
			short int* track_ptr = thrust::raw_pointer_cast(&track_non_zero_inds_dev[0]);


			//const int NIter = 50;

			float conv_err = 1.0;
			float conv_thold = 0.0001;

			long long seed_tmp = seed;
			conv_err = LanczosEigVals_CUDA(&E_tmp, 1, NIter, SRS_unique_count,
				max_terms, seed_tmp, ind_mat_ptr, val_mat_ptr, track_ptr,
				krylov_cand_vec, extract_Krylov, checkEigVec);

			int conv_test = 1;
			srand(time(NULL));
			while ((conv_err > conv_thold) && conv_test < 10) {
				printf("\n Eigenvalue not passing the convergence test, trying different seed. conv_test: %d\n", conv_test);
				seed_tmp = (long long)rand();
				printf("With seed %lld \n", seed_tmp);
				conv_err = LanczosEigVals_CUDA(&E_tmp, 1, NIter, SRS_unique_count,
					max_terms, seed_tmp, ind_mat_ptr, val_mat_ptr, track_ptr,
					krylov_cand_vec, extract_Krylov, checkEigVec);
				conv_test += 1;
			}
			if (conv_err > conv_thold) {
				E_vec_tmp[i_ex] = 10000000;
				printf("Eigensolution was not found!");
			}
			else E_vec_tmp[i_ex] = E_tmp;
			//E_vec_tmp[i_ex] = E_tmp;
			printf("E_tmp is %f", E_tmp);
			seed_vec_tmp[i_ex] = seed_tmp;
		}

		E_vecs.emplace_back(E_vec_tmp);
		seed_vecs.emplace_back(seed_vec_tmp);

		auto end_Lanczos = std::chrono::high_resolution_clock::now();
		auto duration_Lanczos = std::chrono::duration_cast<std::chrono::milliseconds> (end_Lanczos - start_Lanczos);
		std::cout << "\n Elapsed time of the Lanczos CUDA: " << duration_Lanczos.count() << " ms" << std::endl;

		std::vector<float> curr_krylov_vec(krylov_cand_vec, krylov_cand_vec + NIter);
		KrylovGS_vecs.emplace_back(curr_krylov_vec);
		free(krylov_cand_vec);
	}

	// Set the ground state properties:
	E_gs = E_vecs[0][0];
	GS_irrep = 0;
	GS_seed = seed_vecs[0][0];
	for (int alpha = 1; alpha < NIr; alpha++) {
		if (E_vecs[alpha][0] < E_gs) {
			E_gs = E_vecs[alpha][0];
			GS_irrep = alpha;
			GS_seed = seed_vecs[alpha][0];
		}
	}
	KrylovGS_vec = KrylovGS_vecs[GS_irrep];

	auto end_time_hs_loop = std::chrono::high_resolution_clock::now();
	auto duration_time_hs = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_hs_loop - start_time_hs_loop);
	std::cout << "\n Elapsed time of hs loop: " << duration_time_hs.count() << " ms" << std::endl;

	printf("//////////////////////////////\n");
	printf("Ground state energy: %f, ground state irrep: %d\n", E_gs, GS_irrep);
	printf("//////////////////////////////\n");
}

template HeisenbergHamAbelianSymms_CUDA<float>::HeisenbergHamAbelianSymms_CUDA(
	T_standard<float> T_mat_in,
	std::vector<float> B_field_in,
	std::vector<float> J_dim_weights,
	int GS_sector_in,
	SymmetryClass<float>* SymmGroup_in,
	int hop_f_in,
	int NStates_in = 1,
	long long seed = 2212456,
	const int NIter_in = 50,
	float tol_norm_in = 0.00001
);

template HeisenbergHamAbelianSymms_CUDA<complex_th>::HeisenbergHamAbelianSymms_CUDA(
	T_standard<complex_th> T_mat_in,
	std::vector<float> B_field_in,
	std::vector<float> J_dim_weights,
	int GS_sector_in,
	SymmetryClass<complex_th>* SymmGroup_in,
	int hop_f_in,
	int NStates_in = 1,
	long long seed = 2212456,
	const int NIter_in = 50,
	float tol_norm_in = 0.00001
);



template <typename T>
void HeisenbergHamAbelianSymms_CUDA<T>::BuildBasisStates(
	thrust::device_vector<uint32_t>& bas_states_dev, 
	unsigned long long int nobv
) 
{
	int ind_tmp = 0;
	thrust::host_vector<uint32_t> bas_states_host(nobv_gs);

	for (int i = 0; i < nobv; i++) {

		int count_tmp = std::bitset<32>(i).count();
		if (count_tmp == GS_sector) {
			bas_states_host[ind_tmp] = i; // set to a thrust host vec
			ind_tmp = ind_tmp += 1;
		}
		if (i % 80000000 == 0) std::cout << "building the Hilbert space: " << static_cast<float>(i) / static_cast<float>(nobv) << "\n";
	}
	printf("copy the relevant Fock states to the device memory\n");
	thrust::copy(bas_states_host.begin(), bas_states_host.end(), bas_states_dev.begin());
}
template void HeisenbergHamAbelianSymms_CUDA<float>::BuildBasisStates(
	thrust::device_vector<uint32_t>& bas_states_dev,
	unsigned long long int nobv
);
template void HeisenbergHamAbelianSymms_CUDA<complex_th>::BuildBasisStates(
	thrust::device_vector<uint32_t>& bas_states_dev,
	unsigned long long int nobv
);


template <typename T>
void HeisenbergHamAbelianSymms_CUDA<T>::BuildTmats(
	int T_size, 
	thrust::device_vector<int>& T_ind1_dev,
	thrust::device_vector<int>& T_ind2_dev, 
	thrust::device_vector<T>& T_val_dev
) 
{
	std::vector<std::vector<T>> T_mat0 = T_mat.getTmat();

	for (int i = 0; i < T_size; i++) {
		T_ind1_dev[i] = (int)ExtReal(T_mat0[i][0]);
		T_ind2_dev[i] = (int)ExtReal(T_mat0[i][1]);
		T_val_dev[i] = (T)T_mat0[i][2];
	}
}
template void HeisenbergHamAbelianSymms_CUDA<float>::BuildTmats(
	int T_size,
	thrust::device_vector<int>& T_ind1_dev,
	thrust::device_vector<int>& T_ind2_dev,
	thrust::device_vector<float>& T_val_dev
);
template void HeisenbergHamAbelianSymms_CUDA<complex_th>::BuildTmats(
	int T_size,
	thrust::device_vector<int>& T_ind1_dev,
	thrust::device_vector<int>& T_ind2_dev,
	thrust::device_vector<complex_th>& T_val_dev
);


template <typename T>
int HeisenbergHamAbelianSymms_CUDA<T>::BuildSRStates(SRSBuildInfo build_info,
	thrust::device_vector<uint32_t>& bas_states_dev,
	thrust::device_vector<uint32_t>& SRS_states,
	thrust::device_vector<uint32_t>& SRS_states_min,
	thrust::device_vector<uint32_t>& group_el_dev,
	thrust::device_vector<float>& norm_vecs,
	thrust::device_vector<uint32_t>& orbit_indices,
	thrust::device_vector<T>& char_mat_dev)
{
	int nobv_Sz0_cuda = build_info.nobv_Sz0_cuda;
	int threads_GS = build_info.threads_GS;
	int GSize = build_info.GSize;
	int LS = build_info.LS;
	int GS_fill = build_info.GS_fill;
	int NIr = build_info.NIr;

	int blocks_GS = (nobv_Sz0_cuda * GSize + threads_GS - 1) / threads_GS;

	SetUpSRS << <blocks_GS, threads_GS >> > (bas_states_dev.data().get(), SRS_states.data().get()
		, group_el_dev.data().get(), nobv_Sz0_cuda, GSize, LS, GS_fill);
	cudaDeviceSynchronize();

	printf("Solving the minimum SRS states\n");
	blocks_GS = (nobv_Sz0_cuda + threads_GS - 1) / threads_GS;
	SolveSRS_min << <blocks_GS, threads_GS >> > (SRS_states.data().get(), SRS_states_min.data().get(), nobv_Sz0_cuda, GSize);
	cudaDeviceSynchronize();

	thrust::sort(SRS_states_min.begin(), SRS_states_min.end());
	int SRS_unique_count = thrust::unique_count(thrust::device, SRS_states_min.begin(), SRS_states_min.end());
	printf("SRS_unique_count: %d, nobv_gs: %d \n", SRS_unique_count, nobv_Sz0_cuda);

	//printf("Sorted form:\n");
	//thrust::copy(SRS_states_min.begin(), SRS_states_min.end(), std::ostream_iterator<int>(std::cout, "\n"));

	auto SRS_p = thrust::unique(SRS_states_min.begin(), SRS_states_min.end());

	//printf("Sorted unique form:\n");
	//thrust::copy(SRS_states_min.begin(), SRS_states_min.begin()+ SRS_unique_count, std::ostream_iterator<int>(std::cout, "\n"));

	/*
	thrust::host_vector<uint32_t> SRS_states_h(nobv_Sz0_cuda * GSize);
	thrust::copy(SRS_states.begin(), SRS_states.end(), SRS_states_h.begin());
	printMatrixRowMajor_NonSq(thrust::raw_pointer_cast(&SRS_states_h[0]), nobv_Sz0_cuda, GSize);
	printf("bas vecs:\n");
	thrust::copy(bas_states_dev.begin(), bas_states_dev.end(), std::ostream_iterator<int>(std::cout, "\n"));
	*/

	// One has to also actually form the orbits from the generators of the orbits
	// This could be done in a smart way by permuting in an approriate way the elements of SRS_states. Here we use a simpler brute-force way
	// by using SetUpSRS
	printf("Setting up the final orbits\n");
	blocks_GS = (SRS_unique_count * GSize + threads_GS - 1) / threads_GS;
	SetUpSRS_from_indices << < blocks_GS, threads_GS >> > (SRS_states_min.data().get(), SRS_states.data().get(), group_el_dev.data().get(), SRS_unique_count, GSize, LS, GS_fill, bas_states_dev.data().get());
	cudaDeviceSynchronize();

	// NOTE THAT SRS STATES ARE LABELLED BY THEIR INDEX IN THE CORRESPONDING HILBERT SPACE SECTOR.

	//printf("Sorted unique form:\n");
	//thrust::copy(SRS_states_min.begin(), SRS_states_min.begin()+ SRS_unique_count, std::ostream_iterator<uint32_t>(std::cout, "\n"));
	//printf("And final created SRS state indices:\n");
	//thrust::copy(SRS_states.begin(), SRS_states.begin() + SRS_unique_count*GSize, std::ostream_iterator<uint32_t>(std::cout, "\n"));

	// Now first SRS_unique_count rows of SRS_states should hold all the required orbits
	// Next, We need to normalize these orbits:

	printf("Compute the norm of the SRSs for each irrep\n");
	/* Remember that SRS_unique_count is now the number of orbits. */
	norm_vecs.resize(SRS_unique_count * NIr);
	blocks_GS = (SRS_unique_count * NIr + threads_GS - 1) / threads_GS;
	ComputeSRSNorms << <blocks_GS, threads_GS >> > (norm_vecs.data().get(), SRS_states.data().get(), char_mat_dev.data().get(), SRS_unique_count, NIr, GSize, bas_states_dev.data().get());
	cudaDeviceSynchronize();

	//printf("Norms:\n");
	//thrust::copy(norm_vecs.begin(), norm_vecs.begin()+ SRS_unique_count*NIr, std::ostream_iterator<float>(std::cout, "\n"));

	// Finally, one needs to have mapping from the bas_vecs to the orbit and the corresponding order index within the orbit:
	printf("Compute the mapping from the Hilbert space to the orbits\n");

	blocks_GS = (SRS_unique_count * GSize + threads_GS - 1) / threads_GS;
	ComputeOrbitIndices << <blocks_GS, threads_GS >> > (orbit_indices.data().get(), SRS_unique_count, GSize, SRS_states.data().get(), GS_fill);
	cudaDeviceSynchronize();

	//printDeviceMatrix<uint32_t>(orbit_indices, nobv_Sz0_cuda, 2,"OrbitIndices");
	//thrust::copy(orbit_indices.begin(), orbit_indices.end(), std::ostream_iterator<T>(std::cout, ", "));

	return SRS_unique_count;
}

template int HeisenbergHamAbelianSymms_CUDA<float>::BuildSRStates(SRSBuildInfo build_info,
	thrust::device_vector<uint32_t>& bas_states_dev,
	thrust::device_vector<uint32_t>& SRS_states,
	thrust::device_vector<uint32_t>& SRS_states_min,
	thrust::device_vector<uint32_t>& group_el_dev,
	thrust::device_vector<float>& norm_vecs,
	thrust::device_vector<uint32_t>& orbit_indices,
	thrust::device_vector<float>& char_mat_dev);

template int HeisenbergHamAbelianSymms_CUDA<complex_th>::BuildSRStates(SRSBuildInfo build_info,
	thrust::device_vector<uint32_t>& bas_states_dev,
	thrust::device_vector<uint32_t>& SRS_states,
	thrust::device_vector<uint32_t>& SRS_states_min,
	thrust::device_vector<uint32_t>& group_el_dev,
	thrust::device_vector<float>& norm_vecs,
	thrust::device_vector<uint32_t>& orbit_indices,
	thrust::device_vector<complex_th>& char_mat_dev);


template <typename T> complex_th HeisenbergHamAbelianSymms_CUDA<T>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
)
{
	int threads_GS = 32;
	// The implementation of this function depends on the both the model and the method. Thus, we determine here which
	// single-particle operators are appropriate and how they map to the CUDA functions. This implementation is bit cumbersome due
	// to the limitations of CUDA.

	// First determine the mapping from operators to CUDA action:
	std::map<OperatorType, int> op_action_map;
	op_action_map[OperatorType::Sz] = 0;
	op_action_map[OperatorType::Sp] = 1;
	op_action_map[OperatorType::Sm] = -1;

	int GSize = SymmGroup->GetGroupElemVec().size();

	// Allocate CUDA operators:
	int A_size = A.GetElems().size();
	int A_col_size = A.GetMaxTerms();
	thrust::host_vector<int> A_decode_table(A_size * A_col_size, 0);
	thrust::host_vector<complex_th> A_scalar_table(A_size, (complex_th)1.0);
	thrust::host_vector<int> A_site_table(A_size * A_col_size, -1); // We initialize this to -1 as -1 indicates dummy elements
	thrust::host_vector<int> A_NOTerms(A_size, 0); // It is convenient to have this

	for (int ai = 0; ai < A_size; ai++) {
		A_NOTerms[ai] = A.GetElems()[ai].size();
		for (int ai2 = 0; ai2 < A.GetElems()[ai].size(); ai2++) {
			A_decode_table[ai * A_col_size + ai2] = (-1) * op_action_map[A.GetElems()[ai][ai2].GetType()]; //-1 implements the hermitian conjugate (should be generalized)
			A_site_table[ai * A_col_size + ai2] = A.GetElems()[ai][ai2].GetSite();
			A_scalar_table[ai] = A_scalar_table[ai] * (GetComplexConjugateHost(A.GetElems()[ai][ai2].GetScalar()));
		}
	}

	// Copy to device memory:
	thrust::device_vector<int> A_decode_table_dev(A_size * A_col_size);
	thrust::device_vector<complex_th> A_scalar_table_dev(A_size);
	thrust::device_vector<int> A_site_table_dev(A_size * A_col_size);
	thrust::device_vector<int> A_NOTerms_dev(A_size);

	thrust::copy(A_decode_table.begin(), A_decode_table.end(), A_decode_table_dev.begin());
	thrust::copy(A_scalar_table.begin(), A_scalar_table.end(), A_scalar_table_dev.begin());
	thrust::copy(A_site_table.begin(), A_site_table.end(), A_site_table_dev.begin());
	thrust::copy(A_NOTerms.begin(), A_NOTerms.end(), A_NOTerms_dev.begin());


	// One needs to unfortunately maybe build again explicitly the required SRS states and norms in the same way as in case of the Hamiltonian
	unsigned long long int nobv = pow(2, LS);

	int GS_fill = GS_sector;
	if (max_terms == 0) max_terms = (GS_fill * hop_f) / 2;
	int nobv_Sz0_cuda = nobv_gs;

	int NIr = SymmGroup->GetCharTable().size(); // number of irreducible representations

	thrust::device_vector<uint32_t> bas_states_dev(nobv_Sz0_cuda);
	BuildBasisStates(bas_states_dev, nobv);

	printf("Set up the super representative states(SRS)\n");
	thrust::device_vector<uint32_t> SRS_states(nobv_Sz0_cuda * GSize);
	thrust::device_vector<uint32_t> SRS_states_min(nobv_Sz0_cuda);
	thrust::device_vector<float> norm_vecs(1, 0.0);
	thrust::device_vector<uint32_t> orbit_indices(2 * nobv_Sz0_cuda, 0);
	thrust::device_vector<uint32_t> group_el_dev(group_el_arr, group_el_arr + LS * GSize);
	thrust::device_vector<T> char_mat_dev(char_mat_arr, char_mat_arr + GSize * GSize); // Char mat

	SRSBuildInfo SRS_build_info;
	SRS_build_info.nobv_Sz0_cuda = nobv_Sz0_cuda;
	SRS_build_info.GSize = GSize;
	SRS_build_info.threads_GS = threads_GS;
	SRS_build_info.NIr = NIr;
	SRS_build_info.LS = LS;
	SRS_build_info.GS_fill = GS_fill;

	int SRS_unique_count;

	SRS_unique_count = BuildSRStates(SRS_build_info,
		bas_states_dev, SRS_states, SRS_states_min,
		group_el_dev, norm_vecs, orbit_indices, char_mat_dev);

	// Unfortunately, we also need to build the Hamiltonian for the ground state irrep.
	printf("Allocate the arrays needed to represnt the Hamiltonian\n");
	thrust::device_vector<uint32_t> index_mat_dev(SRS_unique_count * max_terms);
	thrust::device_vector<T> vals_mat_dev(SRS_unique_count * max_terms, (T)0.0);
	thrust::device_vector<short int> track_non_zero_inds_dev(SRS_unique_count, 0);

	uint32_t* ind_mat_ptr = thrust::raw_pointer_cast(&index_mat_dev[0]);
	T* val_mat_ptr = thrust::raw_pointer_cast(&vals_mat_dev[0]);
	short int* track_ptr = thrust::raw_pointer_cast(&track_non_zero_inds_dev[0]);

	int blocks_GS = (SRS_unique_count + threads_GS - 1) / threads_GS;

	int T_size = T_mat.getTSize();
	thrust::device_vector<int> T_ind1_dev(T_size, 0);
	thrust::device_vector<int> T_ind2_dev(T_size, 0);
	thrust::device_vector<T> T_val_dev(T_size, 0);
	BuildTmats(T_size, T_ind1_dev, T_ind2_dev, T_val_dev);

	BuildHamiltonianForAbelianGroup << <blocks_GS, threads_GS >> > (
		J_weights[0], 
		J_weights[2], 
		GS_fill, 
		bas_states_dev.data().get(), 
		index_mat_dev.data().get(),
		vals_mat_dev.data().get(), 
		track_non_zero_inds_dev.data().get(), 
		max_terms, 
		SRS_unique_count,
		T_ind1_dev.data().get(), 
		T_ind2_dev.data().get(), 
		T_val_dev.data().get(), 
		T_size,
		SRS_states.data().get(), 
		GSize, 
		orbit_indices.data().get(), 
		char_mat_dev.data().get(), 
		GS_irrep,
		norm_vecs.data().get(), 
		NIr, 
		tol_norm
		);
	cudaDeviceSynchronize();

	// Retrieve the ground state
	thrust::device_vector<T> final_eig_vec(SRS_unique_count, 0.0);

	LanczosEigenStateFromSeed(GS_seed, thrust::raw_pointer_cast(&final_eig_vec[0]),
		KrylovGS_vec, SRS_unique_count, NIter, max_terms,
		ind_mat_ptr, val_mat_ptr, track_ptr, 512);

	// call CUDA function to compture the expectation value
	// As we compute a simple zero-T expectation value, we do not need to store the operator as a matrix but we can directly compute the 
	// expectation value

	// temporal solution for char_mat, substitute later with a more general solution:
	thrust::device_vector<thrust::complex<float>> char_mat_dev_c(GSize * GSize);
	for (int gi = 0; gi < GSize * GSize; gi++) char_mat_dev_c[gi] = thrust::complex<float>(char_mat_arr[gi]);

	printDeviceMatrix(final_eig_vec, 10, 1, "final eig vec in dev space:", "\n");
	thrust::device_vector<complex_th> A_exp_vec(SRS_unique_count, (complex_th)0.0);

	ExpecValArbOperatorAbelianIrrepProjHeisenbergZeroT << <blocks_GS, threads_GS >> > (
		A_scalar_table_dev.data().get(),
		A_decode_table_dev.data().get(), 
		A_site_table_dev.data().get(), 
		A_NOTerms_dev.data().get(),
		max_terms, SRS_unique_count, 
		SRS_states.data().get(),
		GSize,
		GS_sector,
		orbit_indices.data().get(),
		char_mat_dev_c.data().get(),
		GS_irrep,
		norm_vecs.data().get(),
		NIr,
		tol_norm,
		A_size,
		A_col_size,
		bas_states_dev.data().get(),
		final_eig_vec.data().get(),
		A_exp_vec.data().get()
		);

	complex_th A_sum = thrust::reduce(A_exp_vec.begin(), A_exp_vec.end(), (complex_th)0.0, thrust::plus<complex_th>());
	A_sum = A_sum / GSize; // this step is apparently needed due to the Kronecker delta form within the operator
	printf("Result:\n");
	std::cout << "A sum is: " << A_sum << std::endl;
	return A_sum;
}

template complex_th HeisenbergHamAbelianSymms_CUDA<float>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
);

template complex_th HeisenbergHamAbelianSymms_CUDA<complex_th>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
);
// max_terms = 0