#include "HeisenbergGenericXXZ_AbelianSymms_CUDA.h"
#include <thrust/copy.h>
#include "thrust/sort.h"
#include "thrust/fill.h"
#include "thrust/host_vector.h"
#include "thrust/reduce.h"
#include <iostream>
#include <bitset>
#include <thrust/unique.h>
#include <thrust/execution_policy.h>
#include "../utils/print_funcs.h"
#include "../utils/misc_funcs.h"
#include "../utils/misc_funcs_gpu.h"
#include <chrono>
#include "lanczos.h"
#include "../quantum_operators/SingleParticleOperators.h"

using complex_th = thrust::complex<float>;

template <typename T> HeisenbergXXZAbelianSymms_CUDA<T>::HeisenbergXXZAbelianSymms_CUDA(
	Heisenberg<T> _lattice_model,
	SymmetryClass<T>* SymmGroup_in,
	int hop_f,
	int GS_sector_in,
	int NStates_in,
	long long seed,
	const int NIter_in,
	float tol_norm_in
)
{
	lattice_model = _lattice_model;
	LS = lattice_model.GetLS();
	SymmGroup = SymmGroup_in;

	if (GS_sector_in < 0) GS_sector = LS / 2;
	else GS_sector = GS_sector_in;

	NStates = NStates_in;
	tol_norm = tol_norm_in;
	NIter = NIter_in;

	int threads_GS = 32;
	int blocks_GS;

	vector<vector<T>> char_mat = SymmGroup->GetCharTable(); // this assumes that char table in SymmGroup is form of T. Should be made more safe

	int GSize = SymmGroup->GetGroupElemVec().size();
	int NIr = SymmGroup->GetCharTable().size(); // number of irreducible representations

	// As the old code uses armadillo, one needs to transform the group elements to a format accessible to CUDA:
	group_el_arr = new uint32_t[LS * GSize];
	SymmGroup->GetGroupElemArr(lattice_model.GetR1(), lattice_model.GetR2(), group_el_arr); // Store the information on the group elements to array
	printMatrixRowMajor_NonSq(group_el_arr, GSize, LS);

	// Copy the group elements to the device memory:
	thrust::device_vector<uint32_t> group_el_dev(group_el_arr, group_el_arr + LS * GSize);

	// Copy the character matrix from Arma form:
	char_mat_arr = new T[GSize * GSize];
	TransformMatArr(char_mat_arr, &char_mat, GSize, GSize);
	thrust::device_vector<T> char_mat_dev(char_mat_arr, char_mat_arr + GSize * GSize);

	// From here onwards, we will build Hamiltonians. 
	unsigned long long int nobv = pow(2, LS);

	auto start_time_hs_loop = std::chrono::high_resolution_clock::now();

	// Set up the corresponding Hilbert space:
	int GS_fill = GS_sector;
	// 2 * TLat.getTSize() / (N1 * N2);
	if (hop_f > 0) max_coupling_terms = (GS_fill * hop_f);
	else max_coupling_terms = lattice_model.GetH().GetElems().size() * GS_fill / LS + 1;
	int max_terms = max_coupling_terms;

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

	// Allocate CUDA operators:
	ManyBodyOperator<T> H = lattice_model.GetH();
	OperatorVectors<T> Ham_vecs = BuildOperatorVecs(H);
	int H_size = Ham_vecs.H_size;
	int H_col_size = Ham_vecs.H_col_size;

	// Copy to device memory:
	thrust::device_vector<OperatorType> H_decode_table_dev(H_size * H_col_size);
	thrust::device_vector<T> H_scalar_table_dev(H_size);
	thrust::device_vector<int> H_site_table_dev(H_size * H_col_size);
	thrust::device_vector<int> H_NOTerms_dev(H_size);

	thrust::copy(Ham_vecs.H_decode_table.begin(), Ham_vecs.H_decode_table.end(), H_decode_table_dev.begin());
	thrust::copy(Ham_vecs.H_scalar_table.begin(), Ham_vecs.H_scalar_table.end(), H_scalar_table_dev.begin());
	thrust::copy(Ham_vecs.H_site_table.begin(), Ham_vecs.H_site_table.end(), H_site_table_dev.begin());
	thrust::copy(Ham_vecs.H_NOTerms.begin(), Ham_vecs.H_NOTerms.end(), H_NOTerms_dev.begin());

	printf("Allocate the arrays needed to represent the Hamiltonian\n");
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
		BuildHamiltonianXXZForAbelianGroup << <blocks_GS, threads_GS >> > (
			H_scalar_table_dev.data().get(),
			H_decode_table_dev.data().get(),
			H_site_table_dev.data().get(),
			H_NOTerms_dev.data().get(),
			Ham_vecs.H_size,
			Ham_vecs.H_col_size,
			GS_fill,
			bas_states_dev.data().get(),
			index_mat_dev.data().get(),
			vals_mat_dev.data().get(),
			track_non_zero_inds_dev.data().get(),
			max_terms,
			SRS_unique_count,
			SRS_states.data().get(),
			GSize,
			orbit_indices.data().get(),
			char_mat_dev.data().get(),
			alpha,
			norm_vecs.data().get(),
			NIr,
			tol_norm
			);
		cudaDeviceSynchronize();
		auto end_time_CUDA_create_Ham3 = std::chrono::high_resolution_clock::now();
		auto duration_time_CUDA3 = std::chrono::duration_cast<std::chrono::milliseconds> (end_time_CUDA_create_Ham3 - start_time_CUDA_create_Ham3);
		std::cout << "\n Elapsed time of creating the Hamiltonian: " << duration_time_CUDA3.count() << " ms" << std::endl;

		int max_term_real = *(thrust::max_element(track_non_zero_inds_dev.begin(), track_non_zero_inds_dev.end()));
		printf("max term per row is: %d and the max_terms parameter is %d\n", max_term_real, max_terms);

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

template HeisenbergXXZAbelianSymms_CUDA<float>::HeisenbergXXZAbelianSymms_CUDA(
	Heisenberg<float> _lattice_model,
	SymmetryClass<float>* SymmGroup_in,
	int hop_f_in=-1,
	int GS_sector_in=-1,
	int NStates_in = 1,
	long long seed = 2212456,
	const int NIter_in = 50,
	float tol_norm_in = 0.00001
);

template HeisenbergXXZAbelianSymms_CUDA<complex_th>::HeisenbergXXZAbelianSymms_CUDA(
	Heisenberg<complex_th> _lattice_model,
	SymmetryClass<complex_th>* SymmGroup_in,
	int hop_f_in=-1,
	int GS_sector_in=-1,
	int NStates_in = 1,
	long long seed = 2212456,
	const int NIter_in = 50,
	float tol_norm_in = 0.00001
);


template <typename T>
void HeisenbergXXZAbelianSymms_CUDA<T>::BuildBasisStates(
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
template void HeisenbergXXZAbelianSymms_CUDA<float>::BuildBasisStates(
	thrust::device_vector<uint32_t>& bas_states_dev,
	unsigned long long int nobv
);
template void HeisenbergXXZAbelianSymms_CUDA<complex_th>::BuildBasisStates(
	thrust::device_vector<uint32_t>& bas_states_dev,
	unsigned long long int nobv
);

template <typename T>
OperatorVectors<T> HeisenbergXXZAbelianSymms_CUDA<T>::BuildOperatorVecs(ManyBodyOperator<T> H)
{
	
	int H_size = H.GetElems().size();
	int H_col_size = H.GetMaxTerms();
	thrust::host_vector<OperatorType> H_decode_table(H_size * H_col_size);
	thrust::host_vector<T> H_scalar_table(H_size, (T)1.0);
	thrust::host_vector<int> H_site_table(H_size * H_col_size, -1); // We initialize this to -1 as -1 indicates dummy elements
	thrust::host_vector<int> H_NOTerms(H_size, 0); // It is convenient to have this

	for (int hi = 0; hi < H_size; hi++) {
		H_NOTerms[hi] = H.GetElems()[hi].size();
		for (int hi2 = 0; hi2 < H.GetElems()[hi].size(); hi2++) {
			H_decode_table[hi * H_col_size + hi2] = GetConjugateOperatorType(H.GetElems()[hi][hi2].GetType()); //Needs to be conjugated due to the operator order
			H_site_table[hi * H_col_size + hi2] = H.GetElems()[hi][hi2].GetSite();
			H_scalar_table[hi] = H_scalar_table[hi] * (GetComplexConjugateHost(H.GetElems()[hi][hi2].GetScalar()));
			//std::cout << PrintOperatorType(H_decode_table[hi * H_col_size + hi2]);
		}
		//std::cout << H_scalar_table[hi] << std::endl;
	}

	OperatorVectors<T> Ham_vecs;
	Ham_vecs.H_size = H_size;
	Ham_vecs.H_col_size = H_col_size;
	Ham_vecs.H_decode_table = H_decode_table;
	Ham_vecs.H_scalar_table = H_scalar_table;
	Ham_vecs.H_site_table = H_site_table;
	Ham_vecs.H_NOTerms = H_NOTerms;
	return Ham_vecs;
}
template OperatorVectors<float> HeisenbergXXZAbelianSymms_CUDA<float>::BuildOperatorVecs(ManyBodyOperator<float> H);
template OperatorVectors<complex_th> HeisenbergXXZAbelianSymms_CUDA<complex_th>::BuildOperatorVecs(ManyBodyOperator<complex_th> H);

template <typename T>
int HeisenbergXXZAbelianSymms_CUDA<T>::BuildSRStates(SRSBuildInfo build_info,
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
	auto SRS_p = thrust::unique(SRS_states_min.begin(), SRS_states_min.end());

	// One has to also actually form the orbits from the generators of the orbits
	// This could be done in a smart way by permuting in an approriate way the elements of SRS_states. Here we use a simpler brute-force way
	// by using SetUpSRS
	printf("Setting up the final orbits\n");
	blocks_GS = (SRS_unique_count * GSize + threads_GS - 1) / threads_GS;
	SetUpSRS_from_indices << < blocks_GS, threads_GS >> > (SRS_states_min.data().get(), SRS_states.data().get(), group_el_dev.data().get(), SRS_unique_count, GSize, LS, GS_fill, bas_states_dev.data().get());
	cudaDeviceSynchronize();
	// NOTE THAT SRS STATES ARE LABELLED BY THEIR INDEX IN THE CORRESPONDING HILBERT SPACE SECTOR.

	// Now first SRS_unique_count rows of SRS_states should hold all the required orbits
	// Next, We need to normalize these orbits:
	printf("Compute the norm of the SRSs for each irrep\n");
	/* Remember that SRS_unique_count is now the number of orbits. */
	norm_vecs.resize(SRS_unique_count * NIr);
	blocks_GS = (SRS_unique_count * NIr + threads_GS - 1) / threads_GS;
	ComputeSRSNorms << <blocks_GS, threads_GS >> > (norm_vecs.data().get(), SRS_states.data().get(), char_mat_dev.data().get(), SRS_unique_count, NIr, GSize, bas_states_dev.data().get());
	cudaDeviceSynchronize();

	// Finally, one needs to have mapping from the bas_vecs to the orbit and the corresponding order index within the orbit:
	printf("Compute the mapping from the Hilbert space to the orbits\n");

	blocks_GS = (SRS_unique_count * GSize + threads_GS - 1) / threads_GS;
	ComputeOrbitIndices << <blocks_GS, threads_GS >> > (orbit_indices.data().get(), SRS_unique_count, GSize, SRS_states.data().get(), GS_fill);
	cudaDeviceSynchronize();

	//printDeviceMatrix<uint32_t>(orbit_indices, nobv_Sz0_cuda, 2,"OrbitIndices");
	//thrust::copy(orbit_indices.begin(), orbit_indices.end(), std::ostream_iterator<T>(std::cout, ", "));
	return SRS_unique_count;
}

template int HeisenbergXXZAbelianSymms_CUDA<float>::BuildSRStates(SRSBuildInfo build_info,
	thrust::device_vector<uint32_t>& bas_states_dev,
	thrust::device_vector<uint32_t>& SRS_states,
	thrust::device_vector<uint32_t>& SRS_states_min,
	thrust::device_vector<uint32_t>& group_el_dev,
	thrust::device_vector<float>& norm_vecs,
	thrust::device_vector<uint32_t>& orbit_indices,
	thrust::device_vector<float>& char_mat_dev);

template int HeisenbergXXZAbelianSymms_CUDA<complex_th>::BuildSRStates(SRSBuildInfo build_info,
	thrust::device_vector<uint32_t>& bas_states_dev,
	thrust::device_vector<uint32_t>& SRS_states,
	thrust::device_vector<uint32_t>& SRS_states_min,
	thrust::device_vector<uint32_t>& group_el_dev,
	thrust::device_vector<float>& norm_vecs,
	thrust::device_vector<uint32_t>& orbit_indices,
	thrust::device_vector<complex_th>& char_mat_dev);


template <typename T> complex_th HeisenbergXXZAbelianSymms_CUDA<T>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
)
{
	int threads_GS = 32;
	// The implementation of this function depends on the both the model and the method. Thus, we determine here which
	// single-particle operators are appropriate and how they map to the CUDA functions. This implementation is bit cumbersome and
	// should be improved.

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
	if (max_terms == 0) max_terms = max_coupling_terms;
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

	// Allocate CUDA operators:
	ManyBodyOperator<T> H = lattice_model.GetH();
	OperatorVectors<T> Ham_vecs = BuildOperatorVecs(H);
	int H_size = Ham_vecs.H_size;
	int H_col_size = Ham_vecs.H_col_size;

	// Copy to device memory:
	thrust::device_vector<OperatorType> H_decode_table_dev(H_size * H_col_size);
	thrust::device_vector<T> H_scalar_table_dev(H_size);
	thrust::device_vector<int> H_site_table_dev(H_size * H_col_size);
	thrust::device_vector<int> H_NOTerms_dev(H_size);

	thrust::copy(Ham_vecs.H_decode_table.begin(), Ham_vecs.H_decode_table.end(), H_decode_table_dev.begin());
	thrust::copy(Ham_vecs.H_scalar_table.begin(), Ham_vecs.H_scalar_table.end(), H_scalar_table_dev.begin());
	thrust::copy(Ham_vecs.H_site_table.begin(), Ham_vecs.H_site_table.end(), H_site_table_dev.begin());
	thrust::copy(Ham_vecs.H_NOTerms.begin(), Ham_vecs.H_NOTerms.end(), H_NOTerms_dev.begin());

	printf("Allocate the arrays needed to represnt the Hamiltonian\n");
	thrust::device_vector<uint32_t> index_mat_dev(SRS_unique_count * max_terms);
	thrust::device_vector<T> vals_mat_dev(SRS_unique_count * max_terms, (T)0.0);
	thrust::device_vector<short int> track_non_zero_inds_dev(SRS_unique_count, 0);

	uint32_t* ind_mat_ptr = thrust::raw_pointer_cast(&index_mat_dev[0]);
	T* val_mat_ptr = thrust::raw_pointer_cast(&vals_mat_dev[0]);
	short int* track_ptr = thrust::raw_pointer_cast(&track_non_zero_inds_dev[0]);

	int blocks_GS = (SRS_unique_count + threads_GS - 1) / threads_GS;
	BuildHamiltonianXXZForAbelianGroup << <blocks_GS, threads_GS >> > (
		H_scalar_table_dev.data().get(),
		H_decode_table_dev.data().get(),
		H_site_table_dev.data().get(),
		H_NOTerms_dev.data().get(),
		Ham_vecs.H_size,
		Ham_vecs.H_col_size,
		GS_fill,
		bas_states_dev.data().get(),
		index_mat_dev.data().get(),
		vals_mat_dev.data().get(),
		track_non_zero_inds_dev.data().get(),
		max_terms,
		SRS_unique_count,
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

template complex_th HeisenbergXXZAbelianSymms_CUDA<float>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
);

template complex_th HeisenbergXXZAbelianSymms_CUDA<complex_th>::ComputeStaticExpValZeroT(
	ManyBodyOperator<complex_th> A,
	int max_terms
);
// max_terms = 0