/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

#include <cuComplex.h>
#include <cuda_runtime_api.h>
#include <cublas_api.h>
#include <cusolverDn.h>
#include <library_types.h>



// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)


// type traits
template <typename T> struct traits;

template <> struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr cudaDataType cuda_data_type = CUDA_R_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_32F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr cudaDataType cuda_data_type = CUDA_R_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_R_64F;
#endif

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<cuFloatComplex> {
    // scalar type
    typedef float S;
    typedef cuFloatComplex T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr cudaDataType cuda_data_type = CUDA_C_32F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_32F;
#endif

    inline static S abs(T val) { return cuCabsf(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuFloatComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCaddf(a, b); }
    inline static T add(T a, S b) { return cuCaddf(a, make_cuFloatComplex(b, 0.f)); }

    inline static T mul(T v, double f) { return make_cuFloatComplex(v.x * f, v.y * f); }
};

template <> struct traits<cuDoubleComplex> {
    // scalar type
    typedef double S;
    typedef cuDoubleComplex T;

    static constexpr T zero = {0., 0.};
    static constexpr cudaDataType cuda_data_type = CUDA_C_64F;
#if CUDART_VERSION >= 11000
    static constexpr cusolverPrecType_t cusolver_precision_type = CUSOLVER_C_64F;
#endif

    inline static S abs(T val) { return cuCabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return make_cuDoubleComplex((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return cuCadd(a, b); }
    inline static T add(T a, S b) { return cuCadd(a, make_cuDoubleComplex(b, 0.)); }

    inline static T mul(T v, double f) { return make_cuDoubleComplex(v.x * f, v.y * f); }
};




// Returns cusolverIRSRefinement_t value as defined in cusolver_common.h for the string containing
// solver name
cusolverIRSRefinement_t get_cusolver_refinement_solver(std::string solver_string) {
    if (solver_string.compare("CUSOLVER_IRS_REFINE_NONE") == 0)
        return CUSOLVER_IRS_REFINE_NONE;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_CLASSICAL_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_CLASSICAL_GMRES;
    else if (solver_string.compare("CUSOLVER_IRS_REFINE_GMRES_GMRES") == 0)
        return CUSOLVER_IRS_REFINE_GMRES_GMRES;
    else
        printf("Unknown solver parameter: \"%s\"\n", solver_string.c_str());

    return CUSOLVER_IRS_REFINE_NOT_SET;
}
