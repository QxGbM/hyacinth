
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <float4.hpp>

int32_t align_up(
  int32_t ld,
  int32_t align
);

int32_t align_c_fp32(int32_t ld);
int32_t align_c_fp64(int32_t ld);
int32_t align_c_i8(int32_t ld);

void imax_double(cudaStream_t stream, int32_t N, double* X, int32_t* piv, double* rsq);

void imax_update_double(cudaStream_t stream, int32_t N, const double* A, double* X, int32_t* piv, double* rsq);

void imax_update_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, double* X, int32_t* piv, double* rsq);

void minus_transAx_plusB_scale_double(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B);

void minus_transAx_plusB_scale_float(cudaStream_t stream, const float* scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B);

void minus_transAx_plusB_scale_float4(cudaStream_t stream, const float4* scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B);

void minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B);

void minus_adjAx_plusB_scale_float_complex(cudaStream_t stream, const float* scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B);

void minus_adjAx_plusB_scale_float4_complex(cudaStream_t stream, const float4* scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B);

void swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda);

void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda);

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv);


