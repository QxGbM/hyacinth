
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>

int32_t align_up(
  int32_t ld,
  int32_t align
);

int32_t align_c_fp32(int32_t ld);
int32_t align_c_fp64(int32_t ld);
int32_t align_c_i8(int32_t ld);

void imax_double(cudaStream_t stream, int32_t N, double* X, int32_t* piv, double* rsq);

void minus_transAx_plusB_scale_double(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B, double* C);

void minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B, double* C);

void copy_col_to_row_double(cudaStream_t stream, int32_t i, int32_t N, double* A, int32_t lda);

void copy_col_to_row_double_complex(cudaStream_t stream, int32_t i, int32_t N, std::complex<double>* A, int32_t lda);

void swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda);

void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda);

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv);


