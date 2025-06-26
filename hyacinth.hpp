
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

std::pair<double, int32_t> imax_double(cudaStream_t stream, int32_t N, const double* X);

std::pair<double, int32_t> imax_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* X);

void minus_transAx_plusB_scale_double(cudaStream_t stream, double scale, int32_t M, int32_t N, const double* A, int32_t lda, const double* X, double* B, double* C);

void minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, double scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const std::complex<double>* X, std::complex<double>* B, double* C);

void copy_col_to_row_double(cudaStream_t stream, int32_t i, int32_t N, double* A, int32_t lda);

void copy_col_to_row_double_complex(cudaStream_t stream, int32_t i, int32_t N, std::complex<double>* A, int32_t lda);

void swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda);

void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda);

void zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv);


