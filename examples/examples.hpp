
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace device {

  int32_t dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau);

  int32_t sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau);

  int32_t zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau);

  int32_t cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau);

  int32_t interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx);

  int32_t interp_decomp_f32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx);

  int32_t interp_decomp_cf64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx);

  int32_t interp_decomp_cf32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx);

};
