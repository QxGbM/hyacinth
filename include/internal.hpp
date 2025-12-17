#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

struct complex_double2;
struct complex_float4;

namespace internal::Cholesky {

  void imax_f64(cudaStream_t stream, int32_t N, const double* X, int32_t incx, double* D, double* diag_piv);

  void imax_f32(cudaStream_t stream, int32_t N, const float* X, int32_t incx, float* D, float* diag_piv);

  void imax_f128_dd(cudaStream_t stream, int32_t N, const double2* X, int32_t incx, double2* D, double2* diag_piv);

  void imax_f128_qf(cudaStream_t stream, int32_t N, const float4* X, int32_t incx, float4* D, float4* diag_piv);

  void imax_f64_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, double* X);

  void imax_f32_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, float* X);

  void imax_f128_dd_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, double2* X);

  void imax_f128_qf_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, float4* X);

  void gemv_cublas_f64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, double* D);

  void gemv_cublas_f32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, float* D);

  void gemv_cublas_cf64(cudaStream_t stream, cublasHandle_t handle, double* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D);

  void gemv_cublas_cf32(cudaStream_t stream, cublasHandle_t handle, float* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D);

  void gemv_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D);

  void gemv_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D);

  void gemv_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D);

  void gemv_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D);

  void gemv_pp_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, double* A, int32_t lda, double* D);

  void gemv_pp_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, float* A, int32_t lda, float* D);

  void gemv_pp_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, double2* A, int32_t lda, double2* D);

  void gemv_pp_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, float4* A, int32_t lda, float4* D);

  void gemv_pp_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* sq, std::complex<double>* A, int32_t lda, double* D);

  void gemv_pp_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* sq, std::complex<float>* A, int32_t lda, float* D);

  void gemv_pp_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* sq, complex_double2* A, int32_t lda, double2* D);

  void gemv_pp_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* sq, complex_float4* A, int32_t lda, float4* D);

  void gemv_pp_nopiv_f64(cudaStream_t stream, int32_t M, int32_t N, double* sq, double* A, int32_t lda, double* D);

  void gemv_pp_nopiv_f32(cudaStream_t stream, int32_t M, int32_t N, float* sq, float* A, int32_t lda, float* D);

  void gemv_pp_nopiv_f128_dd(cudaStream_t stream, int32_t M, int32_t N, double2* sq, double2* A, int32_t lda, double2* D);

  void gemv_pp_nopiv_f128_qf(cudaStream_t stream, int32_t M, int32_t N, float4* sq, float4* A, int32_t lda, float4* D);

  void gemv_pp_nopiv_cf64(cudaStream_t stream, int32_t M, int32_t N, double* sq, std::complex<double>* A, int32_t lda, double* D);

  void gemv_pp_nopiv_cf32(cudaStream_t stream, int32_t M, int32_t N, float* sq, std::complex<float>* A, int32_t lda, float* D);

  void gemv_pp_nopiv_cf128_dd(cudaStream_t stream, int32_t M, int32_t N, double2* sq, complex_double2* A, int32_t lda, double2* D);

  void gemv_pp_nopiv_cf128_qf(cudaStream_t stream, int32_t M, int32_t N, float4* sq, complex_float4* A, int32_t lda, float4* D);

};

namespace internal::int8 {

  void vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, uint64_t* vec_expon);

  void vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, uint64_t* vec_expon);

  void vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint64_t* vec_expon, int32_t incv);

  void vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint64_t* vec_expon, int32_t incv);

  void vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, uint64_t* vec_expon, int32_t incv);

  void vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, uint64_t* vec_expon, int32_t incv);

  void quantize_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda);

  void quantize_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda);

  void quantize_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda);

  void quantize_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda);

  void accumulate_i32tensor(cudaStream_t stream, int64_t N, int32_t sft_lo, int32_t orderX, int32_t alpha, const int32_t* X, int32_t orderA, int32_t beta, uint64_t* A);

  void accumulate_i32tensor_sft2x(cudaStream_t stream, int64_t N, int32_t orderX, const int32_t* X, int32_t orderA, int32_t beta, uint64_t* A);

  void dequantize_f64(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, double* B, int32_t ldb);

  void dequantize_f32(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, float* B, int32_t ldb);

  void dequantize_f128_dd(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, double2* B, int32_t ldb);

  void dequantize_f128_qf(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, float4* B, int32_t ldb);

  void dequantize_cf64(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, std::complex<double>* B, int32_t ldb);

  void dequantize_cf32(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, std::complex<float>* B, int32_t ldb);

  void dequantize_cf128_dd(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, complex_double2* B, int32_t ldb);

  void dequantize_cf128_qf(cudaStream_t stream, int32_t orderA, int32_t M, int32_t N, const uint64_t* A, int32_t lda, int32_t umax, const uint64_t* vec_expon, int32_t incv, complex_float4* B, int32_t ldb);

};

namespace internal::InterpolativeDecomposition {

  void interp_pp_f64_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx);

  void interp_pp_f32_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx);

  void interp_pp_f128_dd_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx);

  void interp_pp_f128_qf_f64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* A, int32_t lda, const int32_t* ipiv, double* X, int32_t ldx);

  void interp_pp_f64_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx);

  void interp_pp_f32_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx);

  void interp_pp_f128_dd_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, double2* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx);

  void interp_pp_f128_qf_f32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, float4* A, int32_t lda, const int32_t* ipiv, float* X, int32_t ldx);

  void interp_pp_cf64_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx);

  void interp_pp_cf32_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx);

  void interp_pp_cf128_dd_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx);

  void interp_pp_cf128_qf_cf64(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* A, int32_t lda, const int32_t* ipiv, std::complex<double>* X, int32_t ldx);

  void interp_pp_cf64_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx);

  void interp_pp_cf32_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx);

  void interp_pp_cf128_dd_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_double2* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx);

  void interp_pp_cf128_qf_cf32(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, complex_float4* A, int32_t lda, const int32_t* ipiv, std::complex<float>* X, int32_t ldx);

};

namespace internal::Orthogonalize {

  void qr_pp_f64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, double* A, int32_t lda, const int32_t* ipiv, double* tau, void** Workspace, int64_t* Lwork);

  void qr_pp_f32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, float* A, int32_t lda, const int32_t* ipiv, float* tau, void** Workspace, int64_t* Lwork);

  void qr_pp_cf64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* tau, void** Workspace, int64_t* Lwork);

  void qr_pp_cf32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* tau, void** Workspace, int64_t* Lwork);

};
