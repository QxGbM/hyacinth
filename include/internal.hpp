#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

struct complex_double2;
struct complex_float4;

namespace internal::Cholesky {

  void imax_f64(cudaStream_t stream, int32_t N, double* X, double* C, int32_t ldc, int32_t* piv, double* rsq);

  void imax_f32(cudaStream_t stream, int32_t N, float* X, float* C, int32_t ldc, int32_t* piv, float* rsq);

  void imax_f128_dd(cudaStream_t stream, int32_t N, double2* X, double2* C, int32_t ldc, int32_t* piv, double2* rsq);

  void imax_f128_qf(cudaStream_t stream, int32_t N, float4* X, float4* C, int32_t ldc, int32_t* piv, float4* rsq);

  void imax_cf64(cudaStream_t stream, int32_t N, double* X, std::complex<double>* C, int32_t ldc, int32_t* piv, double* rsq);

  void imax_cf32(cudaStream_t stream, int32_t N, float* X, std::complex<float>* C, int32_t ldc, int32_t* piv, float* rsq);

  void imax_cf128_dd(cudaStream_t stream, int32_t N, double2* X, complex_double2* C, int32_t ldc, int32_t* piv, double2* rsq);

  void imax_cf128_qf(cudaStream_t stream, int32_t N, float4* X, complex_float4* C, int32_t ldc, int32_t* piv, float4* rsq);

  void gemv_scal_f64(cudaStream_t stream, double scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B, double* D);

  void gemv_scal_f32(cudaStream_t stream, float scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B, float* D);

  void gemv_scal_f128_dd(cudaStream_t stream, double2 scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* B, double2* D);

  void gemv_scal_f128_qf(cudaStream_t stream, float4 scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B, float4* D);

  void gemv_scal_cf64(cudaStream_t stream, double scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B, double* D);

  void gemv_scal_cf32(cudaStream_t stream, float scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B, float* D);

  void gemv_scal_cf128_dd(cudaStream_t stream, double2 scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B, double2* D);

  void gemv_scal_cf128_qf(cudaStream_t stream, float4 scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B, float4* D);

  void reduce_scal_f64(cudaStream_t stream, double scale, int32_t M, int32_t N, double* A, int32_t lda, double* D);

  void reduce_scal_f32(cudaStream_t stream, float scale, int32_t M, int32_t N, float* A, int32_t lda, float* D);

  void reduce_scal_f128_dd(cudaStream_t stream, double2 scale, int32_t M, int32_t N, double2* A, int32_t lda, double2* D);

  void reduce_scal_f128_qf(cudaStream_t stream, float4 scale, int32_t M, int32_t N, float4* A, int32_t lda, float4* D);

  void reduce_scal_cf64(cudaStream_t stream, const double scale, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D);

  void reduce_scal_cf32(cudaStream_t stream, const float scale, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D);

  void reduce_scal_cf128_dd(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D);

  void reduce_scal_cf128_qf(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D);

  void swap_cols_f64(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda);

  void swap_cols_f32(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float* A, int32_t lda);

  void swap_cols_f128_dd(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double2* A, int32_t lda);

  void swap_cols_f128_qf(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float4* A, int32_t lda);

  void swap_cols_cf64(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda);

  void swap_cols_cf32(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<float>* A, int32_t lda);

  void swap_cols_cf128_dd(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_double2* A, int32_t lda);

  void swap_cols_cf128_qf(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_float4* A, int32_t lda);

};

namespace internal::int8 {

  void vexp_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon);

  void vexp_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon);

  void encode_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda);

  void encode_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda);

  void encode_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda);

  void encode_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda);

  void r8i_TN_gemm_stridedA_f64(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, double* C, int32_t* workspace);

  void r8i_TN_gemm_stridedA_f32(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, float* C, int32_t* workspace);

  void r8i_TN_gemm_stridedA_f128_dd(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, double2* C, int32_t* workspace);

  void r8i_TN_gemm_stridedA_f128_qf(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, float4* C, int32_t* workspace);

  void i32_normalization(cudaStream_t stream, uint64_t M, int32_t order, int32_t beta, int32_t* A);

  void scal_exponent_f64(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon);

  void scal_exponent_f32(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon);

  void scal_exponent_f128_dd(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon);

  void scal_exponent_f128_qf(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon);

  void planar_to_interleave_f64(cudaStream_t stream, int32_t N, const double* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, std::complex<double>* B, int32_t ldb);

  void planar_to_interleave_f32(cudaStream_t stream, int32_t N, const float* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, std::complex<float>* B, int32_t ldb);

  void planar_to_interleave_f128_dd(cudaStream_t stream, int32_t N, const double2* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, complex_double2* B, int32_t ldb);

  void planar_to_interleave_f128_qf(cudaStream_t stream, int32_t N, const float4* A, int32_t lda, int32_t gemm_expon, const int32_t* vec_expon, complex_float4* B, int32_t ldb);

  void decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double* A, const int32_t* B, int32_t ld);

  void decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float* A, const int32_t* B, int32_t ld);

  void decode_f128_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double2* A, const int32_t* B, int32_t ld);

  void decode_f128_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float4* A, const int32_t* B, int32_t ld);

};
