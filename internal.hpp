#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

struct complex_double2;
struct complex_float4;

namespace internal::Cholesky {

  void imax_double(cudaStream_t stream, int32_t N, const double* A, double* X, double* C, int32_t ldc, int32_t* piv, double* rsq);

  void imax_float(cudaStream_t stream, int32_t N, const float* A, float* X, float* C, int32_t ldc, int32_t* piv, float* rsq);

  void imax_double2(cudaStream_t stream, int32_t N, const double2* A, double2* X, double2* C, int32_t ldc, int32_t* piv, double2* rsq);

  void imax_float4(cudaStream_t stream, int32_t N, const float4* A, float4* X, float4* C, int32_t ldc, int32_t* piv, float4* rsq);

  void imax_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, double* X, std::complex<double>* C, int32_t ldc, int32_t* piv, double* rsq);

  void imax_float_complex(cudaStream_t stream, int32_t N, const std::complex<float>* A, float* X, std::complex<float>* C, int32_t ldc, int32_t* piv, float* rsq);

  void imax_double2_complex(cudaStream_t stream, int32_t N, const complex_double2* A, double2* X, complex_double2* C, int32_t ldc, int32_t* piv, double2* rsq);

  void imax_float4_complex(cudaStream_t stream, int32_t N, const complex_float4* A, float4* X, complex_float4* C, int32_t ldc, int32_t* piv, float4* rsq);

  void minus_transAx_plusB_scale_double(cudaStream_t stream, const double scale, int32_t M, int32_t N, const double* A, int32_t lda, double* B);

  void minus_transAx_plusB_scale_float(cudaStream_t stream, const float scale, int32_t M, int32_t N, const float* A, int32_t lda, float* B);

  void minus_transAx_plusB_scale_double2(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* B);

  void minus_transAx_plusB_scale_float4(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* B);

  void minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B);

  void minus_adjAx_plusB_scale_float_complex(cudaStream_t stream, const float scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B);

  void minus_adjAx_plusB_scale_double2_complex(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B);

  void minus_adjAx_plusB_scale_float4_complex(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B);

  void swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda);

  void swap_cols_float(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float* A, int32_t lda);

  void swap_cols_double2(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double2* A, int32_t lda);

  void swap_cols_float4(cudaStream_t stream, int32_t i, int32_t j, int32_t N, float4* A, int32_t lda);

  void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda);

  void swap_cols_float_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<float>* A, int32_t lda);

  void swap_cols_double2_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_double2* A, int32_t lda);

  void swap_cols_float4_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, complex_float4* A, int32_t lda);

};

namespace internal::int8 {
  
  void vexp_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon);

  void vexp_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon);

  void encode_f64_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t strideI);

  void encode_f32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t strideI);

  void encode_cf64_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t strideI);

  void encode_cf32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t strideI);

  void normalize_i32(cudaStream_t stream, int32_t M, int32_t N, int32_t* A);

  void normalize_i32_set_high(cudaStream_t stream, int32_t M, int32_t N, int32_t* A);

  void r8i_TN_gemm_strided_AC(cudaStream_t stream, cublasHandle_t handle, int32_t order, int32_t algnM, int32_t algnN, int32_t algnK, const int8_t* AT, int32_t strideA, const int8_t* B, int32_t* C, int32_t strideC);

  void c8i_HN_gemm_strided_AC(cudaStream_t stream, cublasHandle_t handle, int32_t order, int32_t algnM, int32_t algnN, int32_t algnK, const int8_t* AH, int32_t strideA, const int8_t* B, int32_t* C, int32_t strideC);

  void decode_f64_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double* C, int32_t ldc);

  void decode_f32_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float* C, int32_t ldc);

  void decode_dd_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double2* C, int32_t ldc);

  void decode_qf_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float4* C, int32_t ldc);

  void decode_cf64_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, std::complex<double>* C, int32_t ldc);

  void decode_cf32_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, std::complex<float>* C, int32_t ldc);

  void decode_complex_dd_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, complex_double2* C, int32_t ldc);

  void decode_complex_qf_strided_i32(cudaStream_t stream, int32_t order, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, complex_float4* C, int32_t ldc);

};
