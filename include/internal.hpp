#pragma once

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>

struct complex_double2;
struct complex_float4;
struct double_idx;
struct float_idx;
struct double2_idx;
struct float4_idx;

namespace internal::Cholesky {

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const double* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const float* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const cuDoubleComplex* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const cuComplex* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale);
  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale);

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D);
  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D);

  void gemv_pp(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D);
  void gemv_pp(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D);
  void gemv_pp(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D);
  void gemv_pp(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D);
  void gemv_pp(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D);
  void gemv_pp(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D);
  void gemv_pp(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D);
  void gemv_pp(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D);

  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, void* pinned_work);
  int32_t potrfp(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, void* pinned_work);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, double* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, float* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, __half* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb);

};

namespace internal::int8 {

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* vec_expon);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* vec_expon);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* vec_expon);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* vec_expon);

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);
  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);
  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);
  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);
  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);
  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb);
  void dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb);
  void dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb);
  void dequantize(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb);

  void dequantize_complex(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuDoubleComplex* B, int32_t ldb);
  void dequantize_complex(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, cuComplex* B, int32_t ldb);
  void dequantize_complex(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb);
  void dequantize_complex(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb);

  void accumulate_i32tensor(cudaStream_t stream, int32_t option, int32_t N, int32_t sft, uint32_t sft_iter, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void accumulate_remainder_i32tensor(cudaStream_t stream, int32_t option, int32_t N, int32_t orderX, int32_t iter, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void accumulate_conv_i63_u47(cudaStream_t stream, int32_t orderA, int32_t Complex, int64_t N, uint64_t* A);

  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);
  void i63AHA_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

};

namespace Timer {
  void register_kernel(cudaStream_t stream, void* timer);
  void register_comm(cudaStream_t stream, void* timer);
};
