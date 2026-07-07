#pragma once

#include <cstdint>
#include <complex>
#include <cublas_v2.h>
#include <cuda_fp16.h>

struct complex_double2;
struct complex_float4;
struct double_idx;
struct float_idx;
struct double2_idx;
struct float4_idx;

namespace internal::Cholesky {

  void imax_f64(cudaStream_t stream, double epi, int32_t N, const double* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale);

  void imax_f32(cudaStream_t stream, double epi, int32_t N, const float* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale);

  void imax_f128_dd(cudaStream_t stream, double epi, int32_t N, const double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale);

  void imax_f128_qf(cudaStream_t stream, double epi, int32_t N, const float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale);

  void imax_cf64(cudaStream_t stream, double epi, int32_t N, const std::complex<double>* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale);

  void imax_cf32(cudaStream_t stream, double epi, int32_t N, const std::complex<float>* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale);

  void imax_cf128_dd(cudaStream_t stream, double epi, int32_t N, const complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale);

  void imax_cf128_qf(cudaStream_t stream, double epi, int32_t N, const complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale);

  void gemv_cublas_f64(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D);

  void gemv_cublas_f32(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D);

  void gemv_cublas_cf64(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, double* D);

  void gemv_cublas_cf32(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, float* D);

  void gemv_scal_f128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D);

  void gemv_scal_f128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D);

  void gemv_scal_cf128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D);

  void gemv_scal_cf128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D);

  void gemv_pp_f64(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D);

  void gemv_pp_f32(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D);

  void gemv_pp_f128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D);

  void gemv_pp_f128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D);

  void gemv_pp_cf64(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, double* D);

  void gemv_pp_cf32(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, float* D);

  void gemv_pp_cf128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D);

  void gemv_pp_cf128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D);

  int32_t potrfp_f64(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_f32(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_f128_dd(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_f128_qf(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_cf64(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_cf32(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_cf128_dd(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

  int32_t potrfp_cf128_qf(cudaStream_t stream, cublasHandle_t handle, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, void* dev_work, void* pinned_work);

};

namespace internal::int8 {

  void vexp_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* vec_expon);

  void vexp_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* vec_expon);

  void vexp_f16(cudaStream_t stream, int32_t M, int32_t N, const half* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* vec_expon);

  void vexp_cf16(cudaStream_t stream, int32_t M, int32_t N, const std::complex<half>* A, int32_t lda, int32_t* vec_expon);

  void vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void vsum_f16(cudaStream_t stream, int32_t M, int32_t N, const half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void vsum_cf16(cudaStream_t stream, int32_t M, int32_t N, const std::complex<half>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv);

  void quantize_f64(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_f32(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_f16(cudaStream_t stream, int32_t M, int32_t N, const half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf16(cudaStream_t stream, int32_t M, int32_t N, const std::complex<half>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);
  
  void quantize_f64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_f32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_f16_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<double>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<float>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void quantize_cf16_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<half>* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A);

  void dequantize_i63_f64(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double* B, int32_t ldb);

  void dequantize_i63_f32(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float* B, int32_t ldb);

  void dequantize_i63_f128_dd(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, double2* B, int32_t ldb);

  void dequantize_i63_f128_qf(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, float4* B, int32_t ldb);

  void dequantize_i63_cf64(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, std::complex<double>* B, int32_t ldb);

  void dequantize_i63_cf32(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, std::complex<float>* B, int32_t ldb);

  void dequantize_i63_cf128_dd(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_double2* B, int32_t ldb);

  void dequantize_i63_cf128_qf(cudaStream_t stream, int32_t bits, int32_t orderA, int32_t K, int32_t N, const uint64_t* A, int32_t umax, const int32_t* vec_expon, complex_float4* B, int32_t ldb);

  void accumulate_i32tensor(cudaStream_t stream, int32_t option, int32_t N, int32_t sft, uint32_t sft_iter, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);

  void accumulate_remainder_i32tensor(cudaStream_t stream, int32_t option, int32_t N, int32_t orderX, int32_t iter, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);

  void accumulate_conv_i63_u47(cudaStream_t stream, int32_t orderA, int32_t Complex, int64_t N, uint64_t* A);

  void i63ATA_f64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63ATA_f32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63ATA_f16_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf16_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<half>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63ATA_f32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63ATA_f16_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

  void i63AHA_cf16_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<half>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t algnM, int32_t algnN, int32_t orderA, int32_t orderC, uint64_t* C, int8_t* workspace);

};

namespace Timer {
  void register_kernel(cudaStream_t stream, void* timer);
  void register_comm(cudaStream_t stream, void* timer);
};
