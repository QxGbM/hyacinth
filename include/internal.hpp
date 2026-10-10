#pragma once

#include <cstdint>
#include <cublas_v2.h>
#include <cuda_fp16.h>

struct complex_double2;
struct complex_float3;
struct complex_float4;

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, complex_float3* A, int32_t lda, int32_t* jpiv, float3* dev_work, int32_t* r);
  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t K, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* r);

};

namespace internal::int8 {

  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);
  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);
  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);
  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);
  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);
  void vector_range(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* u, const int32_t* vexp, int32_t* vbuf);

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum);

  void accumulate_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t sft, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void accumulate_remainder_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong2* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B);
  void triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong4_32a* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B);

};

namespace internal {

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, double* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, float* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const double2* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float3* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const float4* A, int32_t lda, __half* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb, const int32_t* jpiv = nullptr);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const __half2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float3* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb, const int32_t* jpiv = nullptr);

  std::pair<int32_t, int32_t> gram_algorithm(char& alg, int32_t M, int32_t& u, int32_t Complex);

};

namespace Timer {
  void register_distribute_kernel(cudaStream_t stream, void* timer);
  void register_replicate_kernel(cudaStream_t stream, void* timer);
  void register_comm(cudaStream_t stream, void* timer);
};
