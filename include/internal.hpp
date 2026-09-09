#pragma once

#include <cstdint>
#include <tuple>
#include <map>
#include <vector>
#include <string>
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

};

namespace internal::int8 {

  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* umax, int32_t* vexp);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* umax, int32_t* vexp);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t* umax, int32_t* vexp);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t* umax, int32_t* vexp);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t* umax, int32_t* vexp);
  void vector_exponents(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t* umax, int32_t* vexp);

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);
  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb);

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum);
  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum);

  void accumulate_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t sft, int32_t sft_iter, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void accumulate_remainder_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A);
  void triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong2* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B);
  void triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong4_32a* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B);
  int32_t gram_algorithm(char& alg, int32_t M, int32_t& u);

};

namespace internal {

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, double* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, double* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, float* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, float* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const double2* A, int32_t lda, __half* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const float4* A, int32_t lda, __half* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuDoubleComplex* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, cuComplex* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, cuComplex* B, int32_t ldb);

  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuDoubleComplex* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const cuComplex* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const __half2* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_double2* A, int32_t lda, __half2* B, int32_t ldb);
  void scatter_matcopy(cudaStream_t stream, cublasHandle_t handle, char mode, int32_t M, int32_t N, const int32_t* jpiv, const complex_float4* A, int32_t lda, __half2* B, int32_t ldb);

  int32_t device_is_f64_capable();
  int32_t device_num_sms();

};

namespace Timer {
  void register_kernel(cudaStream_t stream, void* timer);
  void register_comm(cudaStream_t stream, void* timer);
};

namespace Batch {

  class BatchArgs {
  private:
    std::map<int32_t, std::tuple<int32_t, int32_t, int32_t, int32_t, char>> tensor;
    // tuple is [Tensor Order, Tensor Panel Prefix, Tensor ColSums Prefix, Tensor Rows, Algorithm 'L' or 'C']

  public:
    const int32_t batchMaxK;
    BatchArgs(int32_t K, const std::string& str, int32_t& order, int32_t& count_crt);

    // op: 'E'=eager eval; 'Z'=lazy batch; 'F'=lazy+flush; 'S' = skip;
    std::tuple<int32_t, int32_t, int64_t, int64_t, int32_t, char> processA(int32_t M, int32_t N, int32_t uc, char alg, char& op);
    void flush(int32_t N, std::vector<std::tuple<int32_t, int32_t, int64_t, int64_t, int32_t, char>>& list);
  };

};
