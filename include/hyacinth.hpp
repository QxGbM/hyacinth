
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>

namespace device::Config {
  // ------------ Changing Config requires recompile the library !! ------------
  // Exponent BASE :: ranged from [4, 7] for integer quantization

#ifdef I8_EXP_BASE
  constexpr int32_t exp_base = I8_EXP_BASE;
#else
  constexpr int32_t exp_base = 6;
#endif
};

struct complex_double2;
struct complex_float4;

namespace device::Cholesky {
  // ipiv :: host page-locked, minimal length N + 8, 16 byte aligned
  // A :: device, minimal length lda * (N + 1), 16 byte aligned

  int32_t dpotrfp(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t* ipiv);

  int32_t spotrfp(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t* ipiv);

  int32_t double_double_potrfp(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int32_t* ipiv);

  int32_t quad_float_potrfp(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t* ipiv);

  int32_t zpotrfp(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv);

  int32_t cpotrfp(cudaStream_t stream, int32_t N, std::complex<float>* A, int32_t lda, int32_t* ipiv);

  int32_t complex_double_double_potrfp(cudaStream_t stream, int32_t N, complex_double2* A, int32_t lda, int32_t* ipiv);

  int32_t complex_quad_float_potrfp(cudaStream_t stream, int32_t N, complex_float4* A, int32_t lda, int32_t* ipiv);

};

namespace device::QR {
  // ipiv :: host page-locked, minimal length N + 8, 16 byte aligned
  // A :: device, minimal length lda * N, 16 byte aligned
  // work :: device, queried in params, 256 byte aligned

  struct geqp3_params {
    int32_t M, N, algnM, algnN, orderA, elem_bytes, iter_k, use_fp64_over_32;
    uint64_t n_elem, n_i8, v_exp, scratchpad, work_bytes;
  };

  void dgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N);

  void sgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N);

  void zgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N);

  void cgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N);

  void set_double_double_as_fp128(geqp3_params* params);

  void set_quad_float_as_fp128(geqp3_params* params);

  int32_t dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, double* A, int32_t lda, int32_t* ipiv, void* workspace);

  int32_t sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, float* A, int32_t lda, int32_t* ipiv, void* workspace);

  int32_t zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<double>* A, int32_t lda, int32_t* ipiv, void* workspace);

  int32_t cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<float>* A, int32_t lda, int32_t* ipiv, void* workspace);

};
