
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
enum class Precision { FP32, FP64, FP128_DD, FP128_QF };

namespace device {
  // triangluar copy :: copies M * i items from column i, only accept M = [1, 2]
  // rectangle copy :: no limits
  // conversion :: float <--> float,double; double <--> all; fp128 <--> double,fp128; [ no direct conversion float <--> fp128 ]

  void copy_upper_triangular(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB);

  void copy_rectangle(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB);

};

namespace device::Cholesky {
  // jpiv :: host page-locked, minimal length N + 8, 16 byte aligned
  // A :: device, minimal length lda * (N + 1), 16 byte aligned
  // epi :: Early termination, For [0., 1.] epi, it serves as relative error; For [1., N], it serves as fixed rank
  //        epi = 0., Will not terminate early, unless divided-by-0 occurs at diagonal

  // Return :: Number of iterations [0, N] = matrix rank
  //           The last interation is on the fly when function returns, will need synchronization to access

  int32_t rpotrfp(cudaStream_t stream, double epi, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv);

  int32_t cpotrfp(cudaStream_t stream, double epi, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv);

};

namespace device::QR {
  // jpiv :: host page-locked, minimal length N + 8, 16 byte aligned
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

  int32_t dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, double* A, int32_t lda, int32_t* jpiv, void* workspace);

  int32_t sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, float* A, int32_t lda, int32_t* jpiv, void* workspace);

  int32_t zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<double>* A, int32_t lda, int32_t* jpiv, void* workspace);

  int32_t cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, geqp3_params params, std::complex<float>* A, int32_t lda, int32_t* jpiv, void* workspace);

};

namespace device::ID {
  
};
