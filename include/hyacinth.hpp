
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

  // Mixed precision geqp3, synchronous call, includes memory alloc/dealloc
  // A :: device, minimal length lda * N
  // jpiv :: host/device, minimal length N

  int32_t dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv);

  int32_t sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv);

  int32_t zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv);

  int32_t cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv);

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

namespace device::MixPrecAHA {
  // A :: device, minimal length lda * N
  // C :: device, queried in param, 256 byte aligned

  // Output :: C = A^H * A, Stored in the leading positions
  //           C is [param.N] by [param.N] with stride [param.algnN]
  //           Precision used is [param.precC], with [param.C_elem_bytes] sized elements

  struct gemm_params {
    Precision precA;
    int32_t M;
    int32_t N;
    int32_t algnM;
    int32_t orderA;
    int32_t iter_k;

    Precision precC;
    int32_t algnN;
    int32_t C_elem_bytes;

    uint64_t acc_bytes;
    uint64_t i8_bytes;
    uint64_t exp_bytes;
    uint64_t scratch_bytes;
    uint64_t C_bytes;
  };

  void rATA_params_query(gemm_params* param, double epi, int32_t M, int32_t N, Precision precA);

  void cAHA_params_query(gemm_params* param, double epi, int32_t M, int32_t N, Precision precA);

  void rATA(cudaStream_t stream, cublasHandle_t handle, gemm_params param, const void* A, int32_t lda, void* C);

  void cAHA(cudaStream_t stream, cublasHandle_t handle, gemm_params param, const void* A, int32_t lda, void* C);

};
