
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace device {
  namespace Config {
    // ------------ Changing Config requires recompile the library !! ------------
    // Exponent BASE :: ranged from [4, 7] for integer quantization

    constexpr int32_t exp_base = 7;
  };

  enum class Precision { FP32, FP64, FP128_DD, FP128_QF };

  // rectangle copy :: mode='U/u' will copy only upper triangular parts, 'U': two scalars=one elem (for complex)
  //                   mode=other will copy the entire matrix
  // conversion :: [all] to [all] is okay

  void copy_signs(cudaStream_t stream, int32_t N, const void* X, int32_t incx, Precision precX, void* signs, int32_t sign_bytes);

  void convert_and_copy(cudaStream_t stream, char mode, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB, const int8_t* signs);

  void copy_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec);

  void copy_scatter(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec);

  void strided_identity(cudaStream_t stream, int32_t M, int32_t N, int32_t strideD, void* A, int32_t lda, Precision prec);

  // Mixed precision geqp3, synchronous call, includes memory alloc/dealloc
  // Will always attempt to create the complete R unless rank deficient
  // A :: device, minimal length lda * N, M <= lda
  // jpiv :: host/device, minimal length N

  int32_t dgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau);

  int32_t sgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau);

  int32_t zgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau);

  int32_t cgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau);

  // Mixed precision interpolative decomposition, synchronous call, includes memory alloc/dealloc
  // Outputs A ~= A[:jpiv(1, rank)] * X, rank of X as function return
  // A :: device, minimal length lda * N, M <= lda
  // jpiv :: host/device, minimal length N
  // X :: device, minimal length ldx * N, rank <= ldx

  int32_t interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx);

  int32_t interp_decomp_f32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx);

  int32_t interp_decomp_cf64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx);

  int32_t interp_decomp_cf32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx);

  double check_interp_decomp_f64(cublasHandle_t handle, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* jpiv, const double* X, int32_t ldx);

  double check_interp_decomp_f32(cublasHandle_t handle, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* jpiv, const float* X, int32_t ldx);

  double check_interp_decomp_cf64(cublasHandle_t handle, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* jpiv, const std::complex<double>* X, int32_t ldx);

  double check_interp_decomp_cf32(cublasHandle_t handle, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* jpiv, const std::complex<float>* X, int32_t ldx);

  namespace Cholesky {
    // jpiv :: host minimal length N
    // A :: device, minimal length lda * (N + 1), 16 byte aligned
    // epi :: Early termination by relative error, truncated to [0., 1.];
    // [start,end] :: control indices for partial factorization;
    // epi = 0. && end = N, Will not terminate early, unless divided-by-0 occurs at diagonal
    // pinned_work :: host page-locked memory, minimal length is 8kb (8192 bytes), used in hybrid reduction for pivoting

    // Return :: Number of iterations [0, N] = matrix rank
    //           The last interation is on the fly when function returns, will need synchronization to access

    void rpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work);

    void cpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work);

  };

  namespace MixPrecAHA {
    // A :: device, minimal length lda * N
    // C :: device, queried in param, 256 byte aligned

    // Output :: C = A^H * A, Stored in the leading positions
    //           C is [param.N] by [param.N] with stride [param.algnN]
    //           Precision used is [param.precC], with [param.C_elem_bytes] sized elements

    // Query :: params [out] is written with specified dimensions and accuracy
    //       :: epi [in/out] contain the accuracy used to compute at output

    void mpgemm_params(double* epi, int32_t M, int32_t N, int32_t* algnM, int32_t* algnN, int32_t* orderA, Precision precA, Precision* precC);

    void rATA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnM, int32_t algnN, int32_t orderA, const void* A, int32_t lda, Precision precA, void* C, Precision precC);

    void cAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnM, int32_t algnN, int32_t orderA, const void* A, int32_t lda, Precision precA, void* C, Precision precC);

  };
};
