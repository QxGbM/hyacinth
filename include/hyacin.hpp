
#pragma once

#include <cstdint>
#include <complex>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

namespace device {

  enum class Precision { FP32, FP64, FP128_DD, FP128_QF, FP32_COMPLEX, FP64_COMPLEX, FP128_DD_COMPLEX, FP128_QF_COMPLEX };

  enum class Algorithm { Limbs, CRT };

  int32_t dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau);

  int32_t sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau);

  int32_t zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau);

  int32_t cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau);

  int32_t interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx);

  int32_t interp_decomp_f32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx);

  int32_t interp_decomp_cf64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx);

  int32_t interp_decomp_cf32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx);

  namespace Cholesky {

    void Xpotrfp(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t* iters, int32_t N, void* A, int32_t lda, Precision precA, int32_t* jpiv, void* pinned_work);

  };

  namespace MixPrecAHA {

    void igemm_params(double* epi, int32_t N, int32_t* algnN, int32_t* umax, Precision precA, Precision* precC, Algorithm* alg);

    void igemm_workspace(int32_t M, int32_t N, int32_t algnN, int32_t umax, Precision precC, Algorithm alg, int64_t* workspace);

    void iAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC, Algorithm alg);

  };

  namespace Utils {
    
    void convert_and_copy(cudaStream_t stream, int32_t M, int32_t N, const void* A, int32_t lda, Precision precA, void* B, int32_t ldb, Precision precB);

    void inplace_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, void* A, int32_t lda, void* workspace, int64_t Lwork, Precision prec);

    void copy_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec);

    void copy_scatter(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb, Precision prec);

    void strided_identity(cudaStream_t stream, int32_t M, int32_t N, void* A, int32_t lda, Precision prec);

    void workspace_realloc(cudaStream_t stream, void** ptr, int64_t* bytes_old, int64_t bytes_required);

  };

};
