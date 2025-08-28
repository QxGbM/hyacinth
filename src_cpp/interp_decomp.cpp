
#include <hyacinth.hpp>
#include <internal.hpp>
#include <cuComplex.h>

template <device::Precision precA>
inline void cublas_trsm_real(cublasHandle_t handle, int32_t M, int32_t N, void* A, int32_t lda) {
  if (M < N) {
    if constexpr(precA == device::Precision::FP64) {
      double* R = (double*)A;
      double* R2 = &R[M * lda], one = 1.;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, R, lda, R2, lda);
    }
    else if constexpr(precA == device::Precision::FP32) {
      float* R = (float*)A;
      float* R2 = &R[M * lda], one = 1.f;
      cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, R, lda, R2, lda);
    }
  }
}

template <device::Precision precX>
inline void interp_pp_real(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, void* A, int32_t lda, const int32_t* ipiv, device::Precision precA, void* X, int32_t ldx, void* work) {
  if (precA == precX) {
    cublas_trsm_real<precX>(handle, M, N, A, lda);
    strided_identity(stream, M, M, M + 1, A, lda, precA);
    copy_permute(stream, 0, M, N, ipiv, A, lda, X, ldx, precA);
  }
  else if (precA == device::Precision::FP32) {
    cublas_trsm_real<device::Precision::FP32>(handle, M, N, A, lda);
    strided_identity(stream, M, M, M + 1, A, lda, device::Precision::FP32);
    copy_permute(stream, 0, M, N, ipiv, A, lda, work, lda, device::Precision::FP32);
    convert_and_copy(stream, M, N, work, lda, device::Precision::FP32, X, ldx, precX);
  }
  else {
    convert_and_copy(stream, M, N, A, lda, precA, work, lda, precX);
    cublas_trsm_real<precX>(handle, M, N, work, lda);
    strided_identity(stream, M, M, M + 1, work, lda, precX);
    copy_permute(stream, 0, M, N, ipiv, work, lda, X, ldx, precX);
  }
}

int32_t device::interp_decomp_f64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  
  MixPrecAHA::gemm_params param;
  MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  rank = Cholesky::rpotrfp(stream, handle, epi, 0, rank, N, work, algnN, param.precC, dpiv);
  if (0 < rank)
    interp_pp_real<Precision::FP64>(stream, handle, rank, N, work, algnN, dpiv, param.precC, X, ldx, &((int8_t*)work)[param.acc_bytes]);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_f32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx) {

  MixPrecAHA::gemm_params param;
  MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP32);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  rank = Cholesky::rpotrfp(stream, handle, epi, 0, rank, N, work, algnN, param.precC, dpiv);
  if (0 < rank)
    interp_pp_real<Precision::FP32>(stream, handle, rank, N, work, algnN, dpiv, param.precC, X, ldx, &((int8_t*)work)[param.acc_bytes]);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

template <device::Precision precA>
inline void cublas_trsm_complex(cublasHandle_t handle, int32_t M, int32_t N, void* A, int32_t lda) {
  if (M < N) {
    if constexpr(precA == device::Precision::FP64) {
      cuDoubleComplex* R = (cuDoubleComplex*)A;
      cuDoubleComplex* R2 = &R[M * lda];
      std::complex<double> one(1., 0.);
      cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuDoubleComplex*)&one, R, lda, R2, lda);
    }
    else if constexpr(precA == device::Precision::FP32) {
      cuComplex* R = (cuComplex*)A;
      cuComplex* R2 = &R[M * lda];
      std::complex<float> one(1.f, 0.f);
      cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuComplex*)&one, R, lda, R2, lda);
    }
  }
}

template <device::Precision precX>
inline void interp_pp_complex(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, void* A, int32_t lda, const int32_t* ipiv, device::Precision precA, void* X, int32_t ldx, void* work) {
  if (precA == precX) {
    cublas_trsm_complex<precX>(handle, M, N, A, lda);
    strided_identity(stream, 2 * M, M, 2 * M + 2, A, 2 * lda, precA);
    copy_permute(stream, 0, 2 * M, N, ipiv, A, 2 * lda, X, 2 * ldx, precA);
  }
  else if (precA == device::Precision::FP32) {
    cublas_trsm_complex<device::Precision::FP32>(handle, M, N, A, lda);
    strided_identity(stream, 2 * M, M, 2 * M + 2, A, 2 * lda, device::Precision::FP32);
    copy_permute(stream, 0, 2 * M, N, ipiv, A, 2 * lda, work, 2 * lda, device::Precision::FP32);
    convert_and_copy(stream, 2 * M, N, work, 2 * lda, device::Precision::FP32, X, 2 * ldx, precX);
  }
  else {
    convert_and_copy(stream, 2 * M, N, A, 2 * lda, precA, work, 2 * lda, precX);
    cublas_trsm_complex<precX>(handle, M, N, work, lda);
    strided_identity(stream, 2 * M, M, 2 * M + 2, work, 2 * lda, precX);
    copy_permute(stream, 0, 2 * M, N, ipiv, work, 2 * lda, X, 2 * ldx, precX);
  }
}

int32_t device::interp_decomp_cf64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx) {

  MixPrecAHA::gemm_params param;
  MixPrecAHA::cAHA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  MixPrecAHA::cAHA(stream, handle, param, A, lda, work);
  rank = Cholesky::cpotrfp(stream, handle, epi, 0, rank, N, work, algnN, param.precC, dpiv);
  if (0 < rank)
    interp_pp_complex<Precision::FP64>(stream, handle, rank, N, work, algnN, dpiv, param.precC, X, ldx, &((int8_t*)work)[param.acc_bytes]);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_cf32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx) {

  MixPrecAHA::gemm_params param;
  MixPrecAHA::cAHA_params_query(&param, epi, M, N, Precision::FP32);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  MixPrecAHA::cAHA(stream, handle, param, A, lda, work);
  rank = Cholesky::cpotrfp(stream, handle, epi, 0, rank, N, work, algnN, param.precC, dpiv);
  if (0 < rank)
    interp_pp_complex<Precision::FP32>(stream, handle, rank, N, work, algnN, dpiv, param.precC, X, ldx, &((int8_t*)work)[param.acc_bytes]);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}
