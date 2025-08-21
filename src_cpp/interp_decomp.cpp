
#include <hyacinth.hpp>
#include <internal.hpp>
#include <cuComplex.h>

int32_t device::interp_decomp_f64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  
  MixPrecAHA::gemm_params param;
  MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  rank = Cholesky::rpotrfp(stream, epi, rank, N, work, algnN, param.precC, dpiv);
  double* R = (double*)(&((int8_t*)work)[param.acc_bytes]);
  convert_and_copy(stream, rank, N, work, algnN, param.precC, R, algnN, Precision::FP64);

  if (0 < rank) {
    if (rank < N) {
      double* R2 = &R[rank * algnN], one = 1.;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, &one, R, algnN, R2, algnN);
    }
    strided_identity(stream, rank, rank, rank + 1, R, algnN, Precision::FP64);
    copy_permute(stream, 0, rank, N, dpiv, R, algnN, X, ldx, Precision::FP64);
  }

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
  rank = Cholesky::rpotrfp(stream, epi, rank, N, work, algnN, param.precC, dpiv);
  float* R = (float*)(&((int8_t*)work)[param.acc_bytes]);
  convert_and_copy(stream, rank, N, work, algnN, param.precC, R, algnN, Precision::FP32);

  if (0 < rank) {
    if (rank < N) {
      float* R2 = &R[rank * algnN], one = 1.f;
      cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, &one, R, algnN, R2, algnN);
    }
    strided_identity(stream, rank, rank, rank + 1, R, algnN, Precision::FP32);
    copy_permute(stream, 0, rank, N, dpiv, R, algnN, X, ldx, Precision::FP32);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
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
  rank = Cholesky::cpotrfp(stream, epi, rank, N, work, algnN, param.precC, dpiv);
  cuDoubleComplex* R = (cuDoubleComplex*)(&((int8_t*)work)[param.acc_bytes]);
  convert_and_copy(stream, 2 * rank, N, work, 2 * algnN, param.precC, R, 2 * algnN, Precision::FP64);

  if (0 < rank) {
    if (rank < N) {
      cuDoubleComplex* R2 = &R[rank * algnN];
      std::complex<double> one(1., 0.);
      cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, (cuDoubleComplex*)&one, R, algnN, R2, algnN);
    }
    strided_identity(stream, 2 * rank, rank, 2 * (rank + 1), R, 2 * algnN, Precision::FP64);
    copy_permute(stream, 0, 2 * rank, N, dpiv, R, 2 * algnN, X, 2 * ldx, Precision::FP64);
  }

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
  rank = Cholesky::cpotrfp(stream, epi, rank, N, work, algnN, param.precC, dpiv);
  cuComplex* R = (cuComplex*)(&((int8_t*)work)[param.acc_bytes]);
  convert_and_copy(stream, 2 * rank, N, work, 2 * algnN, param.precC, R, 2 * algnN, Precision::FP32);

  if (0 < rank) {
    if (rank < N) {
      cuComplex* R2 = &R[rank * algnN];
      std::complex<float> one(1., 0.);
      cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, (cuComplex*)&one, R, algnN, R2, algnN);
    }
    strided_identity(stream, 2 * rank, rank, 2 * (rank + 1), R, 2 * algnN, Precision::FP32);
    copy_permute(stream, 0, 2 * rank, N, dpiv, R, 2 * algnN, X, 2 * ldx, Precision::FP32);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}
