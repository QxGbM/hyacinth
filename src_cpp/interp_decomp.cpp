
#include <hyacinth.hpp>
#include <internal.hpp>
#include <cuComplex.h>

int32_t device::interp_decomp_f64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  
  device::MixPrecAHA::gemm_params param;
  device::MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr, algnN = param.algnN;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 8) * sizeof(int32_t));

  device::MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  rank = device::Cholesky::rpotrfp(stream, epi, rank, N, work, algnN, param.precC, dpiv);
  double* R = (double*)(&((int8_t*)work)[param.acc_bytes]);
  copy_upper_triangular(stream, 1, N, work, algnN, param.precC, R, algnN, Precision::FP64);

  if (0 < rank) {
    if (rank < N) {
      double* R2 = &R[rank * algnN], one = 1.;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, rank, N - rank, &one, R, algnN, R2, algnN);
    }
    strided_identity(stream, rank, rank, rank + 1, R, algnN, Precision::FP64);
    copy_permute(stream, rank, N, dpiv, R, algnN, X, ldx, Precision::FP64);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_f32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx) {

  return rank;
}

int32_t device::interp_decomp_cf64(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx) {

  return rank;
}

int32_t device::interp_decomp_cf32(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t rank,
  int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx) {

  return rank;
}
