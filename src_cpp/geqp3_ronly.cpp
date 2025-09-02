
#include <hyacinth.hpp>
#include <internal.hpp>

int32_t device::dgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv) {
  MixPrecAHA::gemm_params param;
  MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  Cholesky::rpotrfp(stream, handle, 0., &iters, N, work, param.algnN, param.precC, dpiv);
  convert_and_copy(stream, N, N, work, param.algnN, param.precC, 0, A, lda, param.precA);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (1 + iters);
}

int32_t device::sgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv) {
  MixPrecAHA::gemm_params param;
  MixPrecAHA::rATA_params_query(&param, epi, M, N, Precision::FP32);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, handle, param, A, lda, work);
  Cholesky::rpotrfp(stream, handle, 0., &iters, N, work, param.algnN, param.precC, dpiv);
  convert_and_copy(stream, N, N, work, param.algnN, param.precC, 0, A, lda, param.precA);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (1 + iters);
}

int32_t device::zgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv) {
  MixPrecAHA::gemm_params param;
  MixPrecAHA::cAHA_params_query(&param, epi, M, N, Precision::FP64);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, handle, param, A, lda, work);
  Cholesky::cpotrfp(stream, handle, 0., &iters, N, work, param.algnN, param.precC, dpiv);
  convert_and_copy(stream, 2 * N, N, work, 2 * param.algnN, param.precC, 0, A, 2 * lda, param.precA);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (1 + iters);
}

int32_t device::cgeqp3_ronly(cudaStream_t stream, cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv) {
  MixPrecAHA::gemm_params param;
  MixPrecAHA::cAHA_params_query(&param, epi, M, N, Precision::FP32);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, param.C_bytes);
  cudaMallocHost((void**)(&dpiv), (N + 12) * sizeof(int32_t));

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, handle, param, A, lda, work);
  Cholesky::cpotrfp(stream, handle, 0., &iters, N, work, param.algnN, param.precC, dpiv);
  convert_and_copy(stream, 2 * N, N, work, 2 * param.algnN, param.precC, 0, A, 2 * lda, param.precA);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, dpiv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (1 + iters);
}

