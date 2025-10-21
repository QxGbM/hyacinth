
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>

int32_t device::dgeqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP64, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : (precC == Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(N) * elem_bytes + sizeof(int32_t));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
  Cholesky::rpotrfp(stream, handle, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  convert_and_copy(stream, N, N, work, algnN, precC, 0, A, lda, Precision::FP64);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::sgeqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP32, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : sizeof(double);
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(N) * elem_bytes + sizeof(int32_t));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
  Cholesky::rpotrfp(stream, handle, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  convert_and_copy(stream, N, N, work, algnN, precC, 0, A, lda, Precision::FP32);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::zgeqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP64, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : (precC == Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(2) * int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(2) * int64_t(N) * elem_bytes + sizeof(int32_t));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
  Cholesky::cpotrfp(stream, handle, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  convert_and_copy(stream, 2 * N, N, work, 2 * algnN, precC, 0, A, 2 * lda, Precision::FP64);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::cgeqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP32, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : sizeof(double);
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(2) * int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(2) * int64_t(N) * elem_bytes + sizeof(int32_t));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
  Cholesky::cpotrfp(stream, handle, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  convert_and_copy(stream, 2 * N, N, work, 2 * algnN, precC, 0, A, 2 * lda, Precision::FP32);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

