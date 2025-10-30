
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>

inline void workspace_realloc(cudaStream_t stream, void** ptr, size_t bytes_old, size_t bytes_required, size_t bytes_migrate) {
  if (bytes_old < bytes_required) {
    void* workspace = nullptr;
    cudaStreamSynchronize(stream);
    cudaMalloc(&workspace, bytes_required);
    if (bytes_migrate)
      cudaMemcpy(workspace, *ptr, bytes_migrate, cudaMemcpyDeviceToDevice);
    if (*ptr)
      cudaFree(*ptr);
    *ptr = workspace;
  }
}

int32_t device::dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
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
  MixPrecAHA::rATA(stream, cublasH, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
  Cholesky::rpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    convert_and_copy(stream, iters, iters, work, algnN, precC, A, lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f64(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &C_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
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
  MixPrecAHA::rATA(stream, cublasH, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
  Cholesky::rpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    convert_and_copy(stream, iters, iters, work, algnN, precC, A, lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f32(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &C_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
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
  MixPrecAHA::cAHA(stream, cublasH, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
  Cholesky::cpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    convert_and_copy(stream, 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf64(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &C_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
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
  MixPrecAHA::cAHA(stream, cublasH, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
  Cholesky::cpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    convert_and_copy(stream, 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf32(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &C_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

