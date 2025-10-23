
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

int32_t device::dgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP64, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : (precC == Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t acc_bytes = int64_t(algnN) * (int64_t(N) * elem_bytes + sizeof(int32_t));
  int64_t C_bytes = acc_bytes + std::max(int64_t(N) * int64_t(orderA) * (int64_t(algnM) + int64_t(algnN) * int64_t(sizeof(int32_t))), int64_t(8192) * int64_t(N) * int64_t(sizeof(double)));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, cublasH, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
  Cholesky::rpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  int8_t* C = &((int8_t*)work)[int64_t(algnN) * int64_t(N) * elem_bytes];

  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost;
  cusolverDnXgeqrf_bufferSize(cusolverH, params, M, iters, CUDA_R_64F, A, lda, CUDA_R_64F, tau, CUDA_R_64F, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
  cudaMemcpyAsync(C, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyHostToDevice, stream);
  inplace_gather(stream, M, iters, (int32_t*)C, A, lda, &C[int64_t(algnN) * sizeof(int32_t)], Precision::FP64);

  workspace_realloc(stream, &work, C_bytes, acc_bytes + workspaceInBytesOnDevice, acc_bytes);
  C = &((int8_t*)work)[int64_t(algnN) * int64_t(N) * elem_bytes];

  std::vector<uint8_t> bufferOnHost(workspaceInBytesOnHost);
  cusolverDnXgeqrf(cusolverH, params, M, iters, CUDA_R_64F, A, lda, CUDA_R_64F, tau, CUDA_R_64F, C, workspaceInBytesOnDevice, &bufferOnHost[0], workspaceInBytesOnHost, (int32_t*)&C[workspaceInBytesOnDevice]);

  if (iters < N) {
    copy_signs(stream, iters, A, lda + 1, Precision::FP64, C, 1);
    convert_and_copy(stream, 'u', iters, N - iters, &((int8_t*)work)[int64_t(algnN) * int64_t(iters) * elem_bytes], algnN, precC, &A[int64_t(lda) * int64_t(iters)], lda, Precision::FP64, C);
    cudaMemsetAsync(&tau[iters], 0, int64_t(N - iters) * sizeof(double), stream);
    cudaMemset2DAsync(&A[int64_t(lda + 1) * int64_t(iters)], int64_t(lda) * sizeof(double), 0, int64_t(M - iters) * sizeof(double), N - iters, stream);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::sgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
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
  int8_t* C = &((int8_t*)work)[int64_t(algnN) * int64_t(N) * elem_bytes];
  cudaMemsetAsync(C, 0, iters, stream);
  convert_and_copy(stream, 'u', iters, iters, work, algnN, precC, A, lda, Precision::FP32, C);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::zgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
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
  int8_t* C = &((int8_t*)work)[int64_t(2) * int64_t(algnN) * int64_t(N) * elem_bytes];
  cudaMemsetAsync(C, 0, 2 * iters, stream);
  convert_and_copy(stream, 'U', 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP64, C);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::cgeqp3_ronly(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
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
  int8_t* C = &((int8_t*)work)[int64_t(2) * int64_t(algnN) * int64_t(N) * elem_bytes];
  cudaMemsetAsync(C, 0, 2 * iters, stream);
  convert_and_copy(stream, 'U', 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP32, C);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

