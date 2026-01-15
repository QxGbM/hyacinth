
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>
#include <numeric>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

int32_t device::dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP64, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, 0, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP64, work, precC, alg);
  Cholesky::rpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, iters, iters, work, algnN, precC, A, lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f64(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP32, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, 0, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t iters = N; 
  MixPrecAHA::rATA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP32, work, precC, alg);
  Cholesky::rpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, iters, iters, work, algnN, precC, A, lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f32(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP64, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, 1, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP64, work, precC, alg);
  Cholesky::cpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf64(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

int32_t device::cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP32, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, 1, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t iters = N; 
  MixPrecAHA::cAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP32, work, precC, alg);
  Cholesky::cpotrfp(stream, cublasH, epi, &iters, N, work, algnN, precC, &hpiv[0], dpiv);
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, 2 * iters, iters, work, 2 * algnN, precC, A, 2 * lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf32(stream, cusolverH, M, N, iters, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return iters == N ? 0 : (iters + 1);
}

