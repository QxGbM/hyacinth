
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>
#include <numeric>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

int32_t device::dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP64, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  rank = MixPrecAHA::iAHA(cublasH, 'N', epi, M, N, N, p, umax, Precision::FP64, const_cast<double*>(A), lda, &hpiv[0], Precision::FP64, nullptr, 0, precC, work, dpiv, alg);

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP32, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  rank = MixPrecAHA::iAHA(cublasH, 'N', epi, M, N, N, p, umax, Precision::FP32, const_cast<float*>(A), lda, &hpiv[0], Precision::FP32, nullptr, 0, precC, work, dpiv, alg);

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP64_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  rank = MixPrecAHA::iAHA(cublasH, 'N', epi, M, N, N, p, umax, Precision::FP64_COMPLEX, const_cast<std::complex<double>*>(A), lda, &hpiv[0], Precision::FP64_COMPLEX, nullptr, 0, precC, work, dpiv, alg);
  
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP64_COMPLEX);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP32_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  rank = MixPrecAHA::iAHA(cublasH, 'N', epi, M, N, N, p, umax, Precision::FP32_COMPLEX, const_cast<std::complex<float>*>(A), lda, &hpiv[0], Precision::FP32_COMPLEX, nullptr, 0, precC, work, dpiv, alg);

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP32_COMPLEX);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

