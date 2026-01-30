
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>
#include <numeric>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

int32_t device::dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP64, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  MixPrecAHA::iAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP64, work, precC, alg);
  switch (precC) {
    case Precision::FP64:
      rank = internal::Cholesky::potrfp_f64(stream, cublasH, epi, rank, p, N, (double*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP32:
      rank = internal::Cholesky::potrfp_f32(stream, cublasH, epi, rank, p, N, (float*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_DD:
      rank = internal::Cholesky::potrfp_f128_dd(stream, cublasH, epi, rank, p, N, (double2*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_QF:
      rank = internal::Cholesky::potrfp_f128_qf(stream, cublasH, epi, rank, p, N, (float4*)work, algnN, &hpiv[0], dpiv); break;
    default: break;
  }

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP64);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP32, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  MixPrecAHA::iAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP32, work, precC, alg);
  switch (precC) {
    case Precision::FP64:
      rank = internal::Cholesky::potrfp_f64(stream, cublasH, epi, rank, p, N, (double*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP32:
      rank = internal::Cholesky::potrfp_f32(stream, cublasH, epi, rank, p, N, (float*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_DD:
      rank = internal::Cholesky::potrfp_f128_dd(stream, cublasH, epi, rank, p, N, (double2*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_QF:
      rank = internal::Cholesky::potrfp_f128_qf(stream, cublasH, epi, rank, p, N, (float4*)work, algnN, &hpiv[0], dpiv); break;
    default: break;
  }

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP32);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_f32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP64_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  MixPrecAHA::iAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP64_COMPLEX, work, precC, alg);
  switch (precC) {
    case Precision::FP64_COMPLEX:
      rank = internal::Cholesky::potrfp_cf64(stream, cublasH, epi, rank, p, N, (std::complex<double>*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP32_COMPLEX:
      rank = internal::Cholesky::potrfp_cf32(stream, cublasH, epi, rank, p, N, (std::complex<float>*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_DD_COMPLEX:
      rank = internal::Cholesky::potrfp_cf128_dd(stream, cublasH, epi, rank, p, N, (complex_double2*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_QF_COMPLEX:
      rank = internal::Cholesky::potrfp_cf128_qf(stream, cublasH, epi, rank, p, N, (complex_float4*)work, algnN, &hpiv[0], dpiv); break;
    default: break;
  }
  
  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP64_COMPLEX);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

int32_t device::cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t algnN, umax = umax_exp_extra; Precision precC; Algorithm alg; int64_t work_bytes;
  MixPrecAHA::igemm_params(&epi, N, &algnN, &umax, Precision::FP32_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, algnN, umax, precC, alg, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  MixPrecAHA::iAHA(stream, cublasH, M, N, algnN, umax, A, lda, Precision::FP32_COMPLEX, work, precC, alg);
  switch (precC) {
    case Precision::FP64_COMPLEX:
      rank = internal::Cholesky::potrfp_cf64(stream, cublasH, epi, rank, p, N, (std::complex<double>*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP32_COMPLEX:
      rank = internal::Cholesky::potrfp_cf32(stream, cublasH, epi, rank, p, N, (std::complex<float>*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_DD_COMPLEX:
      rank = internal::Cholesky::potrfp_cf128_dd(stream, cublasH, epi, rank, p, N, (complex_double2*)work, algnN, &hpiv[0], dpiv); break;
    case Precision::FP128_QF_COMPLEX:
      rank = internal::Cholesky::potrfp_cf128_qf(stream, cublasH, epi, rank, p, N, (complex_float4*)work, algnN, &hpiv[0], dpiv); break;
    default: break;
  }

  if (mode == 'R' || mode == 'r')
    device::Utils::convert_and_copy(stream, rank, rank, work, algnN, precC, A, lda, Precision::FP32_COMPLEX);
  else if (mode != 'P' && mode != 'p')
    internal::Orthogonalize::qr_pp_cf32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &work_bytes);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

