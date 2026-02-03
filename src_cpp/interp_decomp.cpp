
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>
#include <numeric>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

int32_t device::interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP64, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t p = 0;
  rank = MixPrecAHA::iAHA(handle, 'N', epi, M, N, N, p, umax, Precision::FP64, const_cast<double*>(A), lda, &hpiv[0], Precision::FP64, nullptr, 0, precC, work, dpiv, alg);

  switch (precC) {
    case Precision::FP64:
      internal::InterpolativeDecomposition::interp_pp_f64_f64(stream, handle, rank, N, (double*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32:
      internal::InterpolativeDecomposition::interp_pp_f32_f64(stream, handle, rank, N, (float*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD:
      internal::InterpolativeDecomposition::interp_pp_f128_dd_f64(stream, handle, rank, N, (double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF:
      internal::InterpolativeDecomposition::interp_pp_f128_qf_f64(stream, handle, rank, N, (float4*)work, algnN, &hpiv[0], X, ldx); break;
    default: break;
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_f32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP32, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t p = 0;
  rank = MixPrecAHA::iAHA(handle, 'N', epi, M, N, N, p, umax, Precision::FP32, const_cast<float*>(A), lda, &hpiv[0], Precision::FP32, nullptr, 0, precC, work, dpiv, alg);

  switch (precC) {
    case Precision::FP64:
      internal::InterpolativeDecomposition::interp_pp_f64_f32(stream, handle, rank, N, (double*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32:
      internal::InterpolativeDecomposition::interp_pp_f32_f32(stream, handle, rank, N, (float*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD:
      internal::InterpolativeDecomposition::interp_pp_f128_dd_f32(stream, handle, rank, N, (double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF:
      internal::InterpolativeDecomposition::interp_pp_f128_qf_f32(stream, handle, rank, N, (float4*)work, algnN, &hpiv[0], X, ldx); break;
    default: break;
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_cf64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP64_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t p = 0;
  rank = MixPrecAHA::iAHA(handle, 'N', epi, M, N, N, p, umax, Precision::FP64_COMPLEX, const_cast<std::complex<double>*>(A), lda, &hpiv[0], Precision::FP64_COMPLEX, nullptr, 0, precC, work, dpiv, alg);

  switch (precC) {
    case Precision::FP64_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf64_cf64(stream, handle, rank, N, (std::complex<double>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf32_cf64(stream, handle, rank, N, (std::complex<float>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf64(stream, handle, rank, N, (complex_double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf64(stream, handle, rank, N, (complex_float4*)work, algnN, &hpiv[0], X, ldx); break;
    default: break;
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}

int32_t device::interp_decomp_cf32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnN = (N + 63) & (~63), umax; Precision precC; Algorithm alg; uint64_t dev_work_bytes, pinned_work_bytes;
  MixPrecAHA::igemm_params(epi, M, umax_exp_extra, &umax, Precision::FP32_COMPLEX, &precC, &alg);
  MixPrecAHA::igemm_workspace(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t p = 0;
  rank = MixPrecAHA::iAHA(handle, 'N', epi, M, N, N, p, umax, Precision::FP64_COMPLEX, const_cast<std::complex<float>*>(A), lda, &hpiv[0], Precision::FP64_COMPLEX, nullptr, 0, precC, work, dpiv, alg);

  switch (precC) {
    case Precision::FP64_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf64_cf32(stream, handle, rank, N, (std::complex<double>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf32_cf32(stream, handle, rank, N, (std::complex<float>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf32(stream, handle, rank, N, (complex_double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF_COMPLEX:
      internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf32(stream, handle, rank, N, (complex_float4*)work, algnN, &hpiv[0], X, ldx); break;
    default: break;
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}
