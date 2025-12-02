
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>
#include <numeric>

int32_t device::interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnN, umax; Precision precC; int64_t work_bytes;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnN, &umax, 0, Precision::FP64, &precC, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  MixPrecAHA::rATA(stream, handle, M, N, algnN, umax, A, lda, Precision::FP64, work, precC);
  Cholesky::rpotrfp(stream, handle, epi, &rank, N, work, algnN, precC, &hpiv[0], dpiv);
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
  int32_t algnN, umax; Precision precC; int64_t work_bytes;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnN, &umax, 0, Precision::FP32, &precC, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  MixPrecAHA::rATA(stream, handle, M, N, algnN, umax, A, lda, Precision::FP32, work, precC);
  Cholesky::rpotrfp(stream, handle, epi, &rank, N, work, algnN, precC, &hpiv[0], dpiv);
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
  int32_t algnN, umax; Precision precC; int64_t work_bytes;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnN, &umax, 1, Precision::FP64, &precC, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  MixPrecAHA::cAHA(stream, handle, M, N, algnN, umax, A, lda, Precision::FP64, work, precC);
  Cholesky::cpotrfp(stream, handle, epi, &rank, N, work, algnN, precC, &hpiv[0], dpiv);
  switch (precC) {
    case Precision::FP64:
      internal::InterpolativeDecomposition::interp_pp_cf64_cf64(stream, handle, rank, N, (std::complex<double>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32:
      internal::InterpolativeDecomposition::interp_pp_cf32_cf64(stream, handle, rank, N, (std::complex<float>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD:
      internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf64(stream, handle, rank, N, (complex_double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF:
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
  int32_t algnN, umax; Precision precC; int64_t work_bytes;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnN, &umax, 1, Precision::FP32, &precC, &work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, work_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  MixPrecAHA::cAHA(stream, handle, M, N, algnN, umax, A, lda, Precision::FP32, work, precC);
  Cholesky::cpotrfp(stream, handle, epi, &rank, N, work, algnN, precC, &hpiv[0], dpiv);
  switch (precC) {
    case Precision::FP64:
      internal::InterpolativeDecomposition::interp_pp_cf64_cf32(stream, handle, rank, N, (std::complex<double>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP32:
      internal::InterpolativeDecomposition::interp_pp_cf32_cf32(stream, handle, rank, N, (std::complex<float>*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_DD:
      internal::InterpolativeDecomposition::interp_pp_cf128_dd_cf32(stream, handle, rank, N, (complex_double2*)work, algnN, &hpiv[0], X, ldx); break;
    case Precision::FP128_QF:
      internal::InterpolativeDecomposition::interp_pp_cf128_qf_cf32(stream, handle, rank, N, (complex_float4*)work, algnN, &hpiv[0], X, ldx); break;
    default: break;
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank;
}
