
#include <hyacin.hpp>
#include <internal.hpp>
#include <vector>

int32_t device::interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP64, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : (precC == Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(N) * elem_bytes + sizeof(int32_t));
  C_bytes = std::max(C_bytes, int64_t(algnN) * (int64_t(N) * (elem_bytes + std::min(elem_bytes, int64_t(sizeof(double)))) + int64_t(sizeof(int32_t))));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  MixPrecAHA::rATA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
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
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP32, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : sizeof(double);
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(N) * elem_bytes + sizeof(int32_t));
  C_bytes = std::max(C_bytes, int64_t(algnN) * (int64_t(N) * (elem_bytes + int64_t(sizeof(float))) + int64_t(sizeof(int32_t))));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  MixPrecAHA::rATA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
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
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP64, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : (precC == Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(2) * int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(2) * int64_t(N) * elem_bytes + sizeof(int32_t));
  C_bytes = std::max(C_bytes, int64_t(algnN) * (int64_t(2) * int64_t(N) * (elem_bytes + std::min(elem_bytes, int64_t(sizeof(double)))) + int64_t(sizeof(int32_t))));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  MixPrecAHA::cAHA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP64, work, precC);
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
  int32_t algnM, algnN, orderA; Precision precC;
  MixPrecAHA::mpgemm_params(&epi, M, N, &algnM, &algnN, &orderA, Precision::FP32, &precC);

  int64_t elem_bytes = precC == Precision::FP32 ? sizeof(float) : sizeof(double);
  int64_t C_bytes = int64_t(N) * int64_t(orderA) * (int64_t(2) * int64_t(algnM) + int64_t(algnN) * sizeof(int32_t))
    + int64_t(algnN) * (int64_t(2) * int64_t(N) * elem_bytes + sizeof(int32_t));
  C_bytes = std::max(C_bytes, int64_t(algnN) * (int64_t(2) * int64_t(N) * (elem_bytes + int64_t(sizeof(float))) + int64_t(sizeof(int32_t))));

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, C_bytes);
  cudaMallocHost((void**)(&dpiv), 8192);
  std::vector<int32_t> hpiv(N);

  MixPrecAHA::cAHA(stream, handle, M, N, algnM, algnN, orderA, A, lda, Precision::FP32, work, precC);
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
