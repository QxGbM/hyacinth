
#include <hyacin.hpp>
#include <internal.hpp>
#include <limits>

inline void gemm_accumulate(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft_lo, int32_t orderA, 
  int32_t alpha, const int8_t* AT, const int8_t* A, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {

  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, strideC, sft_lo, orderA, alpha, workspace, orderC, beta, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, strideC, sft_lo, orderA, alpha, workspace, orderC, k == 0 ? beta : 1, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor(stream, strideC, sft_lo, orderA, alpha, workspace, orderC, range_k == 0 ? beta : 1, C);
    if (iter_h < rem) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor(stream, strideC, sft_lo, orderA, alpha, workspace, orderC, 1, C);
    }
  }
}

inline void gemm_accumulate_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, 
  const int8_t* A, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor_sft2x(stream, strideC, orderA, workspace, orderC, beta, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor_sft2x(stream, strideC, orderA, workspace, orderC, k == 0 ? beta : 1, C);
    }

    const int8_t* A_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    internal::int8::accumulate_i32tensor_sft2x(stream, strideC, orderA, workspace, orderC, range_k == 0 ? beta : 1, C);
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::accumulate_i32tensor_sft2x(stream, strideC, orderA, workspace, orderC, 1, C);
    }
  }
}

inline void i8gemm_full(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, i, orderA, 1, &AT[AT_i], A, i == 0 ? beta : 1, C, orderC, workspace);
  }
}

inline void i8gemmt(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  gemm_accumulate_diag(stream, handle, algnN, N, algnK, orderA, A, beta, C, orderC, workspace);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t A_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, (i << 1) - 1, orderA - i, 2, &A[A_i - strideA], &A[A_i], 1, C, orderC, workspace);
  }
}

void device::MixPrecAHA::igemm_params(double* epi, int32_t N, int32_t* algnN, int32_t* umax, Precision precA, Precision* precC) {
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device = 100 * major + minor;

  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  std::pair<Precision, double> f128 = (device == 800 || device == 900 || device == 1000) ? 
    std::make_pair(device::Precision::FP128_DD, std::numeric_limits<double>::epsilon()) : 
    std::make_pair(device::Precision::FP128_QF, std::pow(double(std::numeric_limits<float>::epsilon()), 2));

  double machine_epi = precA == Precision::FP32 ? double(std::numeric_limits<float>::epsilon()) : f128.second;
  *algnN = (N + 63) & (~63);
  *epi = std::min(1., std::max(std::abs(*epi), machine_epi));
  *precC = epi_f32 <= *epi ? Precision::FP32 : (epi_f64 <= *epi ? Precision::FP64 : f128.first);
  *umax += int32_t(std::ceil(-std::log2(*epi)));
}

inline std::tuple<int32_t, int32_t, int32_t, int64_t, int64_t, int64_t, int64_t> i8gemm_ext_params(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, device::Precision prec) {
  int32_t algnM = (M + 255) & (~255);
  int32_t orderA = (umax + 9) >> 3;
  int64_t elem_bytes = prec == device::Precision::FP32 ? sizeof(float) : (prec == device::Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = int64_t(algnN) * int64_t(N) * elem_bytes;
  int64_t i8_bytes = int64_t(algnM) * int64_t(N) * int64_t(orderA);
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, (C_bytes - i8_bytes) << Complex);

  int32_t bits = int32_t(std::ceil(std::log2(double(std::max(M, 1))))) + (umax << 1) + Complex;
  int32_t orderC = 1 + (bits / 63);
  int64_t acc_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderC) * sizeof(uint64_t);
  int64_t vec_bytes = int64_t(algnN) * int64_t(Complex ? 8 : 6) * sizeof(uint64_t);
  return std::tie(algnM, orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes);
}

void device::MixPrecAHA::igemm_limbed_workspace(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, Precision precC, int64_t* workspace) {
  int32_t algnM, orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, Complex, precC);
  *workspace = ((i8_bytes + acc_bytes) << Complex) + scratch_bytes + vec_bytes;
}

void device::MixPrecAHA::rATA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t algnM, orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, 0, precC);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes];
  cudaMemsetAsync(iA, 0, i8_bytes, stream);
  cudaMemsetAsync(v_exp, 0, vec_bytes, stream);

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp);
    internal::int8::quantize_f64(stream, orderA, M, N, (const double*)A, lda, (uint64_t*)v_exp, iA, algnM);
    internal::int8::vsum_f64(stream, M, N, (const double*)A, lda, (uint64_t*)v_exp, algnN);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp);
    internal::int8::quantize_f32(stream, orderA, M, N, (const float*)A, lda, (uint64_t*)v_exp, iA, algnM);
    internal::int8::vsum_f32(stream, M, N, (const float*)A, lda, (uint64_t*)v_exp, algnN);
  }

  i8gemmt(stream, handle, N, algnN, algnM, iA, orderA, 0, (uint64_t*)acc, orderC, (int32_t*)workspace);

  if (precC == Precision::FP64)
    internal::int8::dequantize_f64(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (double*)iA, algnN);
  else if (precC == Precision::FP32)
    internal::int8::dequantize_f32(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (float*)iA, algnN);
  else if (precC == Precision::FP128_DD)
    internal::int8::dequantize_f128_dd(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (double2*)iA, algnN);
  else if (precC == Precision::FP128_QF)
    internal::int8::dequantize_f128_qf(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (float4*)iA, algnN);
}

void device::MixPrecAHA::cAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t algnM, orderA, orderC; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, orderA, orderC, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, 1, precC);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes + i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes + acc_bytes];
  int8_t* iA_imag = &iA[i8_bytes], *acc_imag = &acc[acc_bytes];
  cudaMemsetAsync(iA, 0, i8_bytes + i8_bytes, stream);
  cudaMemsetAsync(v_exp, 0, vec_bytes, stream);

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, 2 * M, N, (const double*)A, 2 * lda, umax, (uint64_t*)v_exp);
    internal::int8::quantize_cf64(stream, orderA, M, N, (const std::complex<double>*)A, lda, (uint64_t*)v_exp, iA, algnM);
    internal::int8::vsum_cf64(stream, M, N, (const std::complex<double>*)A, lda, (uint64_t*)v_exp, algnN);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, 2 * M, N, (const float*)A, 2 * lda, umax, (uint64_t*)v_exp);
    internal::int8::quantize_cf32(stream, orderA, M, N, (const std::complex<float>*)A, lda, (uint64_t*)v_exp, iA, algnM);
    internal::int8::vsum_cf32(stream, M, N, (const std::complex<float>*)A, lda, (uint64_t*)v_exp, algnN);
  }

  i8gemmt(stream, handle, N, algnN, algnM, iA, orderA, 0, (uint64_t*)acc, orderC, (int32_t*)workspace);
  i8gemmt(stream, handle, N, algnN, algnM, iA_imag, orderA, 1, (uint64_t*)acc, orderC, (int32_t*)workspace);
  i8gemm_full(stream, handle, N, algnN, algnM, iA, iA_imag, orderA, 0, (uint64_t*)acc_imag, orderC, (int32_t*)workspace);

  if (precC == Precision::FP64)
    internal::int8::dequantize_cf64(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (std::complex<double>*)iA, algnN);
  else if (precC == Precision::FP32)
    internal::int8::dequantize_cf32(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (std::complex<float>*)iA, algnN);
  else if (precC == Precision::FP128_DD)
    internal::int8::dequantize_cf128_dd(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (complex_double2*)iA, algnN);
  else if (precC == Precision::FP128_QF)
    internal::int8::dequantize_cf128_qf(stream, orderC, N, (uint64_t*)acc, algnN, (uint64_t*)v_exp, algnN, (complex_float4*)iA, algnN);
}
