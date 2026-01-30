
#include <hyacin.hpp>
#include <internal.hpp>
#include <crt_selector.hpp>
#include <limits>
#include <tuple>

const int32_t umax_threshold = 30; // umax < 30 : Limbs, 30 <= umax : CRT

inline device::Precision real_precision(device::Precision prec) {
  switch (prec) {
    case device::Precision::FP64_COMPLEX: return device::Precision::FP64;
    case device::Precision::FP32_COMPLEX: return device::Precision::FP32;
    case device::Precision::FP128_DD_COMPLEX: return device::Precision::FP128_DD;
    case device::Precision::FP128_QF_COMPLEX: return device::Precision::FP128_QF;
    default: return prec;
  }
}

inline device::Precision complex_precision(device::Precision prec) {
  switch (prec) {
    case device::Precision::FP64: return device::Precision::FP64_COMPLEX;
    case device::Precision::FP32: return device::Precision::FP32_COMPLEX;
    case device::Precision::FP128_DD: return device::Precision::FP128_DD_COMPLEX;
    case device::Precision::FP128_QF: return device::Precision::FP128_QF_COMPLEX;
    default: return prec;
  }
}

void device::MixPrecAHA::igemm_params(double* epi, int32_t N, int32_t* algnN, int32_t* umax, Precision precA, Precision* precC, Algorithm* alg) {
  Precision precR = real_precision(precA);
  int32_t device, major, minor;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device = 100 * major + minor;

  double epi_f32 = std::sqrt(double(std::numeric_limits<float>::epsilon()));
  double epi_f64 = std::sqrt(std::numeric_limits<double>::epsilon());
  std::pair<Precision, double> f128 = (device == 800 || device == 900 || device == 1000) ? 
    std::make_pair(Precision::FP128_DD, std::numeric_limits<double>::epsilon()) :
    std::make_pair(Precision::FP128_QF, std::pow(double(std::numeric_limits<float>::epsilon()), 2));

  double machine_epi = precR == Precision::FP32 ? double(std::numeric_limits<float>::epsilon()) : f128.second;
  *algnN = (N + 63) & (~63);
  *epi = std::min(1., std::max(std::abs(*epi), machine_epi));
  *umax = *umax + int32_t(std::ceil(-std::log2(*epi)));

  int32_t Complex = int32_t(precA != precR);
  Precision prec = epi_f32 <= *epi ? Precision::FP32 : (epi_f64 <= *epi ? Precision::FP64 : f128.first);
  *precC = Complex ? complex_precision(prec) : prec;
  *alg = (*umax < umax_threshold) ? Algorithm::Limbs : Algorithm::CRT;
}

inline std::tuple<int32_t, int32_t, int32_t, int32_t, int64_t, int64_t, int64_t, int64_t> i8gemm_ext_params(int32_t M, int32_t N, int32_t algnN, int32_t umax, device::Precision prec, device::Algorithm alg) {
  device::Precision precR = real_precision(prec);
  int32_t Complex = int32_t(prec != precR);
  int32_t algnM = (M + 255) & (~255);
  int32_t orderA = (alg == device::Algorithm::Limbs) ? ((umax + 10) >> 3) : 8;
  int64_t elem_bytes = precR == device::Precision::FP32 ? sizeof(float) : (precR == device::Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t C_bytes = (int64_t(algnN) * int64_t(N) * elem_bytes) << Complex;
  int64_t i8_bytes = (int64_t(algnM) * int64_t(N) * int64_t(orderA)) << Complex;
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, C_bytes - i8_bytes);

  int32_t bits = int32_t(std::ceil(std::log2(double(M)))) + (umax << 1) + 2 + Complex;
  int32_t n_moduli = (bits + 8) >> 3;
  int32_t orderC = (alg == device::Algorithm::Limbs) ? ((bits + 63) / 63) : (((n_moduli << 3) + 63) / 63);
  int64_t acc_bytes = (int64_t(algnN) * int64_t(N) * int64_t(orderC) * sizeof(uint64_t)) << Complex;
  int64_t vec_bytes = int64_t(algnN) * int64_t(Complex ? 5 : 3) * sizeof(uint64_t);
  return std::tie(algnM, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes);
}

void device::MixPrecAHA::igemm_workspace(int32_t M, int32_t N, int32_t algnN, int32_t umax, Precision precC, Algorithm alg, int64_t* workspace) {
  int32_t algnM, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, precC, alg);
  *workspace = i8_bytes + acc_bytes + scratch_bytes + vec_bytes;
}

inline void gemm_accumulate(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t sft_lo, int32_t orderA, const int8_t* AT, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, K, &one, AT, CUDA_R_8I, K, A, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, strideC, sft_lo, orderA, workspace, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = op | 1;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, iter_k, &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, std::min(rem, iter_h), &one, AT_k, CUDA_R_8I, K, AN_k, CUDA_R_8I, K,
      &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    if (iter_h < rem) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N * orderA, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, &AN_k[iter_h], CUDA_R_8I, K,
        &zero, workspace, CUDA_R_32I, M, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, sft_lo, orderA, workspace, orderC, C);
    }
  }
}

inline void gemm_accumulate_diag(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t orderA, const int8_t* A, int32_t op, uint64_t* C, int32_t orderC, int32_t* workspace) {
  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t one = 1, zero = 0;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, op, strideC, 0, orderA, workspace, orderC, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem, op_acc = 5;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* A_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, k == 0 ? op : op_acc, strideC, 0, orderA, workspace, orderC, C);
    }

    const int8_t* A_k = &A[int64_t(range_k)];
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, std::min(rem, iter_h), &one, A_k, CUDA_R_8I, K, strideA, A_k, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_i32tensor(stream, range_k == 0 ? op : op_acc, strideC, 0, orderA, workspace, orderC, C);
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &A_k[iter_h], CUDA_R_8I, K, strideA, &A_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, orderA, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_i32tensor(stream, op_acc, strideC, 0, orderA, workspace, orderC, C);
    }
  }
}

inline void i8GemmF(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  for (int32_t i = 0; i < orderA; ++i) {
    int64_t AT_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, i, orderA, &AT[AT_i], A, i == 0 ? beta : 1, C, orderC, workspace);
  }
}

inline void i8GemmT(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, int32_t beta, uint64_t* C, int32_t orderC, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  gemm_accumulate_diag(stream, handle, algnN, N, algnK, orderA, A, 4 | beta, C, orderC, workspace);
  for (int32_t i = 1; i < orderA; ++i) {
    int64_t A_i = int64_t(i) * strideA;
    gemm_accumulate(stream, handle, algnN, N, algnK, (i << 1) - 1, orderA - i, &A[A_i - strideA], &A[A_i], 3, C, orderC, workspace);
  }
}

inline void i63ATA_f64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  internal::int8::quantize_f64(stream, M, N, A, lda, umax, vec_expon, orderA, workspace, algnM);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
}

inline void i63ATA_f32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA);
  int32_t* scratch = (int32_t*)&workspace[strideA];
  internal::int8::quantize_f32(stream, M, N, A, lda, umax, vec_expon, orderA, workspace, algnM);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
}

inline void i63AHA_cf64_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N) * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  internal::int8::quantize_cf64(stream, M, N, A, lda, umax, vec_expon, orderA, workspace, algnM);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, N, ldc, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, N, ldc, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideC], orderC, scratch);
}

inline void i63AHA_cf32_limbs(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t orderA, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N) * int64_t(orderA), strideC = int64_t(ldc) * int64_t(N) * int64_t(orderC);
  int32_t* scratch = (int32_t*)&workspace[strideA << 1];
  internal::int8::quantize_cf32(stream, M, N, A, lda, umax, vec_expon, orderA, workspace, algnM);
  i8GemmT(stream, handle, N, ldc, algnM, workspace, orderA, 0, C, orderC, scratch);
  i8GemmT(stream, handle, N, ldc, algnM, &workspace[strideA], orderA, 1, C, orderC, scratch);
  i8GemmF(stream, handle, N, ldc, algnM, workspace, &workspace[strideA], orderA, 0, &C[strideC], orderC, scratch);
}

inline void gemm_accumulate_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t K, int32_t moduli, const int8_t* AT, const int8_t* A, 
  int32_t n_moduli, int32_t iter, int32_t accum, int32_t last, uint64_t* C, int32_t* workspace) {

  constexpr int32_t iter_k = 131072, iter_h = iter_k / 2;
  int32_t zero = 0, one = 1;
  int64_t strideA = int64_t(N) * int64_t(K), strideC = int64_t(M) * int64_t(N);
  if (K <= iter_k) {
    cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &one, AT, CUDA_R_8I, K, strideA, A, CUDA_R_8I, K, strideA,
      &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
    internal::int8::accumulate_remainder_i32tensor(stream, accum | (last << 1), strideC, n_moduli, iter, workspace, C);
  }
  else {
    int32_t rem = K & (iter_k - 1); rem = rem < iter_h ? (rem + iter_k) : rem;
    int32_t range_k = K - rem;

    for (int32_t k = 0; k < range_k; k += iter_k) {
      const int8_t* AT_k = &AT[int64_t(k)];
      const int8_t* AN_k = &A[int64_t(k)];
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_k, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, k == 0 ? accum : 1, strideC, n_moduli, iter, workspace, C);
    }

    const int8_t* AT_k = &AT[int64_t(range_k)];
    const int8_t* AN_k = &A[int64_t(range_k)];
    if (iter_h < rem) {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, iter_h, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, range_k == 0 ? accum : 1, strideC, n_moduli, iter, workspace, C);
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem - iter_h, &one, &AT_k[iter_h], CUDA_R_8I, K, strideA, &AN_k[iter_h], CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, 1 | (last << 1), strideC, n_moduli, iter, workspace, C);
    }
    else {
      cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, rem, &one, AT_k, CUDA_R_8I, K, strideA, AN_k, CUDA_R_8I, K, strideA,
        &zero, workspace, CUDA_R_32I, M, strideC, moduli, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT);
      internal::int8::accumulate_remainder_i32tensor(stream, (range_k == 0 ? accum : 1) | (last << 1), strideC, n_moduli, iter, workspace, C);
    }
  }
}

inline void i63ATA_f64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t n_moduli, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t moduli = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    internal::int8::quantize_f64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, workspace, algnM);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, n_moduli, i, accum, last, C, scratch);
  }
}

inline void i63ATA_f32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t n_moduli, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 3];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t moduli = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    internal::int8::quantize_f32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, workspace, algnM);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, n_moduli, i, accum, last, C, scratch);
  }
}

inline void i63AHA_cf64_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t n_moduli, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(orderC) * int64_t(ldc) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t moduli = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    internal::int8::quantize_cf64_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, workspace, algnM);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, n_moduli, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, &workspace[stride], &workspace[stride], n_moduli, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, &workspace[stride], n_moduli, i, accum, last, &C[strideC], scratch);
  }
}

inline void i63AHA_cf32_crt(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const uint64_t* vec_expon,
  int32_t algnM, int32_t n_moduli, int32_t orderC, uint64_t* C, int32_t ldc, int8_t* workspace) {

  int64_t strideA = int64_t(algnM) * int64_t(N), strideC = int64_t(orderC) * int64_t(ldc) * int64_t(N);
  int32_t* scratch = (int32_t*)&workspace[strideA << 4];

  for (int32_t i = 0; (i << 3) < n_moduli; ++i) {
    int32_t moduli = CRT::active_moduli(n_moduli, i);
    int32_t accum = int32_t(0 < i), last = int32_t(n_moduli <= ((i + 1) << 3));
    internal::int8::quantize_cf32_modular(stream, M, N, i, A, lda, umax, vec_expon, moduli, workspace, algnM);

    int64_t stride = int64_t(moduli) * strideA;
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, workspace, n_moduli, i, accum, 0, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, &workspace[stride], &workspace[stride], n_moduli, i, 1, last, C, scratch);
    gemm_accumulate_crt(stream, handle, ldc, N, algnM, moduli, workspace, &workspace[stride], n_moduli, i, accum, last, &C[strideC], scratch);
  }
}

void device::MixPrecAHA::iAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC, Algorithm alg) {
  int32_t algnM, orderA, orderC, n_moduli; int64_t i8_bytes, scratch_bytes, acc_bytes, vec_bytes;
  std::tie(algnM, orderA, orderC, n_moduli, i8_bytes, scratch_bytes, acc_bytes, vec_bytes) = i8gemm_ext_params(M, N, algnN, umax, precC, alg);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes];

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f64(stream, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      i63ATA_f64_limbs(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      i63ATA_f64_crt(stream, handle, M, N, (const double*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, (uint64_t*)v_exp);
    internal::int8::vsum_f32(stream, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      i63ATA_f32_limbs(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      i63ATA_f32_crt(stream, handle, M, N, (const float*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, (uint64_t*)acc, algnN, iA);
  }
  else if (precA == Precision::FP64_COMPLEX) {
    internal::int8::vexp_f64(stream, 2 * M, N, (const double*)A, 2 * lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf64(stream, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      i63AHA_cf64_limbs(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      i63AHA_cf64_crt(stream, handle, M, N, (const std::complex<double>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }
  else if (precA == Precision::FP32_COMPLEX) {
    internal::int8::vexp_f32(stream, 2 * M, N, (const float*)A, 2 * lda, (uint64_t*)v_exp);
    internal::int8::vsum_cf32(stream, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnN);
    if (alg == device::Algorithm::Limbs)
      i63AHA_cf32_limbs(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, orderA, orderC, (uint64_t*)acc, algnN, iA);
    else
      i63AHA_cf32_crt(stream, handle, M, N, (const std::complex<float>*)A, lda, umax, (uint64_t*)v_exp, algnM, n_moduli, orderC, (uint64_t*)acc, algnN, iA);
  }

  switch (precC) {
    case Precision::FP64:
      internal::int8::dequantize_f64(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double*)iA, algnN); break;
    case Precision::FP32:
      internal::int8::dequantize_f32(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float*)iA, algnN); break;
    case Precision::FP128_DD:
      internal::int8::dequantize_f128_dd(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (double2*)iA, algnN); break;
    case Precision::FP128_QF:
      internal::int8::dequantize_f128_qf(stream, orderC, M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (float4*)iA, algnN); break;
    case Precision::FP64_COMPLEX:
      internal::int8::dequantize_cf64(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<double>*)iA, algnN); break;
    case Precision::FP32_COMPLEX:
      internal::int8::dequantize_cf32(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (std::complex<float>*)iA, algnN); break;
    case Precision::FP128_DD_COMPLEX:
      internal::int8::dequantize_cf128_dd(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_double2*)iA, algnN); break;
    case Precision::FP128_QF_COMPLEX:
      internal::int8::dequantize_cf128_qf(stream, orderC, 2 * M, N, (uint64_t*)acc, algnN, umax, (uint64_t*)v_exp, algnN, (complex_float4*)iA, algnN); break;
    default: break;
  }
}

