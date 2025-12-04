
#include <hyacin.hpp>
#include <internal.hpp>
#include <limits>

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

inline std::tuple<int32_t, int32_t, int64_t, int64_t, int64_t> i8gemm_work(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, device::Precision prec) {
  int32_t algnM = (M + 255) & (~255);
  int32_t orderA = (umax + device::Config::exp_base - 1) / device::Config::exp_base;
  int64_t elem_bytes = prec == device::Precision::FP32 ? sizeof(float) : (prec == device::Precision::FP64 ? sizeof(double) : sizeof(double2));
  int64_t acc_bytes = int64_t(algnN) * int64_t(N) * elem_bytes;
  int64_t i8_bytes = int64_t(algnM) * int64_t(N) * int64_t(orderA);
  int64_t scratch_bytes = int64_t(algnN) * int64_t(N) * int64_t(orderA) * sizeof(int32_t);
  scratch_bytes = std::max(scratch_bytes, (acc_bytes - i8_bytes) << Complex);
  return std::tie(algnM, orderA, acc_bytes, i8_bytes, scratch_bytes);
}

void device::MixPrecAHA::igemm_limbed_workspace(int32_t M, int32_t N, int32_t algnN, int32_t umax, int32_t Complex, Precision precC, int64_t* workspace) {
  int32_t algnM, orderA; int64_t acc_bytes, i8_bytes, scratch_bytes;
  std::tie(algnM, orderA, acc_bytes, i8_bytes, scratch_bytes) = i8gemm_work(M, N, algnN, umax, Complex, precC);
  int64_t vec_bytes = int64_t(algnN) * sizeof(int32_t);
  *workspace = ((i8_bytes + acc_bytes) << Complex) + scratch_bytes + vec_bytes;
}

template <device::Precision prec>
inline void dequantize_dispatcher(cudaStream_t stream, int32_t depth_lo, int32_t depth_hi, int32_t N, void* A, const int32_t* B, int32_t ld) {
  if constexpr(prec == device::Precision::FP64)
    internal::int8::dequantize_f64_i32tensor(stream, depth_lo, depth_hi, N, (double*)A, B, ld);
  else if constexpr(prec == device::Precision::FP32)
    internal::int8::dequantize_f32_i32tensor(stream, depth_lo, depth_hi, N, (float*)A, B, ld);
  else if constexpr(prec == device::Precision::FP128_DD)
    internal::int8::dequantize_f128_dd_i32tensor(stream, depth_lo, depth_hi, N, (double2*)A, B, ld);
  else if constexpr(prec == device::Precision::FP128_QF)
    internal::int8::dequantize_f128_qf_i32tensor(stream, depth_lo, depth_hi, N, (float4*)A, B, ld);
}

template <device::Precision prec>
inline void i8gemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, void* C, int32_t* workspace) {
  constexpr int32_t iter_k = 131072 << (14 - 2 * device::Config::exp_base), iter_h = iter_k / 2;
  int64_t strideA = int64_t(algnK) * int64_t(N);
  int32_t one = 1, zero = 0;
  
  if (algnK <= iter_k)
    for (int32_t i = 0; i < orderA; ++i) {
      int64_t AT_i = int64_t(i) * strideA;
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, algnK, &one, 
        &AT[AT_i], CUDA_R_8I, algnK, A, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      dequantize_dispatcher<prec>(stream, i, i + orderA, N, C, workspace, algnN);
    }
  else {
    int32_t rem = algnK & (iter_k - 1);
    rem = (rem == 0) ? iter_k : (rem < iter_h ? (rem + iter_k) : rem);
    int32_t range_k = algnK - rem;

    for (int32_t i = 0; i < orderA; ++i) {
      int64_t AT_i = int64_t(i) * strideA;
      for (int32_t k = 0; k < range_k; k += iter_k) {
        const int8_t* AT_k = &AT[int64_t(k) + AT_i];
        const int8_t* AN_k = &A[int64_t(k)];

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, iter_k, &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        dequantize_dispatcher<prec>(stream, i, i + orderA, N, C, workspace, algnN);
      }

      const int8_t* AT_k = &AT[int64_t(range_k) + AT_i];
      const int8_t* AN_k = &A[int64_t(range_k)];
      if (rem <= iter_k)
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, rem, &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      else {
        const int8_t* AT_k2 = &AT[int64_t(range_k + iter_h) + AT_i];
        const int8_t* AN_k2 = &A[int64_t(range_k + iter_h)];

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, iter_h, &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        dequantize_dispatcher<prec>(stream, i, i + orderA, N, C, workspace, algnN);

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, rem - iter_h, &one, 
          AT_k2, CUDA_R_8I, algnK, AN_k2, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      dequantize_dispatcher<prec>(stream, i, i + orderA, N, C, workspace, algnN);
    }
  }
}

void device::MixPrecAHA::rATA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t algnM, orderA; int64_t acc_bytes, i8_bytes, scratch_bytes;
  std::tie(algnM, orderA, acc_bytes, i8_bytes, scratch_bytes) = i8gemm_work(M, N, algnN, umax, 0, precC);

  int8_t* acc = (int8_t*)(C), *iA = &acc[acc_bytes];
  int8_t* workspace = &iA[i8_bytes], *v_exp = &workspace[scratch_bytes];
  cudaMemsetAsync(acc, 0, acc_bytes + i8_bytes, stream);

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, M, N, (const double*)A, lda, umax, (int32_t*)v_exp);
    internal::int8::quantize_f64(stream, orderA, M, N, (const double*)A, lda, (int32_t*)v_exp, iA, algnM);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, M, N, (const float*)A, lda, umax, (int32_t*)v_exp);
    internal::int8::quantize_f32(stream, orderA, M, N, (const float*)A, lda, (int32_t*)v_exp, iA, algnM);
  }

  if (precC == Precision::FP64) {
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f64(stream, N, (double*)acc, algnN, (int32_t*)v_exp);
  }
  else if (precC == Precision::FP32) {
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f32(stream, N, (float*)acc, algnN, (int32_t*)v_exp);
  }
  else if (precC == Precision::FP128_DD) {
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f128_dd(stream, N, (double2*)acc, algnN, (int32_t*)v_exp);
  }
  else if (precC == Precision::FP128_QF) {
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f128_qf(stream, N, (float4*)acc, algnN, (int32_t*)v_exp);
  }
}

void device::MixPrecAHA::cAHA(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, int32_t algnN, int32_t umax, const void* A, int32_t lda, Precision precA, void* C, Precision precC) {
  int32_t algnM, orderA; int64_t acc_bytes, i8_bytes, scratch_bytes;
  std::tie(algnM, orderA, acc_bytes, i8_bytes, scratch_bytes) = i8gemm_work(M, N, algnN, umax, 1, precC);

  int8_t* iA = (int8_t*)(C), *workspace = &iA[i8_bytes + i8_bytes];
  int8_t* acc = &workspace[scratch_bytes], *v_exp = &acc[acc_bytes + acc_bytes];
  int8_t* iA_imag = &iA[i8_bytes], *acc_imag = &acc[acc_bytes];
  cudaMemsetAsync(iA, 0, i8_bytes + i8_bytes, stream);
  cudaMemsetAsync(acc, 0, acc_bytes + acc_bytes, stream);

  if (precA == Precision::FP64) {
    internal::int8::vexp_f64(stream, 2 * M, N, (const double*)A, 2 * lda, umax, (int32_t*)v_exp);
    internal::int8::quantize_cf64(stream, orderA, M, N, (const std::complex<double>*)A, lda, (int32_t*)v_exp, iA, algnM);
  }
  else if (precA == Precision::FP32) {
    internal::int8::vexp_f32(stream, 2 * M, N, (const float*)A, 2 * lda, umax, (int32_t*)v_exp);
    internal::int8::quantize_cf32(stream, orderA, M, N, (const std::complex<float>*)A, lda, (int32_t*)v_exp, iA, algnM);
  }

  if (precC == Precision::FP64) {
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f64(stream, N, (double*)acc, algnN, (int32_t*)v_exp, (std::complex<double>*)iA, algnN);
  }
  else if (precC == Precision::FP32) {
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f32(stream, N, (float*)acc, algnN, (int32_t*)v_exp, (std::complex<float>*)iA, algnN);
  }
  else if (precC == Precision::FP128_DD) {
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f128_dd(stream, N, (double2*)acc, algnN, (int32_t*)v_exp, (complex_double2*)iA, algnN);
  }
  else if (precC == Precision::FP128_QF) {
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f128_qf(stream, N, (float4*)acc, algnN, (int32_t*)v_exp, (complex_float4*)iA, algnN);
  }
}
