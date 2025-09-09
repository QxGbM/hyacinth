
#include <hyacinth.hpp>
#include <internal.hpp>

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
void i8gemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t iter_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, void* C, int32_t* workspace) {
  int64_t strideA = int64_t(algnK) * int64_t(N);
  int32_t one = 1, zero = 0;
  
  if (algnK <= iter_k)
    for (int32_t i = 0; i < orderA; ++i) {
      int64_t AT_i = int64_t(i) * strideA;
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, algnK, &one, 
        &AT[AT_i], CUDA_R_8I, algnK, A, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      dequantize_dispatcher<prec>(stream, i - orderA, i, N, C, workspace, algnN);
    }
  else {
    int32_t iter_h = iter_k >> 1;
    int32_t rem = algnK % iter_k;
    rem = (rem == 0) ? iter_k : (rem < iter_h ? (rem + iter_k) : rem);
    int32_t range_k = algnK - rem;

    for (int32_t i = 0; i < orderA; ++i) {
      int64_t AT_i = int64_t(i) * strideA;
      for (int32_t k = 0; k < range_k; k += iter_k) {
        const int8_t* AT_k = &AT[int64_t(k) + AT_i];
        const int8_t* AN_k = &A[int64_t(k)];

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, iter_k, &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        dequantize_dispatcher<prec>(stream, i - orderA, i, N, C, workspace, algnN);
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
        dequantize_dispatcher<prec>(stream, i - orderA, i, N, C, workspace, algnN);

        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * orderA, rem - iter_h, &one, 
          AT_k2, CUDA_R_8I, algnK, AN_k2, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      dequantize_dispatcher<prec>(stream, i - orderA, i, N, C, workspace, algnN);
    }
  }
}

inline void quantize_dispatcher_real(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const void* C, int32_t ldc, device::Precision prec, int32_t* vec_expon, int8_t* A, int32_t lda) {
  if (prec == device::Precision::FP64) {
    internal::int8::vexp_f64(stream, device::Config::exp_base * order, M, N, (const double*)C, ldc, vec_expon);
    internal::int8::quantize_f64(stream, order, M, N, (const double*)C, ldc, vec_expon, A, lda);
  }
  else if (prec == device::Precision::FP32) {
    internal::int8::vexp_f32(stream, device::Config::exp_base * order, M, N, (const float*)C, ldc, vec_expon);
    internal::int8::quantize_f32(stream, order, M, N, (const float*)C, ldc, vec_expon, A, lda);
  }
}

void device::MixPrecAHA::rATA(cudaStream_t stream, cublasHandle_t handle, gemm_params param, const void* A, int32_t lda, void* C) {
  int32_t M = param.M, N = param.N;
  int32_t algnM = param.algnM, algnN = param.algnN;
  int32_t orderA = param.orderA;

  int8_t* acc = (int8_t*)(C), *iA = &acc[param.acc_bytes];
  int8_t* v_exp = &iA[param.i8_bytes], *workspace = &v_exp[param.exp_bytes];
  cudaMemsetAsync(acc, 0, param.acc_bytes + param.i8_bytes, stream);
  quantize_dispatcher_real(stream, orderA, M, N, A, lda, param.precA, (int32_t*)v_exp, iA, algnM);

  if (param.precC == Precision::FP64) {
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f64(stream, N, (double*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp);
  }
  else if (param.precC == Precision::FP32) {
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f32(stream, N, (float*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp);
  }
  else if (param.precC == Precision::FP128_DD) {
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f128_dd(stream, N, (double2*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp);
  }
  else if (param.precC == Precision::FP128_QF) {
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    internal::int8::scal_exponent_f128_qf(stream, N, (float4*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp);
  }
}

inline void quantize_dispatcher_complex(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const void* C, int32_t ldc, device::Precision prec, int32_t* vec_expon, int8_t* A, int32_t lda) {
  if (prec == device::Precision::FP64) {
    internal::int8::vexp_f64(stream, device::Config::exp_base * order, 2 * M, N, (const double*)C, 2 * ldc, vec_expon);
    internal::int8::quantize_cf64(stream, order, M, N, (const std::complex<double>*)C, ldc, vec_expon, A, lda);
  }
  else if (prec == device::Precision::FP32) {
    internal::int8::vexp_f32(stream, device::Config::exp_base * order, 2 * M, N, (const float*)C, 2 * ldc, vec_expon);
    internal::int8::quantize_cf32(stream, order, M, N, (const std::complex<float>*)C, ldc, vec_expon, A, lda);
  }
}

void device::MixPrecAHA::cAHA(cudaStream_t stream, cublasHandle_t handle, gemm_params param, const void* A, int32_t lda, void* C) {
  int32_t M = param.M, N = param.N;
  int32_t algnM = param.algnM, algnN = param.algnN;
  int32_t orderA = param.orderA;

  int8_t* iA = (int8_t*)(C), *workspace = &iA[param.i8_bytes];
  int8_t* acc = &workspace[param.scratch_bytes], *v_exp = &acc[param.acc_bytes];
  int8_t* iA_imag = &iA[int64_t(orderA) * int64_t(algnM) * int64_t(N)];
  int8_t* acc_imag = &acc[int64_t(param.C_elem_bytes >> 1) * int64_t(algnN) * int64_t(N)];
  cudaMemsetAsync(iA, 0, param.i8_bytes, stream);
  cudaMemsetAsync(acc, 0, param.acc_bytes, stream);
  quantize_dispatcher_complex(stream, orderA, M, N, A, lda, param.precA, (int32_t*)v_exp, iA, algnM);

  if (param.precC == Precision::FP64) {
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, param.iter_k, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP64>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f64(stream, N, (double*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp, (std::complex<double>*)iA, algnN);
  }
  else if (param.precC == Precision::FP32) {
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, param.iter_k, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP32>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f32(stream, N, (float*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp, (std::complex<float>*)iA, algnN);
  }
  else if (param.precC == Precision::FP128_DD) {
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, param.iter_k, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f128_dd(stream, N, (double2*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp, (complex_double2*)iA, algnN);
  }
  else if (param.precC == Precision::FP128_QF) {
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, param.iter_k, algnN, algnM, iA_imag, iA_imag, orderA, acc, (int32_t*)workspace);
    i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, param.iter_k, algnN, algnM, iA, iA_imag, orderA, acc_imag, (int32_t*)workspace);
    internal::int8::planar_to_interleave_f128_qf(stream, N, (float4*)acc, algnN, device::Config::exp_base * orderA, (int32_t*)v_exp, (complex_float4*)iA, algnN);
  }
}
