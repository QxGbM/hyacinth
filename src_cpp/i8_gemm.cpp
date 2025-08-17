
#include <hyacinth.hpp>
#include <internal.hpp>

template <Precision prec>
inline void decode_dispatcher(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, void* A, const int32_t* B, int32_t ld) {
  if constexpr(prec == Precision::FP64)
    internal::int8::decode_f64_strided_i32(stream, order_lo, order_hi, N, (double*)A, B, ld);
  else if constexpr(prec == Precision::FP32)
    internal::int8::decode_f32_strided_i32(stream, order_lo, order_hi, N, (float*)A, B, ld);
  else if constexpr(prec == Precision::FP128_DD)
    internal::int8::decode_f128_dd_strided_i32(stream, order_lo, order_hi, N, (double2*)A, B, ld);
  else if constexpr(prec == Precision::FP128_QF)
    internal::int8::decode_f128_qf_strided_i32(stream, order_lo, order_hi, N, (float4*)A, B, ld);
}

template <Precision prec>
void i8gemm_dispatcher(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, void* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  int32_t iter_k = 1 << order_k, one = 1, zero = 0;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;
    uint64_t AT_i = uint64_t(i) * strideA;
    uint64_t AN_i = uint64_t(first_iter) * strideA;

    if (algnK <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, algnK, &one, 
        &AT[AT_i], CUDA_R_8I, algnK, &A[AN_i], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      decode_dispatcher<prec>(stream, i - order_i, i, N, C, workspace, algnN);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, iter_k, &one, 
        &AT[AT_i], CUDA_R_8I, algnK, &A[AN_i], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      internal::int8::i32_normalization(stream, strideC, order_i, 0, workspace);

      for (int32_t k = iter_k; k < algnK; k += iter_k) {
        const int8_t* AT_k = &AT[uint64_t(k) + AT_i];
        const int8_t* AN_k = &A[uint64_t(k) + AN_i];

        if (k != iter_k)
          internal::int8::i32_normalization(stream, strideC, order_i, 1, workspace);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      decode_dispatcher<prec>(stream, i - order_i, i + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f64(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, double* C, int32_t* workspace) {
  i8gemm_dispatcher<Precision::FP64>(stream, handle, N, order_k, algnN, algnK, AT, A, orderA, C, workspace);
}

void internal::int8::r8i_TN_gemm_stridedA_f32(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, float* C, int32_t* workspace) {
  i8gemm_dispatcher<Precision::FP32>(stream, handle, N, order_k, algnN, algnK, AT, A, orderA, C, workspace);
}

void internal::int8::r8i_TN_gemm_stridedA_f128_dd(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, double2* C, int32_t* workspace) {
  i8gemm_dispatcher<Precision::FP128_DD>(stream, handle, N, order_k, algnN, algnK, AT, A, orderA, C, workspace);
}

void internal::int8::r8i_TN_gemm_stridedA_f128_qf(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, float4* C, int32_t* workspace) {
  i8gemm_dispatcher<Precision::FP128_QF>(stream, handle, N, order_k, algnN, algnK, AT, A, orderA, C, workspace);
}
