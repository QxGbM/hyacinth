
#include <hyacinth.hpp>
#include <internal.hpp>

void internal::int8::r8i_TN_gemm_stridedA_f64(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t order_split_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, double* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t iter_k = 1 << order_k, split_k = 1 << order_split_k, one = 1, zero = 0;
  int32_t iters = (algnK + iter_k - 1) / iter_k;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;
    uint64_t strideK = uint64_t(algnN) * uint64_t(N) * uint64_t(order_i);

    for (int32_t k = 0; k < iters; ++k) {
      int32_t k_mod = k & (split_k - 1);
      const int8_t* AT_k = &A[uint64_t(k * iter_k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k * iter_k) + uint64_t(first_iter) * strideA];
      int32_t* C_k = &workspace[uint64_t(k_mod) * strideK];

      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k * iter_k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, C_k, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (k_mod == (split_k - 1) || k == (iters - 1))
        decode_f64_strided_i32(stream, i - order_i, i, (order_k + order_split_k), k_mod + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f32(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t order_split_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, float* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t iter_k = 1 << order_k, split_k = 1 << order_split_k, one = 1, zero = 0;
  int32_t iters = (algnK + iter_k - 1) / iter_k;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;
    uint64_t strideK = uint64_t(algnN) * uint64_t(N) * uint64_t(order_i);

    for (int32_t k = 0; k < iters; ++k) {
      int32_t k_mod = k & (split_k - 1);
      const int8_t* AT_k = &A[uint64_t(k * iter_k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k * iter_k) + uint64_t(first_iter) * strideA];
      int32_t* C_k = &workspace[uint64_t(k_mod) * strideK];

      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k * iter_k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, C_k, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (k_mod == (split_k - 1) || k == (iters - 1))
        decode_f32_strided_i32(stream, i - order_i, i, (order_k + order_split_k), k_mod + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f128_dd(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t order_split_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, double2* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t iter_k = 1 << order_k, split_k = 1 << order_split_k, one = 1, zero = 0;
  int32_t iters = (algnK + iter_k - 1) / iter_k;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;
    uint64_t strideK = uint64_t(algnN) * uint64_t(N) * uint64_t(order_i);

    for (int32_t k = 0; k < iters; ++k) {
      int32_t k_mod = k & (split_k - 1);
      const int8_t* AT_k = &A[uint64_t(k * iter_k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k * iter_k) + uint64_t(first_iter) * strideA];
      int32_t* C_k = &workspace[uint64_t(k_mod) * strideK];

      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k * iter_k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, C_k, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (k_mod == (split_k - 1) || k == (iters - 1))
        decode_f128_dd_strided_i32(stream, i - order_i, i, (order_k + order_split_k), k_mod + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f128_qf(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t order_split_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, float4* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t iter_k = 1 << order_k, split_k = 1 << order_split_k, one = 1, zero = 0;
  int32_t iters = (algnK + iter_k - 1) / iter_k;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;
    uint64_t strideK = uint64_t(algnN) * uint64_t(N) * uint64_t(order_i);

    for (int32_t k = 0; k < iters; ++k) {
      int32_t k_mod = k & (split_k - 1);
      const int8_t* AT_k = &A[uint64_t(k * iter_k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k * iter_k) + uint64_t(first_iter) * strideA];
      int32_t* C_k = &workspace[uint64_t(k_mod) * strideK];

      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k * iter_k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &zero, C_k, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      if (k_mod == (split_k - 1) || k == (iters - 1))
        decode_f128_qf_strided_i32(stream, i - order_i, i, (order_k + order_split_k), k_mod + 1, N, C, workspace, algnN);
    }
  }
}
