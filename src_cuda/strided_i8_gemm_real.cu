
#include <hyacinth.hpp>
#include <internal.hpp>
#include <algorithm>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t beta, uint32_t N, uint32_t BASE> struct normalize_i32 {
  int4* A;
  uint64_t stride;
  normalize_i32(uint64_t M, int32_t* A) : A((int4*)A), stride(M >> 2) {}

  __device__ __forceinline__ void operator()(uint64_t i) {
    constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;

    int4 A_i = A[i];
    A[i] = make_int4(A_i.x & iBASE, A_i.y & iBASE, A_i.z & iBASE, A_i.w & iBASE);
    A_i = make_int4(A_i.x >> BASE, A_i.y >> BASE, A_i.z >> BASE, A_i.w >> BASE);

    #pragma unroll
    for (uint32_t k = 1; k < N; ++k) {
      uint64_t j = i + uint64_t(k) * stride;
      int4 A_k = A[j];
      int4 val = make_int4(A_i.x + A_k.x, A_i.y + A_k.y, A_i.z + A_k.z, A_i.w + A_k.w);

      A_k.x = val.x & iBASE;
      A_k.y = val.y & iBASE;
      A_k.z = val.z & iBASE;
      A_k.w = val.w & iBASE;
      A[j] = A_k;

      A_i.x = val.x >> BASE;
      A_i.y = val.y >> BASE;
      A_i.z = val.z >> BASE;
      A_i.w = val.w >> BASE;
    }

    uint64_t j = i + uint64_t(N) * stride;
    if constexpr (beta) {
      int4 A_k = A[j];
      A[j] = make_int4(A_i.x + A_k.x, A_i.y + A_k.y, A_i.z + A_k.z, A_i.w + A_k.w);
    }
    else
      A[j] = A_i;
  }
};

template <uint32_t beta>
void normalization_dispatcher(cudaStream_t stream, uint64_t M, int32_t order, int32_t* A) {
  thrust::counting_iterator<uint64_t> iter(0);
  auto policy = thrust::cuda::par_nosync.on(stream);
  switch(order) {
    case 1: { normalize_i32<beta, 1, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 2: { normalize_i32<beta, 2, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 3: { normalize_i32<beta, 3, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 4: { normalize_i32<beta, 4, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 5: { normalize_i32<beta, 5, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 6: { normalize_i32<beta, 6, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 7: { normalize_i32<beta, 7, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 8: { normalize_i32<beta, 8, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 9: { normalize_i32<beta, 9, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 10: { normalize_i32<beta, 10, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 11: { normalize_i32<beta, 11, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 12: { normalize_i32<beta, 12, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 13: { normalize_i32<beta, 13, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 14: { normalize_i32<beta, 14, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    case 15: { normalize_i32<beta, 15, device::Config::exp_base> normalize(M, A);
      thrust::for_each_n(policy, iter, normalize.stride, normalize); break; }
    default: break;
  }
}

void internal::int8::r8i_TN_gemm_stridedA(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t iter_k, int32_t algnN, int32_t algnK, const int8_t* AT, const int8_t* A, int32_t orderA, int32_t* C, int32_t orderC) {
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  int32_t order_begin = orderC - 2 * orderA;
  int32_t one = 1;
  
  for (int32_t i = 0; i < orderA; ++i) {
    std::pair<int32_t, int32_t> min_max = std::minmax(order_begin + i, 0);
    int32_t order_i = orderA + min_max.first;
    int32_t* C_i = &C[uint64_t(min_max.second) * strideC];

    for (int32_t k = 0; k < algnK; k += iter_k) {
      const int8_t* AT_k = &AT[uint64_t(k) + uint64_t(i) * strideA];
      const int8_t* AN_k = &A[uint64_t(k) + uint64_t(orderA - order_i) * strideA];
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
        AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, C_i, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

      normalization_dispatcher<1>(stream, strideC, order_i, C_i);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f64(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, double* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  int32_t iter_k = 1 << order_k, one = 1, zero = 0;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;

    if (algnK <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, algnK, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      decode_f64_strided_i32(stream, i - order_i, i, N, C, workspace, algnN);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, iter_k, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      normalization_dispatcher<0>(stream, strideC, order_i, workspace);

      for (int32_t k = iter_k; k < algnK; k += iter_k) {
        const int8_t* AT_k = &A[uint64_t(k) + uint64_t(i) * strideA];
        const int8_t* AN_k = &A[uint64_t(k) + uint64_t(first_iter) * strideA];

        if (k != iter_k)
          normalization_dispatcher<1>(stream, strideC, order_i, workspace);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      decode_f64_strided_i32(stream, i - order_i, i + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f32(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, float* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  int32_t iter_k = 1 << order_k, one = 1, zero = 0;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;

    if (algnK <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, algnK, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      decode_f32_strided_i32(stream, i - order_i, i, N, C, workspace, algnN);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, iter_k, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      normalization_dispatcher<0>(stream, strideC, order_i, workspace);

      for (int32_t k = iter_k; k < algnK; k += iter_k) {
        const int8_t* AT_k = &A[uint64_t(k) + uint64_t(i) * strideA];
        const int8_t* AN_k = &A[uint64_t(k) + uint64_t(first_iter) * strideA];

        if (k != iter_k)
          normalization_dispatcher<1>(stream, strideC, order_i, workspace);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      decode_f32_strided_i32(stream, i - order_i, i + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f128_dd(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, double2* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  int32_t iter_k = 1 << order_k, one = 1, zero = 0;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;

    if (algnK <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, algnK, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      decode_f128_dd_strided_i32(stream, i - order_i, i, N, C, workspace, algnN);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, iter_k, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      normalization_dispatcher<0>(stream, strideC, order_i, workspace);

      for (int32_t k = iter_k; k < algnK; k += iter_k) {
        const int8_t* AT_k = &A[uint64_t(k) + uint64_t(i) * strideA];
        const int8_t* AN_k = &A[uint64_t(k) + uint64_t(first_iter) * strideA];

        if (k != iter_k)
          normalization_dispatcher<1>(stream, strideC, order_i, workspace);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      decode_f128_dd_strided_i32(stream, i - order_i, i + 1, N, C, workspace, algnN);
    }
  }
}

void internal::int8::r8i_TN_gemm_stridedA_f128_qf(cudaStream_t stream, cublasHandle_t handle, int32_t N, int32_t order_k, int32_t algnN, int32_t algnK, const int8_t* A, int32_t orderA, float4* C, int32_t* workspace) {
  uint64_t strideA = uint64_t(algnK) * uint64_t(N);
  uint64_t strideC = uint64_t(algnN) * uint64_t(N);
  int32_t iter_k = 1 << order_k, one = 1, zero = 0;
  
  for (int32_t i = 0; i < orderA; ++i) {
    int32_t first_iter = int32_t(i == 0);
    int32_t order_i = orderA - first_iter;

    if (algnK <= iter_k) {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, algnK, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      decode_f128_qf_strided_i32(stream, i - order_i, i, N, C, workspace, algnN);
    }
    else {
      cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, iter_k, &one, 
        &A[uint64_t(i) * strideA], CUDA_R_8I, algnK, &A[uint64_t(first_iter) * strideA], CUDA_R_8I, algnK, &zero, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      normalization_dispatcher<0>(stream, strideC, order_i, workspace);

      for (int32_t k = iter_k; k < algnK; k += iter_k) {
        const int8_t* AT_k = &A[uint64_t(k) + uint64_t(i) * strideA];
        const int8_t* AN_k = &A[uint64_t(k) + uint64_t(first_iter) * strideA];

        if (k != iter_k)
          normalization_dispatcher<1>(stream, strideC, order_i, workspace);
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, algnN, N * order_i, std::min(algnK - k, iter_k), &one, 
          AT_k, CUDA_R_8I, algnK, AN_k, CUDA_R_8I, algnK, &one, workspace, CUDA_R_32I, algnN, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
      }
      decode_f128_qf_strided_i32(stream, i - order_i, i + 1, N, C, workspace, algnN);
    }
  }
}

