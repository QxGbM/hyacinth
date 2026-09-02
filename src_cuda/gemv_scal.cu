
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_f128 {
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::make_complex_double2(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::make_complex_float4(operator()(a.real, b.real), operator()(a.imag, b.imag)); }
};

template <class T> __device__ __forceinline__ T load_const(const T* a) { return __ldg(a); }
template <> __device__ __forceinline__ complex_double2 load_const<complex_double2>(const complex_double2* a) { return device::dd::make_complex_double2(__ldg(&a->real), __ldg(&a->imag)); }
template <> __device__ __forceinline__ complex_float4 load_const<complex_float4>(const complex_float4* a) { return device::qf::make_complex_float4(__ldg(&a->real), __ldg(&a->imag)); }

__device__ __forceinline__ double2 fma_(const double2 a, const double2 b, double2 c) { return device::dd::add(c, device::dd::mul(a, b)); }
__device__ __forceinline__ float4 fma_(const float4 a, const float4 b, float4 c) { return device::qf::add(c, device::qf::mul(a, b)); }
__device__ __forceinline__ complex_double2 fma_(const complex_double2 a, const complex_double2 b, complex_double2 c) {
  using device::dd::add, device::dd::mul, device::dd::negate, device::dd::make_complex_double2;
  return make_complex_double2(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
}
__device__ __forceinline__ complex_float4 fma_(const complex_float4 a, const complex_float4 b, complex_float4 c) {
  using device::qf::add, device::qf::mul, device::qf::negate, device::qf::make_complex_float4;
  return make_complex_float4(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
}

__device__ __forceinline__ double2 neg_(double2 a) { return device::dd::negate(a); }
__device__ __forceinline__ float4 neg_(float4 a) { return device::qf::negate(a); }
__device__ __forceinline__ complex_double2 neg_(complex_double2 a) { return device::dd::make_complex_double2(device::dd::negate(a.real), device::dd::negate(a.imag)); }
__device__ __forceinline__ complex_float4 neg_(complex_float4 a) { return device::qf::make_complex_float4(device::qf::negate(a.real), device::qf::negate(a.imag)); }

template <int32_t WARP_THREADS, int32_t BLOCK_WARPS, class matrix_t>
__global__ void gemv_kernel(int32_t M, int32_t N, matrix_t* __restrict__ A, int64_t ldj, int64_t lda) {
  __shared__ typename cub::BlockReduce<matrix_t, WARP_THREADS>::TempStorage temp_reduce[BLOCK_WARPS];

  int32_t i = BLOCK_WARPS * blockIdx.x + threadIdx.y;
  const matrix_t* A_i = &A[int64_t(i) * lda], *A_j = &A[ldj];

  if (i < N && A_i != A_j) {
    matrix_t threadB = matrix_t();
    for (int32_t k = threadIdx.x; k < M; k += WARP_THREADS)
      threadB = fma_(load_const(&A_i[k]), load_const(&A_j[k]), threadB);

    add_f128 add_;
    threadB = cub::BlockReduce<matrix_t, WARP_THREADS>(temp_reduce[threadIdx.y]).Reduce(threadB, add_);
    A = &A[ldj + int64_t(i + M)];

    if (threadIdx.x == 0)
      *A = add_(neg_(threadB), *A);
  }
}

template <class real_t, class matrix_t, class idx_t>
inline void gemv_dispatcher(cudaStream_t stream, cublasHandle_t handle, idx_t* scale, int32_t j, int32_t M, int32_t N, matrix_t* A, int32_t lda, int32_t* jpiv, real_t* D) {
  if (1 < N) {
    if (0 < M) {
      int64_t lda64 = int64_t(lda), ldj = int64_t(j) * lda64;
      if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>)
      { double one = 1., minus_one = -1.; cublasDgemv(handle, CUBLAS_OP_T, M, N, &minus_one, A, lda, &A[ldj], 1, &one, &A[int64_t(M) + ldj], 1); }
      else if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>)
      { float one = 1.f, minus_one = -1.f; cublasSgemv(handle, CUBLAS_OP_T, M, N, &minus_one, A, lda, &A[ldj], 1, &one, &A[int64_t(M) + ldj], 1); }
      else if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>)
      { cuDoubleComplex one = make_cuDoubleComplex(1., 0.), minus_one = make_cuDoubleComplex(-1., 0.); cublasZgemv(handle, CUBLAS_OP_C, M, N, &minus_one, A, lda, &A[ldj], 1, &one, &A[int64_t(M) + ldj], 1); }
      else if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>)
      { cuComplex one = make_cuComplex(1.f, 0.f), minus_one = make_cuComplex(-1.f, 0.f); cublasCgemv(handle, CUBLAS_OP_C, M, N, &minus_one, A, lda, &A[ldj], 1, &one, &A[int64_t(M) + ldj], 1); }
      else {
        constexpr uint32_t target_blocks = uint32_t(512);
        constexpr dim3 conf0(32, 16, 1), conf1(64, 8, 1), conf2(128, 4, 1), conf3(256, 2, 1), conf4(512, 1, 1);
        uint32_t grid[]{ uint32_t(N + 15) >> 4, uint32_t(N + 7) >> 3, uint32_t(N + 3) >> 2, uint32_t(N + 1) >> 1 };

        if (target_blocks <= grid[0] || M <= int32_t(conf0.x)) { gemv_kernel<conf0.x, conf0.y> <<< grid[0], conf0, 0, stream >>> (M, N, A, ldj, lda64); } else
        if (target_blocks <= grid[1] || M <= int32_t(conf1.x)) { gemv_kernel<conf1.x, conf1.y> <<< grid[1], conf1, 0, stream >>> (M, N, A, ldj, lda64); } else
        if (target_blocks <= grid[2] || M <= int32_t(conf2.x)) { gemv_kernel<conf2.x, conf2.y> <<< grid[2], conf2, 0, stream >>> (M, N, A, ldj, lda64); } else
        if (target_blocks <= grid[3] || M <= int32_t(conf3.x)) { gemv_kernel<conf3.x, conf3.y> <<< grid[3], conf3, 0, stream >>> (M, N, A, ldj, lda64); } else
        { gemv_kernel<conf4.x, conf4.y> <<< N, conf4, 0, stream >>> (M, N, A, ldj, lda64); }
      }
    }
    internal::Cholesky::gemv_pp(stream, scale, j, M, N, A, lda, jpiv, D);
  } else if (1 == N) {
    if constexpr(std::is_same_v<real_t, matrix_t>) { cudaMemcpyAsync(&A[M], scale, sizeof(real_t), cudaMemcpyHostToDevice, stream); }
      else { cudaMemsetAsync(&((real_t*)scale)[1], 0, sizeof(real_t), stream); cudaMemcpyAsync(&A[M], scale, sizeof(matrix_t), cudaMemcpyHostToDevice, stream); }
  }
}

namespace internal::Cholesky {

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double_idx* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float_idx* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

  void gemv_scal(cudaStream_t stream, cublasHandle_t handle, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_dispatcher(stream, handle, scale, j, M, N, A, lda, jpiv, D); }

}
