
#include <hyacinth.hpp>

#include <complex>
#include <cuComplex.h>
#include <cub/cub.cuh>
#include <float4.hpp>

struct add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::f4::add(a, b); }
};

struct minus_conj_a_fma_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y))); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, cuComplex c) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::f4::fma(complex_float4(device::f4::negate(a.real), a.imag), b, c); }
};

struct minus_complex_norm {
  __device__ __forceinline__ double operator()(double a, cuDoubleComplex b) {
    return fma(-b.x, b.x, fma(-b.y, b.y, a)); }
  __device__ __forceinline__ float operator()(float a, cuComplex b) {
    return fmaf(-b.x, b.x, fmaf(-b.y, b.y, a)); }
  __device__ __forceinline__ float4 operator()(float4 a, complex_float4 b) { 
    return device::f4::fma(device::f4::negate(b.real), b.real, device::f4::fma(device::f4::negate(b.imag), b.imag, a)); }
};

struct init_complex {
  __device__ __forceinline__ operator cuDoubleComplex() { return make_cuDoubleComplex(0., 0.); }
  __device__ __forceinline__ operator cuComplex() { return make_cuComplex(0.f, 0.f); }
  __device__ __forceinline__ operator complex_float4() { return complex_float4(make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)); }
};

struct scal_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, double b) { return make_cuDoubleComplex(a.x * b, a.y * b); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, float b) { return make_cuComplex(a.x * b, a.y * b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, float4 b) {
    return complex_float4(device::f4::fma(a.real, b, make_float4(0.f, 0.f, 0.f, 0.f)), device::f4::fma(a.imag, b, make_float4(0.f, 0.f, 0.f, 0.f))); }
};

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::f4::conj(f); }
};

template <class real_t, class real_ptr, class real_const_ptr, class complex_t, class complex_ptr, class complex_const_ptr, int32_t BLOCK_WARPS, int32_t ITEMS_PER_THREAD>
__global__ void minus_adjAx_plusB_scale_complex(real_const_ptr scale, int32_t M, int32_t N, complex_const_ptr A, int32_t lda, complex_ptr B, real_ptr C) {
  using WarpLoad = cub::WarpLoad<complex_t, ITEMS_PER_THREAD>;
  using WarpReduce = cub::WarpReduce<complex_t>;
  constexpr int32_t elements = ITEMS_PER_THREAD << 5;

  __shared__ typename WarpLoad::TempStorage temp_loadA[BLOCK_WARPS], temp_loadX[BLOCK_WARPS];
  __shared__ typename WarpReduce::TempStorage temp_reduce[BLOCK_WARPS];

  add_complex add_func;
  minus_conj_a_fma_complex fma_func;

  complex_t thread_A[ITEMS_PER_THREAD], thread_X[ITEMS_PER_THREAD], thread_B[ITEMS_PER_THREAD], init = init_complex();
  int32_t row = blockIdx.x * BLOCK_WARPS + threadIdx.y;

  if (row < M) {
    complex_const_ptr A_i = &A[row * lda];

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      thread_B[i] = init;

    for (int32_t i = 0; i < N; i += elements) {
      int32_t num_items = min(elements, N - i);
      WarpLoad(temp_loadA[threadIdx.y]).Load(&A_i[i], thread_A, num_items, init);
      WarpLoad(temp_loadX[threadIdx.y]).Load(&A[i], thread_X, num_items, init);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        thread_B[j] = fma_func(thread_A[j], thread_X[j], thread_B[j]);
    }

    complex_t thread_res = thread_B[0];
    #pragma unroll
    for (int32_t i = 1; i < ITEMS_PER_THREAD; ++i)
      thread_res = add_func(thread_res, thread_B[i]);

    complex_t warp_res = WarpReduce(temp_reduce[threadIdx.y]).Reduce(thread_res, add_func);

    if (threadIdx.x == 0) {
      scal_complex scal_func;
      minus_complex_norm fnrm_func;
      conj conj_func;

      complex_t res = scal_func(add_func(warp_res, B[row]), *scale);
      B[row] = res;
      B[row * lda] = conj_func(res);
      C[row] = fnrm_func(C[row], res);
    }
  }
}

void minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B, double* C) {
  constexpr int32_t block_warps = 4;
  constexpr int32_t items_per_thread = 4;

  int32_t grid_size = (M + block_warps - 1) / block_warps;
  minus_adjAx_plusB_scale_complex <double, double* __restrict__, const double* __restrict__,
    cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, block_warps, items_per_thread>
    <<< grid_size, dim3(32, block_warps, 1), 0, stream >>> (scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B, C);
}

