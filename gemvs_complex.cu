
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::add(a, b); }
};

struct minus_conj_a_fma_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    return make_cuDoubleComplex(fma(-a.x, b.x, fma(-a.y, b.y, c.x)), fma(-a.x, b.y, fma(a.y, b.x, c.y))); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, cuComplex c) {
    return make_cuComplex(fmaf(-a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(-a.x, b.y, fmaf(a.y, b.x, c.y))); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) { 
    return device::dd::fma(device::dd::make_complex_double2(device::dd::negate(a.real), a.imag), b, c); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::qf::fma(device::qf::make_complex_float4(device::qf::negate(a.real), a.imag), b, c); }
};

struct scal_add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, double s) { return make_cuDoubleComplex(s * (a.x + b.x), s * (a.y + b.y)); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, float s) { return make_cuComplex(s * (a.x + b.x), s * (a.y + b.y)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, double2 s) { 
    return device::dd::make_complex_double2(device::dd::mul(s, device::dd::add(a.real, b.real)), device::dd::mul(s, device::dd::add(a.imag, b.imag))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, float4 s) { 
    return device::qf::make_complex_float4(device::qf::mul(s, device::qf::add(a.real, b.real)), device::qf::mul(s, device::qf::add(a.imag, b.imag))); }
};

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::conj(f); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::conj(f); }
};

template <class real_t, class real_ptr, class real_const_ptr, class complex_t, class complex_ptr, class complex_const_ptr, int32_t GRID_WARPS, int32_t BLOCK_WARPS, int32_t ITEMS_PER_THREAD>
__global__ void minus_adjAx_plusB_scale_complex(real_const_ptr scale, int32_t M, int32_t N, complex_const_ptr A, int32_t lda, complex_ptr B) {

  __shared__ typename cub::WarpLoad<complex_t, ITEMS_PER_THREAD>::TempStorage temp_load[BLOCK_WARPS];
  __shared__ typename cub::BlockReduce<complex_t, 32>::TempStorage temp_reduce[BLOCK_WARPS];
  complex_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  constexpr int32_t elements = ITEMS_PER_THREAD * 32;
  int32_t warp_id = int32_t(threadIdx.x) >> 5;
  cub::WarpLoad<complex_t, ITEMS_PER_THREAD> warp_load(temp_load[warp_id]);
  cub::BlockReduce<complex_t, 32> warp_reduce(temp_reduce[warp_id]);
  add_complex add_func;
  minus_conj_a_fma_complex fma_func;

  for (int32_t row = blockIdx.x * BLOCK_WARPS + warp_id; row < M; row += GRID_WARPS) {
    complex_const_ptr A_i = &A[row * lda];

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      threadB[i] = complex_t();

    for (int32_t i = 0; i < N; i += elements) {
      int32_t num_items = min(elements, N - i);
      warp_load.Load(&A_i[i], threadA, num_items, complex_t());
      warp_load.Load(&A[i], threadX, num_items, complex_t());

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        threadB[j] = fma_func(threadA[j], threadX[j], threadB[j]);
    }

    complex_t warp_res = warp_reduce.Reduce(threadB, add_func);

    if ((threadIdx.x & 31) == 0) {
      scal_add_complex scal_func;
      conj conj_func;

      complex_t res = scal_func(warp_res, B[row], *scale);
      B[row] = res;
      B[row * lda] = conj_func(res);
    }
  }
}

template <class real_t, class real_ptr, class real_const_ptr, class complex_t, class complex_ptr, class complex_const_ptr, int32_t GRID_WARPS, int32_t BLOCK_WARPS, int32_t ITEMS_PER_THREAD>
__global__ void minus_adjAx_plusB_scale_complex_reduce(real_const_ptr scale, int32_t M, int32_t N, complex_const_ptr A, int32_t lda, complex_ptr B) {
  constexpr int32_t BLOCK_THREADS = BLOCK_WARPS * 32;
  constexpr int32_t THREAD_BLOCKS = GRID_WARPS / BLOCK_WARPS;
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;

  __shared__ typename cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<complex_t, BLOCK_THREADS>::TempStorage temp_reduce;
  complex_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<complex_t, BLOCK_THREADS> block_reduce(temp_reduce);
  add_complex add_func;
  minus_conj_a_fma_complex fma_func;

  for (int32_t row = blockIdx.x; row < M; row += THREAD_BLOCKS) {
    complex_const_ptr A_i = &A[row * lda];

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      threadB[i] = complex_t();

    for (int32_t i = 0; i < N; i += elements) {
      int32_t num_items = min(elements, N - i);
      block_load.Load(&A_i[i], threadA, num_items, complex_t());
      block_load.Load(&A[i], threadX, num_items, complex_t());

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        threadB[j] = fma_func(threadA[j], threadX[j], threadB[j]);
    }

    complex_t block_res = block_reduce.Reduce(threadB, add_func);

    if (threadIdx.x == 0) {
      scal_add_complex scal_func;
      conj conj_func;

      complex_t res = scal_func(block_res, B[row], *scale);
      B[row] = res;
      B[row * lda] = conj_func(res);
    }
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t grid_size = 1024;
constexpr int32_t grid_warps = grid_size * block_warps;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce <= N)
    minus_adjAx_plusB_scale_complex_reduce <double, double* __restrict__, const double* __restrict__,
      cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B);
  else
    minus_adjAx_plusB_scale_complex <double, double* __restrict__, const double* __restrict__,
      cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float_complex(cudaStream_t stream, const float* scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce <= N)
    minus_adjAx_plusB_scale_complex_reduce <float, float* __restrict__, const float* __restrict__,
      cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, (const cuComplex*)A, lda, (cuComplex*)B);
  else
    minus_adjAx_plusB_scale_complex <float, float* __restrict__, const float* __restrict__,
      cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, (const cuComplex*)A, lda, (cuComplex*)B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(cudaStream_t stream, const double2* scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce <= N)
    minus_adjAx_plusB_scale_complex_reduce <double2, double2* __restrict__, const double2* __restrict__,
      complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_adjAx_plusB_scale_complex <double2, double2* __restrict__, const double2* __restrict__,
      complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(cudaStream_t stream, const float4* scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce <= N)
    minus_adjAx_plusB_scale_complex_reduce <float4, float4* __restrict__, const float4* __restrict__,
      complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_adjAx_plusB_scale_complex <float4, float4* __restrict__, const float4* __restrict__,
      complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__, grid_warps, block_warps, items_per_thread>
      <<< grid_size, block_warps * 32, 0, stream >>> (scale, M, N, A, lda, B);
}
