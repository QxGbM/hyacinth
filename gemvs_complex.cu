
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

struct minus_conj_a_mul_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) {
    return make_cuDoubleComplex(fma(-a.x, b.x, -a.y * b.y), fma(-a.x, b.y, a.y * b.x)); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) {
    return make_cuComplex(fmaf(-a.x, b.x, -a.y * b.y), fmaf(-a.x, b.y, a.y * b.x)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { 
    return device::dd::mul(device::dd::make_complex_double2(device::dd::negate(a.real), a.imag), b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { 
    return device::qf::mul(device::qf::make_complex_float4(device::qf::negate(a.real), a.imag), b); }
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

template <class real_t, class complex_t, class complex_ptr, class complex_const_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void minus_adjAx_plusB_scale_complex(real_t scale, int32_t M, int32_t N, complex_const_ptr A, int32_t lda, complex_ptr B) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<complex_t, BLOCK_THREADS>::TempStorage temp_reduce;
  complex_t threadA[ITEMS_PER_THREAD], threadX[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<complex_t, BLOCK_THREADS> block_reduce(temp_reduce);
  add_complex add_func;
  minus_conj_a_mul_complex mul_func;
  minus_conj_a_fma_complex fma_func;

  for (int32_t row = blockIdx.x; row < M; row += GRID_BLOCKS) {
    complex_const_ptr A_i = &A[row * lda];

    block_load.Load(A_i, threadA, N1, complex_t());
    block_load.Load(A, threadX, N1, complex_t());

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      threadB[i] = mul_func(threadA[i], threadX[i]);

    for (int32_t i = elements; i < N1; i += elements) {
      block_load.Load(&A_i[i], threadA);
      block_load.Load(&A[i], threadX);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        threadB[j] = fma_func(threadA[j], threadX[j], threadB[j]);
    }

    if (0 < N2) {
      block_load.Load(&A_i[N1], threadA, N2, complex_t());
      block_load.Load(&A[N1], threadX, N2, complex_t());

      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        threadB[i] = fma_func(threadA[i], threadX[i], threadB[i]);
    }

    complex_t block_res = block_reduce.Reduce(threadB, add_func);
    __syncthreads();

    if (threadIdx.x == 0) {
      scal_add_complex scal_func;
      conj conj_func;

      complex_t res = scal_func(block_res, B[row], scale);
      B[row] = res;
      B[row * lda] = conj_func(res);
    }
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t block_threads = block_warps * 32;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t grid_warps = grid_blocks * block_warps;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::minus_adjAx_plusB_scale_double_complex(cudaStream_t stream, const double scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_adjAx_plusB_scale_complex <double, cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B);
  else
    minus_adjAx_plusB_scale_complex <double, cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, grid_warps, 32, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float_complex(cudaStream_t stream, const float scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_adjAx_plusB_scale_complex <float, cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (scale, M, N, (const cuComplex*)A, lda, (cuComplex*)B);
  else
    minus_adjAx_plusB_scale_complex <float, cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__, grid_warps, 32, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, (const cuComplex*)A, lda, (cuComplex*)B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_adjAx_plusB_scale_complex <double2, complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_adjAx_plusB_scale_complex <double2, complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__, grid_warps, 32, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, A, lda, B);
}

void internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* B) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  constexpr int32_t call_reduce = 64 * block_warps * items_per_thread;

  if (call_reduce < N)
    minus_adjAx_plusB_scale_complex <float4, complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (scale, M, N, A, lda, B);
  else
    minus_adjAx_plusB_scale_complex <float4, complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__, grid_warps, 32, items_per_thread>
      <<< grid_warps, 32, 0, stream >>> (scale, M, N, A, lda, B);
}
