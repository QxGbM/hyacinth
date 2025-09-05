
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

template <class real_t, class matrix_t> struct gemv_pp_fused {
  __device__ __forceinline__ void operator()(double c, double& c_conj, double& d) {
    c_conj = c; d = fma(-c, c, d);
  }
  __device__ __forceinline__ void operator()(float c, float& c_conj, float& d) {
    c_conj = c; d = fmaf(-c, c, d);
  }
  __device__ __forceinline__ void operator()(double2 c, double2& c_conj, double2& d) {
    c_conj = c; d = device::dd::add(device::dd::mul(device::dd::negate(c), c), d);
  }
  __device__ __forceinline__ void operator()(float4 c, float4& c_conj, float4& d) {
    c_conj = c; d = device::qf::add(device::qf::mul(device::qf::negate(c), c), d);
  }
  __device__ __forceinline__ void operator()(cuDoubleComplex c, cuDoubleComplex& c_conj, double& d) {
    c_conj = make_cuDoubleComplex(c.x, -c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d));
  }
  __device__ __forceinline__ void operator()(cuComplex c, cuComplex& c_conj, float& d) {
    c_conj = make_cuComplex(c.x, -c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d));
  }
  __device__ __forceinline__ void operator()(complex_double2 c, complex_double2& c_conj, double2& d) {
    using device::dd::add, device::dd::mul, device::dd::negate;
    c_conj = device::dd::make_complex_double2(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }
  __device__ __forceinline__ void operator()(complex_float4 c, complex_float4& c_conj, float4& d) {
    using device::qf::add, device::qf::mul, device::qf::negate;
    c_conj = device::qf::make_complex_float4(c.real, negate(c.imag));
    d = add(mul(negate(c.real), c.real), add(mul(negate(c.imag), c.imag), d));
  }
};

template <class real_t, class real_ptr, class matrix_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, class matrix_t>
__global__ void gemv_pp_nopiv_kernel(int32_t N, matrix_t sq, matrix_ptr A, int64_t lda, real_ptr D) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t N2 = N & (elements_block - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load1;
  __shared__ typename cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load2;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store1;
  matrix_t thread_i[ITEMS_PER_THREAD]; real_t thread_c[ITEMS_PER_THREAD];
  matrix_ptr B = &A[lda - 1];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load1);
  cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load2);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store_rl(temp_store1);
  gemv_pp_fused<real_t, matrix_t> pp_func;

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A[k], thread_i);
    block_load_rl.Load(&D[k], thread_c);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = k + int32_t(threadIdx.x) + i * BLOCK_THREADS;
      pp_func(thread_i[i], B[int64_t(col) * lda], thread_c[i]);
    }
    block_store_rl.Store(&D[k], thread_c);
  }

  if (0 < N2 && block_offset == (N1 & block_mask)) {
    block_load.Load(&A[N1], thread_i, N2);
    block_load_rl.Load(&D[N1], thread_c, N2);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = N1 + int32_t(threadIdx.x) + i * BLOCK_THREADS;
      if (col < N)
        pp_func(thread_i[i], B[int64_t(col) * lda], thread_c[i]);
    }
    block_store_rl.Store(&D[N1], thread_c, N2);
  }

  if (threadIdx.x == 0 && blockIdx.x == 0)
    A[-1] = sq;
}

constexpr int32_t grid_blocks = 128;
constexpr int32_t block_threads = 128;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::gemv_pp_nopiv_f64(cudaStream_t stream, int32_t M, int32_t N, double sq, double* A, int32_t lda, double* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  gemv_pp_nopiv_kernel <double, double* __restrict__, double* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_f32(cudaStream_t stream, int32_t M, int32_t N, float sq, float* A, int32_t lda, float* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  gemv_pp_nopiv_kernel <float, float* __restrict__, float* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_f128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, double2* A, int32_t lda, double2* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  gemv_pp_nopiv_kernel <double2, double2* __restrict__, double2* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_f128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, float4* A, int32_t lda, float4* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  gemv_pp_nopiv_kernel <float4, float4* __restrict__, float4* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_cf64(cudaStream_t stream, int32_t M, int32_t N, double sq, std::complex<double>* A, int32_t lda, double* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  cuDoubleComplex sqc = make_cuDoubleComplex(sq, 0.);
  gemv_pp_nopiv_kernel <double, double* __restrict__, cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuDoubleComplex*)&A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_cf32(cudaStream_t stream, int32_t M, int32_t N, float sq, std::complex<float>* A, int32_t lda, float* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  cuComplex sqc = make_cuComplex(sq, 0.f);
  gemv_pp_nopiv_kernel <float, float* __restrict__, cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuComplex*)&A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_cf128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, complex_double2* A, int32_t lda, double2* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  complex_double2 sqc = device::dd::make_complex_double2(sq, make_double2(0., 0.));
  gemv_pp_nopiv_kernel <double2, double2* __restrict__, complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1]);
}

void internal::Cholesky::gemv_pp_nopiv_cf128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, complex_float4* A, int32_t lda, float4* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  complex_float4 sqc = device::qf::make_complex_float4(sq, make_float4(0.f, 0.f, 0.f, 0.f));
  gemv_pp_nopiv_kernel <float4, float4* __restrict__, complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1]);
}
