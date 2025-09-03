
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::make_complex_double2(f.real, device::dd::negate(f.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::make_complex_float4(f.real, device::qf::negate(f.imag)); }
};

struct scal_a {
  __device__ __forceinline__ double operator()(double s, double a) { return s * a; }
  __device__ __forceinline__ float operator()(float s, float a) { return s * a; }
  __device__ __forceinline__ double2 operator()(double2 s, double2 a) { return device::dd::mul(s, a); }
  __device__ __forceinline__ float4 operator()(float4 s, float4 a) { return device::qf::mul(s, a); }

  __device__ __forceinline__ cuDoubleComplex operator()(double s, cuDoubleComplex a) { return make_cuDoubleComplex(s * a.x, s * a.y); }
  __device__ __forceinline__ cuComplex operator()(float s, cuComplex a) { return make_cuComplex(s * a.x, s * a.y); }
  __device__ __forceinline__ complex_double2 operator()(double2 s, complex_double2 a) { return device::dd::make_complex_double2(operator()(s, a.real), operator()(s, a.imag)); }
  __device__ __forceinline__ complex_float4 operator()(float4 s, complex_float4 a) { return device::qf::make_complex_float4(operator()(s, a.real), operator()(s, a.imag)); }
};

struct subtract_norm {
  __device__ __forceinline__ double operator()(double a, double c) { return fma(-a, a, c); }
  __device__ __forceinline__ float operator()(float a, float c) { return fmaf(-a, a, c); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 c) { return device::dd::add(c, device::dd::mul(device::dd::negate(a), a)); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 c) { return device::qf::add(c, device::qf::mul(device::qf::negate(a), a)); }

  __device__ __forceinline__ double operator()(cuDoubleComplex a, double c) { return operator()(a.x, operator()(a.y, c)); }
  __device__ __forceinline__ float operator()(cuComplex a, float c) { return operator()(a.x, operator()(a.y, c)); }
  __device__ __forceinline__ double2 operator()(complex_double2 a, double2 c) { return operator()(a.real, operator()(a.imag, c)); }
  __device__ __forceinline__ float4 operator()(complex_float4 a, float4 c) { return operator()(a.real, operator()(a.imag, c)); }
};

template <class real_t, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void array_transform(real_t rsq, matrix_t (&a)[ITEMS_PER_THREAD], matrix_t (&b)[ITEMS_PER_THREAD], real_t (&c)[ITEMS_PER_THREAD]) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  scal_a scal_func; subtract_norm fma_func;

  if constexpr (COMPLEX) {
    conj conj_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      matrix_t bs = scal_func(rsq, b[i]);
      a[i] = conj_func(a[i]);
      b[i] = conj_func(bs);
      c[i] = fma_func(bs, c[i]);
    }
  }
  else {
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      matrix_t bs = scal_func(rsq, b[i]);
      b[i] = bs;
      c[i] = fma_func(bs, c[i]);
    }
  }
}

template <class real_t, class real_ptr, class matrix_t, class matrix_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void swap_cols_kernel(int32_t j, int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_ptr A, int64_t lda, real_ptr D) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t thread_mask = BLOCK_THREADS - 1;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t N2 = N & (elements_block - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load1;
  __shared__ typename cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load2;
  __shared__ typename cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store2;
  __shared__ matrix_t Aij[2];
  matrix_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD];
  real_t thread_c[ITEMS_PER_THREAD];
  matrix_ptr A_i = &A[M], A_col_j = &A[uint64_t(M) + uint64_t(j) * lda], A_row_j = &A[j + M];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load1);
  cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load2);
  cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store2);
  if (threadIdx.x == 0)
    if constexpr(COMPLEX)
    { conj conj_func; Aij[0] = conj_func(A_col_j[0]); Aij[1] = conj_func(A_row_j[0]); }
    else
    { Aij[0] = A_col_j[0]; Aij[1] = A_row_j[0]; }
  __syncthreads();

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A_i[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);
    block_load_rl.Load(&D[k], thread_c);

    block_store.Store(&A_col_j[k], thread_i);
    block_store.Store(&A_i[k], thread_j);
    array_transform(rsq, thread_i, thread_j, thread_c);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l) {
      int32_t col = k + int32_t(threadIdx.x) + l * BLOCK_THREADS;
      if (1 <= col && col != j) {
        int64_t col_idx = uint64_t(col) * lda;
        A_i[col_idx] = thread_j[l];
        A_row_j[col_idx] = thread_i[l];
      }
    }
  }

  if (0 < N2 && block_offset == (N1 & block_mask)) {
    block_load.Load(&A_i[N1], thread_i, N2);
    block_load.Load(&A_col_j[N1], thread_j, N2);
    block_load_rl.Load(&D[N1], thread_c, N2);

    block_store.Store(&A_col_j[N1], thread_i, N2);
    block_store.Store(&A_i[N1], thread_j, N2);
    array_transform(rsq, thread_i, thread_j, thread_c);

    #pragma unroll
    for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l) {
      int32_t col = N1 + int32_t(threadIdx.x) + l * BLOCK_THREADS;
      if (1 <= col && col != j && col < N) {
        int64_t col_idx = uint64_t(col) * lda;
        A_i[col_idx] = thread_j[l];
        A_row_j[col_idx] = thread_i[l];
      }
    }
  }

  if (threadIdx.x == 0 && block_offset == 0)
  { A_i[0] = sq; A_col_j[0] = Aij[0]; D[j] = D[0]; }
  if (threadIdx.x == (j & thread_mask) && block_offset == (j & block_mask))
    A_row_j[0] = Aij[1];

  A_col_j = &A[uint64_t(j) * lda];
  N2 = M & (elements_block - 1);
  N1 = M - N2;

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);
    block_store.Store(&A_col_j[k], thread_i);
    block_store.Store(&A[k], thread_j);
  }

  if (0 < N2 && block_offset == (N1 & block_mask)) {
    block_load.Load(&A[N1], thread_i, N2);
    block_load.Load(&A_col_j[N1], thread_j, N2);
    block_store.Store(&A_col_j[N1], thread_i, N2);
    block_store.Store(&A[N1], thread_j, N2);
  }
}

constexpr int32_t grid_blocks = 128;
constexpr int32_t block_threads = 128;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::swap_cols_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, double* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  swap_cols_kernel <double, double* __restrict__, double, double* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, 0., 1., A, lda, D);
}

void internal::Cholesky::swap_cols_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, float* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  swap_cols_kernel <float, float* __restrict__, float, float* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, 0.f, 1.f, A, lda, D);
}

void internal::Cholesky::swap_cols_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  swap_cols_kernel <double2, double2* __restrict__, double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, make_double2(0., 0.), make_double2(1., 0.), A, lda, D);
}

void internal::Cholesky::swap_cols_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  swap_cols_kernel <float4, float4* __restrict__, float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, make_float4(0.f, 0.f, 0.f, 0.f), make_float4(1.f, 0.f, 0.f, 0.f), A, lda, D);
}

void internal::Cholesky::swap_cols_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  swap_cols_kernel <double, double* __restrict__, cuDoubleComplex, cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, make_cuDoubleComplex(0., 0.), 1., (cuDoubleComplex*)A, lda, D);
}

void internal::Cholesky::swap_cols_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  swap_cols_kernel <float, float* __restrict__, cuComplex, cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, make_cuComplex(0.f, 0.f), 1.f, (cuComplex*)A, lda, D);
}

void internal::Cholesky::swap_cols_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  swap_cols_kernel <double2, double2* __restrict__, complex_double2, complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, device::dd::make_complex_double2(make_double2(0., 0.), make_double2(0., 0.)), make_double2(1., 0.), A, lda, D);
}

void internal::Cholesky::swap_cols_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  swap_cols_kernel <float4, float4* __restrict__, complex_float4, complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, device::qf::make_complex_float4(make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)), make_float4(1.f, 0.f, 0.f, 0.f), A, lda, D);
}
