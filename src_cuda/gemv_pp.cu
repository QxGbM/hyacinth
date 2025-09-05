
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <numeric>

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::make_complex_double2(f.real, device::dd::negate(f.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::make_complex_float4(f.real, device::qf::negate(f.imag)); }
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

template <class real_t, class matrix_t>
__device__ __forceinline__ void elem_transform(matrix_t &a, matrix_t &b, real_t &c) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  subtract_norm fma_func; conj conj_func;
  if constexpr (COMPLEX) { matrix_t bs = b; a = conj_func(a); b = conj_func(bs); c = fma_func(bs, c); }
    else { c = fma_func(b, c); }
}

struct __align__(8) float_idx { float real; int32_t idx; };
struct __align__(16) double_idx { double real; int32_t idx; };

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    double val = less ? b.real : a.real;
    int32_t id = less ? b.idx : par ? min(a.idx, b.idx) : a.idx;
    return double_idx({ val, id });
  }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    float val = less ? b.real : a.real;
    int32_t id = less ? b.idx : par ? min(a.idx, b.idx) : a.idx;
    return float_idx({ val, id });
  }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::dd::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::qf::float4_max(a, b); }

  __host__ __device__ __forceinline__ void init(double_idx& a) { a = double_idx({ 0., -1 }); }
  __host__ __device__ __forceinline__ void init(float_idx& a) { a = float_idx({ 0.f, -1 }); }
  __host__ __device__ __forceinline__ void init(double2_idx& a) { a = double2_idx({ make_double2(0., 0.), -1 }); }
  __host__ __device__ __forceinline__ void init(float4_idx& a) { a = float4_idx({ make_float4(0.f, 0.f, 0.f, 0.f), -1 }); }
};

template <class real_t, class real_ptr, class matrix_ptr, class idx_t, class idx_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, class matrix_t>
__global__ void gemv_pp_kernel(int32_t j, int32_t M, int32_t N, matrix_t sq, matrix_ptr A, int64_t lda, real_ptr D, idx_ptr idx) {
  constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t N2 = N & (elements_block - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load1;
  __shared__ typename cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load2;
  __shared__ typename cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store2;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;

  __shared__ matrix_t Aij[2];
  matrix_t thread_i[ITEMS_PER_THREAD], thread_j[ITEMS_PER_THREAD]; real_t thread_c[ITEMS_PER_THREAD];
  idx_t thread_x[ITEMS_PER_THREAD]; int32_t thread_locs[ITEMS_PER_THREAD];
  matrix_ptr A_i = &A[M], A_col_j = &A[uint64_t(M) + uint64_t(j) * lda], A_row_j = &A[j + M];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load1);
  cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load2);
  cub::BlockStore<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store2);
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  real_max cmp_max;

  if (threadIdx.x == 0 && block_offset == 0) {
    conj conj_func;
    if constexpr(COMPLEX) Aij[0] = conj_func(A_col_j[0]);
      else Aij[0] = A_col_j[0];
    Aij[1] = sq;
  }

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_locs[i] = int32_t(threadIdx.x) + i * BLOCK_THREADS;
  __syncthreads();

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A_i[k], thread_i);
    block_load.Load(&A_col_j[k], thread_j);
    block_store.Store(&A_col_j[k], thread_i);
    block_load_rl.Load(&D[k], thread_c);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = k + thread_locs[i];
      elem_transform(thread_i[i], thread_j[i], thread_c[i]);
      if (col != j) {
        if (0 < col) {
          int64_t col_idx = uint64_t(col) * lda;
          A_i[col_idx] = thread_j[i];
          A_row_j[col_idx] = thread_i[i];
        }
        D[col] = thread_c[i];
      }
      else
        thread_c[i] = real_t();
    }

    if (k == block_offset) {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_x[i] = idx_t({ thread_c[i], k + thread_locs[i] });
    }
    else {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_x[i] = cmp_max(thread_x[i], idx_t({ thread_c[i], k + thread_locs[i] }));
    }
  }

  if (0 < N2 && block_offset == (N1 & block_mask)) {
    block_load.Load(&A_i[N1], thread_i, N2);
    block_load.Load(&A_col_j[N1], thread_j, N2);
    block_store.Store(&A_col_j[N1], thread_i, N2);
    block_load_rl.Load(&D[N1], thread_c, N2);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = N1 + thread_locs[i];
      if (col != j && col < N) {
        elem_transform(thread_i[i], thread_j[i], thread_c[i]);
        if (0 < col) {
          int64_t col_idx = uint64_t(col) * lda;
          A_i[col_idx] = thread_j[i];
          A_row_j[col_idx] = thread_i[i];
        }
        D[col] = thread_c[i];
      }
      else
        thread_c[i] = real_t();
    }

    if (N1 == block_offset) {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_x[i] = idx_t({ thread_c[i], N1 + thread_locs[i] });
    }
    else {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_x[i] = cmp_max(thread_x[i], idx_t({ thread_c[i], N1 + thread_locs[i] }));
    }
  }

  idx_t block_res; cmp_max.init(block_res);
  if (block_offset < N)
    block_res = block_reduce.Reduce(thread_x, cmp_max);

  if (threadIdx.x == 0) {
    idx[blockIdx.x] = idx_t({ block_res.real, (block_res.idx == 0 ? j : block_res.idx) - 1 });
    if (blockIdx.x == 0)
    { A_col_j[0] = Aij[0]; A_i[0] = Aij[1]; D[j] = D[0]; }
  }

  A_col_j = &A[uint64_t(j) * lda];
  N2 = M & (elements_block - 1); N1 = M - N2;

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

void internal::Cholesky::gemv_pp_f64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double sq, double* A, int32_t lda, double* D, double* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  if (0 < j) {
    double_idx* p = (double_idx*)diag_piv, init({ 0., -1 });
    gemv_pp_kernel <double, double* __restrict__, double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sq, A, lda, D, p);

    cudaStreamSynchronize(stream);
    double_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_f64(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_f32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float sq, float* A, int32_t lda, float* D, float* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  if (0 < j) {
    float_idx* p = (float_idx*)diag_piv, init({ 0.f, -1 });
    gemv_pp_kernel <float, float* __restrict__, float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sq, A, lda, D, p);

    cudaStreamSynchronize(stream);
    float_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_f32(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_f128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2 sq, double2* A, int32_t lda, double2* D, double2* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  if (0 < j) {
    double2_idx* p = (double2_idx*)diag_piv, init({ make_double2(0., 0.), -1 });
    gemv_pp_kernel <double2, double2* __restrict__, double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sq, A, lda, D, p);

    cudaStreamSynchronize(stream);
    double2_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_f128_dd(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_f128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4 sq, float4* A, int32_t lda, float4* D, float4* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  if (0 < j) {
    float4_idx* p = (float4_idx*)diag_piv, init({ make_float4(0.f, 0.f, 0.f, 0.f), -1 });
    gemv_pp_kernel <float4, float4* __restrict__, float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sq, A, lda, D, p);

    cudaStreamSynchronize(stream);
    float4_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_f128_qf(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_cf64(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double sq, std::complex<double>* A, int32_t lda, double* D, double* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  if (0 < j) {
    cuDoubleComplex sqc = make_cuDoubleComplex(sq, 0.);
    double_idx* p = (double_idx*)diag_piv, init({ 0., -1 });
    gemv_pp_kernel <double, double* __restrict__, cuDoubleComplex* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, (cuDoubleComplex*)A, lda, D, p);

    cudaStreamSynchronize(stream);
    double_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_cf64(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_cf32(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float sq, std::complex<float>* A, int32_t lda, float* D, float* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  if (0 < j) {
    cuComplex sqc = make_cuComplex(sq, 0.f);
    float_idx* p = (float_idx*)diag_piv, init({ 0.f, -1 });
    gemv_pp_kernel <float, float* __restrict__, cuComplex* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, (cuComplex*)A, lda, D, p);

    cudaStreamSynchronize(stream);
    float_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_cf32(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_cf128_dd(cudaStream_t stream, int32_t j, int32_t M, int32_t N, double2 sq, complex_double2* A, int32_t lda, double2* D, double2* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  if (0 < j) {
    complex_double2 sqc = device::dd::make_complex_double2(sq, make_double2(0., 0.));
    double2_idx* p = (double2_idx*)diag_piv, init({ make_double2(0., 0.), -1 });
    gemv_pp_kernel <double2, double2* __restrict__, complex_double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, A, lda, D, p);

    cudaStreamSynchronize(stream);
    double2_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_cf128_dd(stream, M, N, sq, A, lda, D, diag_piv);
}

void internal::Cholesky::gemv_pp_cf128_qf(cudaStream_t stream, int32_t j, int32_t M, int32_t N, float4 sq, complex_float4* A, int32_t lda, float4* D, float4* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  if (0 < j) {
    complex_float4 sqc = device::qf::make_complex_float4(sq, make_float4(0.f, 0.f, 0.f, 0.f));
    float4_idx* p = (float4_idx*)diag_piv, init({ make_float4(0.f, 0.f, 0.f, 0.f), -1 });
    gemv_pp_kernel <float4, float4* __restrict__, complex_float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
      <<< grid_blocks, block_threads, 0, stream >>> (j, M, N, sqc, A, lda, D, p);

    cudaStreamSynchronize(stream);
    float4_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
    p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
  }
  else
    gemv_pp_nopiv_cf128_qf(stream, M, N, sq, A, lda, D, diag_piv);
}
