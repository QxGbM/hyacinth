
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <numeric>
#include <execution>

struct gemv_pp_fused {
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

struct __align__(8) float_idx { float real; int32_t idx; };
struct __align__(16) double_idx { double real; int32_t idx; };

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    double val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
    return double_idx({ val, id });
  }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    float val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
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
__global__ void gemv_pp_nopiv_kernel(int32_t N, matrix_t sq, matrix_ptr A, int64_t lda, real_ptr D, idx_ptr idx) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t N2 = N & (elements_block - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load1;
  __shared__ typename cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load2;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store1;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  matrix_t thread_i[ITEMS_PER_THREAD]; real_t thread_c[ITEMS_PER_THREAD];
  int32_t thread_locs[ITEMS_PER_THREAD]; idx_t thread_x[ITEMS_PER_THREAD];
  matrix_ptr B = &A[lda - 1];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load1);
  cub::BlockLoad<matrix_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load2);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store_rl(temp_store1);
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  gemv_pp_fused pp_func; real_max cmp_max;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_locs[i] = int32_t(threadIdx.x) + i * BLOCK_THREADS;

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&A[k], thread_i);
    block_load_rl.Load(&D[k], thread_c);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = k + thread_locs[i];
      pp_func(thread_i[i], B[int64_t(col) * lda], thread_c[i]);
    }
    block_store_rl.Store(&D[k], thread_c);

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
    block_load.Load(&A[N1], thread_i, N2);
    block_load_rl.Load(&D[N1], thread_c, N2);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      int32_t col = N1 + thread_locs[i];
      if (col < N)
        pp_func(thread_i[i], B[int64_t(col) * lda], thread_c[i]);
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
    block_store_rl.Store(&D[N1], thread_c, N2);
  }

  idx_t block_res; cmp_max.init(block_res);
  if (block_offset < N)
    block_res = block_reduce.Reduce(thread_x, cmp_max);

  if (threadIdx.x == 0) {
    idx[blockIdx.x] = block_res;
    if (blockIdx.x == 0)
      A[-1] = sq;
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::gemv_pp_nopiv_f64(cudaStream_t stream, int32_t M, int32_t N, double sq, double* A, int32_t lda, double* D, double* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  double_idx* p = (double_idx*)diag_piv, init({ 0., -1 });
  gemv_pp_nopiv_kernel <double, double* __restrict__, double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  double_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_f32(cudaStream_t stream, int32_t M, int32_t N, float sq, float* A, int32_t lda, float* D, float* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  float_idx* p = (float_idx*)diag_piv, init({ 0.f, -1 });
  gemv_pp_nopiv_kernel <float, float* __restrict__, float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  float_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_f128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, double2* A, int32_t lda, double2* D, double2* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  double2_idx* p = (double2_idx*)diag_piv, init({ make_double2(0., 0.), -1 });
  gemv_pp_nopiv_kernel <double2, double2* __restrict__, double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  double2_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_f128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, float4* A, int32_t lda, float4* D, float4* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  float4_idx* p = (float4_idx*)diag_piv, init({ make_float4(0.f, 0.f, 0.f, 0.f), -1 });
  gemv_pp_nopiv_kernel <float4, float4* __restrict__, float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sq, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  float4_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_cf64(cudaStream_t stream, int32_t M, int32_t N, double sq, std::complex<double>* A, int32_t lda, double* D, double* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  cuDoubleComplex sqc = make_cuDoubleComplex(sq, 0.);
  double_idx* p = (double_idx*)diag_piv, init({ 0., -1 });
  gemv_pp_nopiv_kernel <double, double* __restrict__, cuDoubleComplex* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuDoubleComplex*)&A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  double_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_cf32(cudaStream_t stream, int32_t M, int32_t N, float sq, std::complex<float>* A, int32_t lda, float* D, float* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  cuComplex sqc = make_cuComplex(sq, 0.f);
  float_idx* p = (float_idx*)diag_piv, init({ 0.f, -1 });
  gemv_pp_nopiv_kernel <float, float* __restrict__, cuComplex* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, (cuComplex*)&A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  float_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_cf128_dd(cudaStream_t stream, int32_t M, int32_t N, double2 sq, complex_double2* A, int32_t lda, double2* D, double2* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  complex_double2 sqc = device::dd::make_complex_double2(sq, make_double2(0., 0.));
  double2_idx* p = (double2_idx*)diag_piv, init({ make_double2(0., 0.), -1 });
  gemv_pp_nopiv_kernel <double2, double2* __restrict__, complex_double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  double2_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}

void internal::Cholesky::gemv_pp_nopiv_cf128_qf(cudaStream_t stream, int32_t M, int32_t N, float4 sq, complex_float4* A, int32_t lda, float4* D, float4* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  complex_float4 sqc = device::qf::make_complex_float4(sq, make_float4(0.f, 0.f, 0.f, 0.f));
  float4_idx* p = (float4_idx*)diag_piv, init({ make_float4(0.f, 0.f, 0.f, 0.f), -1 });
  gemv_pp_nopiv_kernel <float4, float4* __restrict__, complex_float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread> 
    <<< grid_blocks, block_threads, 0, stream >>> (N - 1, sqc, &A[M + 1], lda, &D[1], p);

  int32_t len = std::min(grid_blocks, (N + elements_block - 1) / elements_block);
  cudaStreamSynchronize(stream);
  float4_idx res = std::reduce(std::execution::unseq, p, &p[len], init, real_max());
  p[0] = (0 <= res.idx && res.idx < (N - 1)) ? res : init;
}
