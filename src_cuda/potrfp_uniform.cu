
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(bool p, double epi, double_idx x, double_idx& rsq) {
  double sqx = sqrt(x.real); int32_t i = rsq.idx - int32_t(x.real < epi);
  rsq = double_idx({ 1. / sqx, (p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(sqx, 0.); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(bool p, float epi, float_idx x, float_idx& rsq) {
  float sqx = sqrtf(x.real); int32_t i = rsq.idx - int32_t(x.real < epi);
  rsq = float_idx({ 1.f / sqx, (p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(sqx, 0.f); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(bool p, double2 epi, double2_idx x, double2_idx& rsq) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_double2(x.real, epi, less, par); int32_t i = rsq.idx - int32_t(less);
  rsq = double2_idx({ rsqx, (p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(sqx, make_double2(0., 0.)); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(bool p, float4 epi, float4_idx x, float4_idx& rsq) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_float4(x.real, epi, less, par); int32_t i = rsq.idx - int32_t(less);
  rsq = float4_idx({ rsqx, (p && (0 < x.idx)) ? i : -1 });
  if constexpr(std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(sqx, make_float4(0.f, 0.f, 0.f, 0.f)); } else { return sqx; }
}

__device__ __forceinline__ double _mul(double a, double b) { return a * b; }
__device__ __forceinline__ float _mul(float a, float b) { return a * b; }
__device__ __forceinline__ double2 _mul(double2 a, double2 b) { return device::dd::mul(a, b); }
__device__ __forceinline__ float4 _mul(float4 a, float4 b) { return device::qf::mul(a, b); }

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_init_kernel(real_t epi, int32_t p, int32_t N, matrix_t* __restrict__ A, int64_t lda_p1, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work) {
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  auto grid = cooperative_groups::this_grid();
  const int32_t nthreads = (grid.num_threads()); device::cmp::idx_max cmp_max;

  idx_t thread_x = idx_t();
  for (int32_t i = int32_t(grid.thread_rank()); i < N; i += nthreads) {
    real_t x; int64_t loc = int64_t(i) * lda_p1;
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { x = A[loc].x; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { x = A[loc].x; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { x = A[loc].real; } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { x = A[loc].real; } else { x = A[loc]; }
    thread_x = cmp_max(thread_x, idx_t({ D[i] = x, jpiv[i] = i + 1 }));
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  idx_t rsq = idx_t({ real_t(), p });
  if (BLOCK_THREADS == nthreads) {
    cooperative_groups::this_thread_block().sync();
    if (int32_t(threadIdx.x) == 0) {
      int32_t j = thread_x.idx - 1;
      A[0] = real_sqrt<matrix_t>(1 < N, D[N] = _mul(epi, thread_x.real), thread_x, rsq);
      work[0] = rsq; work[BLOCK_THREADS].idx = j;
      if (0 <= rsq.idx && 0 < j) { matrix_t* A_jj = &A[int64_t(j) * lda_p1]; *A_jj = A_jj[-j]; D[j] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = t; }
    }
  } else {
    if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
    grid.sync();
    if (int32_t(blockIdx.x) == 0) {
      for (int32_t i = int32_t(threadIdx.x); i < int32_t(gridDim.x); i += BLOCK_THREADS)
      { thread_x = cmp_max(thread_x, work[i]); }
      thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
      if (int32_t(threadIdx.x) == 0) {
        int32_t j = thread_x.idx - 1;
        A[0] = real_sqrt<matrix_t>(1 < N, D[N] = _mul(epi, thread_x.real), thread_x, rsq);
        work[0] = rsq; work[BLOCK_THREADS].idx = j;
        if (0 <= rsq.idx && 0 < j) { matrix_t* A_jj = &A[int64_t(j) * lda_p1]; *A_jj = A_jj[-j]; D[j] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = t; }
      }
    }
  }
}

__device__ __forceinline__ double pp_func(double rsq, double c, double& d) {
  c = rsq * c; d = fma(-c, c, d); return c;
}
__device__ __forceinline__ float pp_func(float rsq, float c, float& d) {
  c = rsq * c; d = fmaf(-c, c, d); return c;
}
__device__ __forceinline__ double2 pp_func(double2 rsq, double2 c, double2& d) {
  c = device::dd::mul(rsq, c); d = device::dd::add(d, device::dd::negate(device::dd::square(c))); return c;
}
__device__ __forceinline__ float4 pp_func(float4 rsq, float4 c, float4& d) {
  c = device::qf::mul(rsq, c); d = device::qf::add(d, device::qf::negate(device::qf::square(c))); return c;
}
__device__ __forceinline__ cuDoubleComplex pp_func(double rsq, cuDoubleComplex c, double& d) {
  c = make_cuDoubleComplex(rsq * c.x, -rsq * c.y); d = fma(-c.x, c.x, fma(-c.y, c.y, d)); return c;
}
__device__ __forceinline__ cuComplex pp_func(float rsq, cuComplex c, float& d) {
  c = make_cuComplex(rsq * c.x, -rsq * c.y); d = fmaf(-c.x, c.x, fmaf(-c.y, c.y, d)); return c;
}
__device__ __forceinline__ complex_double2 pp_func(double2 rsq, complex_double2 c, double2& d) {
  using device::dd::add, device::dd::mul, device::dd::square, device::dd::negate;
  c = device::dd::make_complex_double2(mul(rsq, c.real), negate(mul(rsq, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag)))); return c;
}
__device__ __forceinline__ complex_float4 pp_func(float4 rsq, complex_float4 c, float4& d) {
  using device::qf::add, device::qf::mul, device::qf::square, device::qf::negate;
  c = device::qf::make_complex_float4(mul(rsq, c.real), negate(mul(rsq, c.imag)));
  d = add(d, negate(add(square(c.real), square(c.imag)))); return c;
}

template <class real_t, class matrix_t> struct add_fl {
  __device__ __forceinline__ matrix_t operator()(matrix_t a, matrix_t b) {
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return a + b; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::dd::add(a, b); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::qf::add(a, b); } else
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x + b.x, a.y + b.y); } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(device::dd::add(a.real, b.real), device::dd::add(a.imag, b.imag)); } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(device::qf::add(a.real, b.real), device::qf::add(a.imag, b.imag)); } else
    { return matrix_t(); }
  }
};

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t fma_(matrix_t a, matrix_t b, matrix_t c) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return fma(a, b, c); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return fmaf(a, b, c); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::dd::add(c, device::dd::mul(a, b)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::qf::add(c, device::qf::mul(a, b)); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(fma(a.x, b.x, c.x), fma(a.y, b.y, c.y)); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(fmaf(a.x, b.x, c.x), fmaf(a.y, b.y, c.y)); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) {
    using device::dd::add, device::dd::mul, device::dd::negate, device::dd::make_complex_double2;
    return make_complex_double2(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
  } else if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) {
    using device::qf::add, device::qf::mul, device::qf::negate, device::qf::make_complex_float4;
    return make_complex_float4(add(mul(a.real, b.real), add(mul(a.imag, b.imag), c.real)), add(mul(a.real, b.imag), add(mul(negate(a.imag), b.real), c.imag)));
  } else { return matrix_t(); }
}

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t neg_(matrix_t a) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, double>) { return -a; } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, float>) { return -a; } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, double2>) { return device::dd::negate(a); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, float4>) { return device::qf::negate(a); } else
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(-a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(-a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(device::dd::negate(a.real), device::dd::negate(a.imag)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(device::qf::negate(a.real), device::qf::negate(a.imag)); } else
  { return matrix_t(); }
}

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) {
  if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(a.x, -a.y); } else
  if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); } else
  if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); } else
  { return a; }
}

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void potrf_iter_kernel(int32_t iterMax, int32_t N, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, int32_t* __restrict__ out) {
  __shared__ idx_t rsq; __shared__ int32_t j; add_fl<real_t, matrix_t> add_; device::cmp::idx_max cmp_max; 
  __shared__ typename cub::BlockReduce<matrix_t, BLOCK_THREADS>::TempStorage temp_gemv;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;

  auto grid = cooperative_groups::this_grid();
  const int32_t tid = int32_t(grid.thread_rank()), nthreads = int32_t(grid.num_threads());
  if (int32_t(threadIdx.x) == 0) { rsq = work[0]; j = work[BLOCK_THREADS].idx; }
  cooperative_groups::this_thread_block().sync();

  int32_t M = 0;
  while (0 <= rsq.idx) {
    matrix_t* A_col_j = &A[int64_t(j) * lda];
    if (0 < M) {
      for (int32_t i = int32_t(blockIdx.x); i < N; i += int32_t(gridDim.x)) if (i != j) {
        matrix_t threadB = matrix_t(), *A_col_i = &A[int64_t(i) * lda];
        for (int32_t k = int32_t(threadIdx.x) - M; k < 0; k += BLOCK_THREADS)
        { threadB = fma_<real_t>(A_col_i[k], A_col_j[k], threadB); }
        threadB = cub::BlockReduce<matrix_t, BLOCK_THREADS>(temp_gemv).Reduce(threadB, add_);
        cooperative_groups::this_thread_block().sync();
        if (int32_t(threadIdx.x) == 0) { A_col_j[i ? i : j] = add_(neg_<real_t>(threadB), A_col_j[i]); }
      }
      grid.sync();
    }

    idx_t thread_x = idx_t();
    for (int32_t i = tid + 1; i < N; i += nthreads) {
      matrix_t* A_col_i = &A[int64_t(i) * lda];
      idx_t thread_c = idx_t({ D[i], i });
      A_col_i[0] = pp_func(rsq.real, A_col_j[i], thread_c.real);
      if (0 < j && i != j) { A_col_i[j] = conj<real_t>(A_col_j[i] = A[i]); }
      thread_x = cmp_max(thread_x, thread_c);
      D[i] = thread_c.real;
    }

    if (0 < j) for (int32_t i = tid - M; i < 0; i += nthreads)
    { matrix_t t = A[i]; A[i] = A_col_j[i]; A_col_j[i] = t; }

    A = &(++A)[lda]; ++M; --N; ++jpiv; ++D;
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
    if (BLOCK_THREADS == nthreads) {
      cooperative_groups::this_thread_block().sync();
      if (int32_t(threadIdx.x) == 0) {
        *A = real_sqrt<matrix_t>(1 < N && M < iterMax, D[N], thread_x, rsq);
        j = thread_x.idx - 1;
        if (0 <= rsq.idx && 0 < j) { D[j] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = t; }
      }
    } else {
      if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
      grid.sync();
      if (int32_t(blockIdx.x) == 0) {
        for (int32_t i = int32_t(threadIdx.x); i < int32_t(gridDim.x); i += BLOCK_THREADS)
        { thread_x = cmp_max(thread_x, work[i]); }
        thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
        if (int32_t(threadIdx.x) == 0) {
          *A = real_sqrt<matrix_t>(1 < N && M < iterMax, D[N], thread_x, rsq);
          work[0] = rsq; work[BLOCK_THREADS].idx = (j = thread_x.idx - 1);
          if (0 <= rsq.idx && 0 < j) { D[j] = D[0]; int32_t t = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = t; }
        }
      }
      grid.sync();
      if (int32_t(threadIdx.x) == 0) { rsq = work[0]; j = work[BLOCK_THREADS].idx; }
    }
    cooperative_groups::this_thread_block().sync();
  }

  if (tid == 0) { *out = M + int32_t(N == 1); }
}

template <char mode, class real_t, class matrix_t>
__global__ void matrix_fill_upper_to_full(matrix_t* __restrict__ A, int64_t lda) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U') { pred = y < x; } else if constexpr(mode == 'L') { pred = x < y; } else { pred = false; }
  if (pred) { A[x + y * lda] = conj<real_t>(A[y + x * lda]); }
}

template <class idx_t, class real_t, class matrix_t>
inline int32_t potrfp_dispatcher(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, int32_t* rank) {
  if (fillmode == 'U' || fillmode == 'u')
    matrix_fill_upper_to_full<'U', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);
  else if (fillmode == 'L' || fillmode == 'l')
    matrix_fill_upper_to_full<'L', real_t> <<< dim3(uint32_t(N + 511) >> 9, uint32_t(N)), 512, 0, stream >>> (A, lda);

  k = std::min(N, std::max(0, k)); k = k ? k : N; p = std::max(0, p); epi = std::min(1., std::max(0., std::pow(epi, 2)));
  real_t epi_f = real_t();
  if constexpr(std::is_same_v<real_t, double>) { epi_f = epi; }
  if constexpr(std::is_same_v<real_t, float>) { epi_f = float(epi); } else
  if constexpr(std::is_same_v<real_t, double2>) { epi_f = device::dd::double2dd(epi); } else
  if constexpr(std::is_same_v<real_t, float4>) { epi_f = device::qf::double2qf(epi); }

  constexpr int32_t grid_blocks = 2048, block_threads = 128;
  int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  uint8_t* diag = &((uint8_t*)D)[65536]; int64_t lda_p1 = lda + int64_t(1);
  void* initArgs[]{ &epi_f, &p, &N, &A, &lda_p1, &jpiv, &diag, &D };
  cudaLaunchCooperativeKernel(potrf_init_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, initArgs, 0, stream);

  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), N);
  void* kernelArgs[]{ &k, &N, &A, &lda, &jpiv, &diag, &D, &rank };
  cudaLaunchCooperativeKernel(potrf_iter_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaStreamSynchronize(stream); return *rank;
}

namespace internal::Cholesky {

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<double2_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

  int32_t potrfp(cudaStream_t stream, char fillmode, double epi, int32_t k, int32_t p, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* dev_work, int32_t* pinned_work)
  { return potrfp_dispatcher<float4_idx>(stream, fillmode, epi, k, p, N, A, lda, jpiv, dev_work, pinned_work); }

};