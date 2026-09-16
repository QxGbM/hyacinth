
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(double epi, double_idx x, double_idx& rsq) {
  double sqx = sqrt(x.real); int32_t i = rsq.idx - int32_t(x.real < epi);
  rsq = double_idx({ 1. / sqx, x.idx <= 0 ? -1 : i });
  if constexpr(std::is_same_v<matrix_t, cuDoubleComplex>) { return make_cuDoubleComplex(sqx, 0.); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(float epi, float_idx x, float_idx& rsq) {
  float sqx = sqrtf(x.real); int32_t i = rsq.idx - int32_t(x.real < epi);
  rsq = float_idx({ 1.f / sqx, x.idx <= 0 ? -1 : i });
  if constexpr(std::is_same_v<matrix_t, cuComplex>) { return make_cuComplex(sqx, 0.f); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(double2 epi, double2_idx x, double2_idx& rsq) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_double2(x.real, epi, less, par); int32_t i = rsq.idx - int32_t(less);
  rsq = double2_idx({ rsqx, x.idx <= 0 ? -1 : i });
  if constexpr(std::is_same_v<matrix_t, complex_double2>) { return device::dd::make_complex_double2(sqx, make_double2(0., 0.)); } else { return sqx; }
}
template <class matrix_t> __device__ __forceinline__ matrix_t real_sqrt(float4 epi, float4_idx x, float4_idx& rsq) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_float4(x.real, epi, less, par); int32_t i = rsq.idx - int32_t(less);
  rsq = float4_idx({ rsqx, x.idx <= 0 ? -1 : i });
  if constexpr(std::is_same_v<matrix_t, complex_float4>) { return device::qf::make_complex_float4(sqx, make_float4(0.f, 0.f, 0.f, 0.f)); } else { return sqx; }
}

__device__ __forceinline__ double _mul(double a, double b) { return a * b; }
__device__ __forceinline__ float _mul(float a, float b) { return a * b; }
__device__ __forceinline__ double2 _mul(double2 a, double2 b) { return device::dd::mul(a, b); }
__device__ __forceinline__ float4 _mul(float4 a, float4 b) { return device::qf::mul(a, b); }

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void imax_kernel(real_t epi, int32_t p, int32_t N, matrix_t* __restrict__ X, int64_t incx, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, idx_t* __restrict__ idx) {
  auto grid = cooperative_groups::this_grid();
  const int32_t elements = (grid.num_threads()); idx_t thread_x = idx_t();
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  device::cmp::idx_max cmp_max;

  for (int32_t i = int32_t(grid.thread_rank()); i < N; i += elements) {
    real_t x; int64_t loc = int64_t(i) * incx;
    if constexpr(std::is_same_v<real_t, double> && std::is_same_v<matrix_t, cuDoubleComplex>) { x = X[loc].x; } else
    if constexpr(std::is_same_v<real_t, float> && std::is_same_v<matrix_t, cuComplex>) { x = X[loc].x; } else
    if constexpr(std::is_same_v<real_t, double2> && std::is_same_v<matrix_t, complex_double2>) { x = X[loc].real; } else
    if constexpr(std::is_same_v<real_t, float4> && std::is_same_v<matrix_t, complex_float4>) { x = X[loc].real; } else { x = X[loc]; }
    thread_x = cmp_max(thread_x, idx_t({ D[i] = x, jpiv[i] = i + 1 }));
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (BLOCK_THREADS == elements) {
    if (int32_t(threadIdx.x) == 0) { X[0] = real_sqrt<matrix_t>(D[N] = _mul(epi, thread_x.real), thread_x, idx[0] = idx_t({ real_t(), p })); work->idx = --thread_x.idx; }
    return;
  }

  if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
  grid.sync();
  if (int32_t(blockIdx.x) == 0) {
    for (int32_t i = int32_t(threadIdx.x); i < int32_t(gridDim.x); i += BLOCK_THREADS)
    { thread_x = cmp_max(thread_x, work[i]); }
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
    if (int32_t(threadIdx.x) == 0) { X[0] = real_sqrt<matrix_t>(D[N] = _mul(epi, thread_x.real), thread_x, idx[0] = idx_t({ real_t(), p })); work->idx = --thread_x.idx; }
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
__global__ void gemv_pp_kernel(int32_t M, int32_t N, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, idx_t* __restrict__ idx) {
  auto grid = cooperative_groups::this_grid();
  __shared__ idx_t rsq; __shared__ int32_t j;
  if (int32_t(threadIdx.x) == 0) { rsq = *idx; j = work->idx; }
  cooperative_groups::this_thread_block().sync(); if (rsq.idx < 0) { return; }
  __shared__ typename cub::BlockReduce<matrix_t, BLOCK_THREADS>::TempStorage temp_gemv;
  matrix_t* A_col_j = &A[int64_t(j) * lda]; 

  if (0 < M) {
    for (int32_t i = int32_t(blockIdx.x); i < N; i += int32_t(gridDim.x)) if (i != j) {
      matrix_t threadB = matrix_t(), *A_col_i = &A[int64_t(i) * lda];
      for (int32_t k = int32_t(threadIdx.x) - M; k < 0; k += BLOCK_THREADS)
        threadB = fma_<real_t>(A_col_i[k], A_col_j[k], threadB);

      add_fl<real_t, matrix_t> add_;
      threadB = cub::BlockReduce<matrix_t, BLOCK_THREADS>(temp_gemv).Reduce(threadB, add_);
      cooperative_groups::this_thread_block().sync();
      if (int32_t(threadIdx.x) == 0) { matrix_t* A_ij = &A_col_j[i]; *A_ij = add_(neg_<real_t>(threadB), *A_ij); }
    }
    grid.sync();
  } else { cooperative_groups::this_thread_block().sync(); }

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  device::cmp::idx_max cmp_max; idx_t thread_x = idx_t();
  const int32_t offset = int32_t(grid.thread_rank()) + 1, elements = int32_t(grid.num_threads());

  for (int32_t i = offset; i < N; i += elements) {
    matrix_t* A_col_i = &A[int64_t(i) * lda];
    bool prec = i == j;
    idx_t thread_c = idx_t({ prec ? D[0] : D[i], i });
    A_col_i[0] = pp_func(rsq.real, prec ? A_col_j[0] : A_col_j[i], thread_c.real);
    if (0 < j) { A_col_i[j] = conj<real_t>(A_col_j[i] = A[i]); }
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  if (0 < j) for (int32_t i = offset - (1 + M); i < 0; i += elements)
  { matrix_t a = A[i]; A[i] = A_col_j[i]; A_col_j[i] = a; }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
  if (BLOCK_THREADS == elements) {
    if (int32_t(threadIdx.x) == 0) { A[++lda] = real_sqrt<matrix_t>(D[N], thread_x, idx[0]); work->idx = --thread_x.idx; if (0 < j) { int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }}
    return;
  }

  if (int32_t(threadIdx.x) == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
  grid.sync();
  if (int32_t(blockIdx.x) == 0) {
    for (int32_t i = int32_t(threadIdx.x); i < int32_t(gridDim.x); i += BLOCK_THREADS)
    { thread_x = cmp_max(thread_x, work[i]); }
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce).Reduce(thread_x, cmp_max);
    if (int32_t(threadIdx.x) == 0) { A[++lda] = real_sqrt<matrix_t>(D[N], thread_x, idx[0]); work->idx = --thread_x.idx; if (0 < j) { int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }}
  }
}

constexpr int32_t grid_blocks = 2048;
constexpr int32_t block_threads = 128;

template <class real_t, class matrix_t, class idx_t>
inline int32_t imax_dispatcher(cudaStream_t stream, real_t epi, int32_t p, int32_t N, matrix_t* X, int64_t incx, int32_t* jpiv, real_t* D, idx_t* idx) {
  if (1 <= N) {
    int32_t device_sms = internal::device_num_sms(), maxBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, imax_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
    uint8_t* diag = &((uint8_t*)D)[65536];
    void* kernelArgs[]{ &epi, &p, &N, &X, &incx, &jpiv, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(imax_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);

    int32_t blocks_p = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_p, gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    return std::min(grid_blocks, device_sms * blocks_p);
  } else { return 0; }
}

template <class real_t, class matrix_t, class idx_t>
inline void gemv_pp_dispatcher(cudaStream_t stream, int32_t grid, int32_t M, int32_t N, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, idx_t* idx) {
  if (2 <= N) {
    uint8_t* diag = &((uint8_t*)D)[int64_t(65536) + int64_t(M) * int64_t(sizeof(real_t))];
    grid = std::min(grid, std::max(N - 1, (M + block_threads - 1) / block_threads));
    void* kernelArgs[]{ &M, &N, &A, &lda, &jpiv, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  }
}

namespace internal::Cholesky {

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, double* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale)
  { return imax_dispatcher(stream, epi, p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, float* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale)
  { return imax_dispatcher(stream, float(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale)
  { return imax_dispatcher(stream, device::dd::double2dd(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale)
  { return imax_dispatcher(stream, device::qf::double2qf(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, cuDoubleComplex* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale)
  { return imax_dispatcher(stream, epi, p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, cuComplex* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale)
  { return imax_dispatcher(stream, float(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale)
  { return imax_dispatcher(stream, device::dd::double2dd(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  int32_t imax_initializer(cudaStream_t stream, double epi, int32_t p, int32_t N, complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale)
  { return imax_dispatcher(stream, device::qf::double2qf(epi), p, N, X, int64_t(incx), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, double_idx* scale, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, float_idx* scale, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, double2_idx* scale, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, float4_idx* scale, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, double_idx* scale, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, float_idx* scale, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, double2_idx* scale, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, int32_t grid, float4_idx* scale, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_pp_dispatcher(stream, grid, M, N, A, int64_t(lda), jpiv, D, scale); }

}
