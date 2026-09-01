
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

__device__ __forceinline__ void real_sqrt(double epi, double_idx x, double_idx& sq, double_idx& rsq) {
  double sqx = sqrt(x.real);
  sq = double_idx({ sqx, x.idx - 1 }); rsq = double_idx({ 1. / sqx, int32_t(!(epi <= x.real)) });
}
__device__ __forceinline__ void real_sqrt(float epi, float_idx x, float_idx& sq, float_idx& rsq) {
  float sqx = sqrtf(x.real);
  sq = float_idx({ sqx, x.idx - 1 }); rsq = float_idx({ 1.f / sqx, int32_t(!(epi <= x.real)) });
}
__device__ __forceinline__ void real_sqrt(double2 epi, double2_idx x, double2_idx& sq, double2_idx& rsq) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_double2(epi, x.real, less, par);
  sq = double2_idx({ sqx, x.idx - 1 }); rsq = double2_idx({ rsqx, int32_t(!(less || par)) });
}
__device__ __forceinline__ void real_sqrt(float4 epi, float4_idx x, float4_idx& sq, float4_idx& rsq) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_float4(epi, x.real, less, par);
  sq = float4_idx({ sqx, x.idx - 1 }); rsq = float4_idx({ rsqx, int32_t(!(less || par)) });
}

__device__ __forceinline__ double _mul(double a, double b) { return a * b; }
__device__ __forceinline__ float _mul(float a, float b) { return a * b; }
__device__ __forceinline__ double2 _mul(double2 a, double2 b) { return device::dd::mul(a, b); }
__device__ __forceinline__ float4 _mul(float4 a, float4 b) { return device::qf::mul(a, b); }

template <int32_t BLOCK_THREADS, class real_t, class idx_t>
__global__ void imax_kernel(real_t epi, int32_t N, const real_t* __restrict__ X, int64_t incx, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, idx_t* __restrict__ idx) {
  const int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS, elements = int32_t(gridDim.x) * BLOCK_THREADS;
  idx_t thread_x = idx_t();

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  device::cmp::idx_max cmp_max;

  for (int32_t i = block_offset + int32_t(threadIdx.x); i < N; i += elements)
    thread_x = cmp_max(thread_x, idx_t({ D[i] = X[int64_t(i) * incx], jpiv[i] = i + 1 }));

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_x, cmp_max);
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) { real_sqrt(D[0] = _mul(epi, thread_x.real), thread_x, idx[0], idx[1]); }
    return;
  }

  if (threadIdx.x == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, work[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(D[0] = _mul(epi, thread_x.real), thread_x, idx[0], idx[1]); }
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

template <class real_t, class matrix_t> __device__ __forceinline__ matrix_t conj(matrix_t a) { return a; }
template <> __device__ __forceinline__ cuDoubleComplex conj<double, cuDoubleComplex>(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
template <> __device__ __forceinline__ cuComplex conj<float, cuComplex>(cuComplex a) { return make_cuComplex(a.x, -a.y); }
template <> __device__ __forceinline__ complex_double2 conj<double2, complex_double2>(complex_double2 a) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); }
template <> __device__ __forceinline__ complex_float4 conj<float4, complex_float4>(complex_float4 a) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); }

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void gemv_pp_kernel(int32_t j, int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_t* __restrict__ A, int64_t lda, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ work, idx_t* __restrict__ idx) {
  const int32_t offset = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x) + 1, elements = int32_t(gridDim.x) * BLOCK_THREADS;
  matrix_t* A_col_j = &A[int64_t(j) * lda];
  for (int32_t i = offset - (1 + M); i < 0; i += elements)
  { matrix_t a = A[i]; A[i] = A_col_j[i]; A_col_j[i] = a; }

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  device::cmp::idx_max cmp_max;
  idx_t thread_x = idx_t();

  for (int32_t i = offset; i < N; i += elements) {
    matrix_t* A_col_i = &A[int64_t(i) * lda];
    bool prec = i == j;
    idx_t thread_c = idx_t({ prec ? D[0] : D[i], i });
    A_col_i[0] = pp_func(rsq, prec ? A_col_j[0] : A_col_j[i], thread_c.real);
    A_col_i[j] = conj<real_t>(A_col_j[i] = A[i]);
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_x, cmp_max);
  if (gridDim.x == 1) { 
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); A[0] = sq; int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }
    return;
  }

  if (threadIdx.x == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, work[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); A[0] = sq; int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }
  }
}

template <int32_t BLOCK_THREADS, class real_t, class matrix_t, class idx_t>
__global__ void gemv_pp_nopiv_kernel(int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_t* __restrict__ A, int64_t lda, real_t* __restrict__ D, idx_t* __restrict__ work, idx_t* __restrict__ idx) {
  const int32_t offset = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x) + 1, elements = int32_t(gridDim.x) * BLOCK_THREADS;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  device::cmp::idx_max cmp_max;
  idx_t thread_x = idx_t();

  for (int32_t i = offset; i < N; i += elements) {
    idx_t thread_c = idx_t({ D[i], i });
    A[int64_t(i) * lda] = pp_func(rsq, A[i], thread_c.real);
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_x, cmp_max);
  if (gridDim.x == 1) {
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); A[0] = sq; }
    return;
  }

  if (threadIdx.x == 0) { work[blockIdx.x] = thread_x; } else { thread_x = idx_t(); }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, work[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); A[0] = sq; }
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

template <class real_t, class matrix_t, class idx_t>
inline void imax_dispatcher(cudaStream_t stream, real_t epi, int32_t N, const matrix_t* X, int64_t incx, int32_t* jpiv, real_t* D, idx_t* idx) {
  int32_t device_sms = 0, device = -1;
  cudaGetDevice(&device); cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
  int32_t maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, imax_kernel<block_threads, real_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  uint8_t* diag = &((uint8_t*)D)[8192]; if constexpr(!std::is_same_v<real_t, matrix_t>) { incx <<= 1; }
  void* kernelArgs[]{ &epi, &N, &X, &incx, &jpiv, &diag, &D, &idx };
  cudaLaunchCooperativeKernel(imax_kernel<block_threads, real_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);

  int32_t blocks_p = 0, blocks_np = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_p, gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_np, gemv_pp_nopiv_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
  idx[2].idx = std::min(grid_blocks, device_sms * blocks_p);
  idx[3].idx = std::min(grid_blocks, device_sms * blocks_np);
  cudaStreamSynchronize(stream);
}

template <class real_t, class matrix_t, class idx_t>
inline void gemv_pp_dispatcher(cudaStream_t stream, int32_t j, int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, idx_t* idx) {
  uint8_t* diag = &((uint8_t*)D)[int64_t(8192) + int64_t(M) * sizeof(real_t)]; A = &A[M]; jpiv = &jpiv[M];
  if (j) {
    int32_t grid = std::min(idx[2].idx, std::max(N + block_threads - 2, M + block_threads - 1) / block_threads);
    void* kernelArgs[]{ &j, &M, &N, &sq, &rsq, &A, &lda, &jpiv, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  } else {
    int32_t grid = std::min(idx[3].idx, (N + block_threads - 2) / block_threads);
    void* kernelArgs[]{ &M, &N, &sq, &rsq, &A, &lda, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(gemv_pp_nopiv_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  }
  cudaStreamSynchronize(stream);
}

namespace internal::Cholesky {

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const double* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale)
  { imax_dispatcher(stream, epi, N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const float* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale)
  { imax_dispatcher(stream, float(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale)
  { imax_dispatcher(stream, device::dd::double2dd(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale)
  { imax_dispatcher(stream, device::qf::double2qf(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const cuDoubleComplex* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale)
  { imax_dispatcher(stream, epi, N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const cuComplex* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale)
  { imax_dispatcher(stream, float(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale)
  { imax_dispatcher(stream, device::dd::double2dd(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void imax_initializer(cudaStream_t stream, double epi, int32_t N, const complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale)
  { imax_dispatcher(stream, device::qf::double2qf(epi), N, X, int64_t(incx), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, cuDoubleComplex* A, int32_t lda, int32_t* jpiv, double* D)
  { gemv_pp_dispatcher(stream, j, M, N, make_cuDoubleComplex(scale[0].real, 0.), scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, cuComplex* A, int32_t lda, int32_t* jpiv, float* D)
  { gemv_pp_dispatcher(stream, j, M, N, make_cuComplex(scale[0].real, 0.f), scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D)
  { gemv_pp_dispatcher(stream, j, M, N, device::dd::make_complex_double2(scale[0].real, make_double2(0., 0.)), scale[1].real, A, int64_t(lda), jpiv, D, scale); }

  void gemv_pp(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D)
  { gemv_pp_dispatcher(stream, j, M, N, device::qf::make_complex_float4(scale[0].real, make_float4(0.f, 0.f, 0.f, 0.f)), scale[1].real, A, int64_t(lda), jpiv, D, scale); }

}
