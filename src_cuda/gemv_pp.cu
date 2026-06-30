
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>

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

__device__ __forceinline__ cuDoubleComplex conj(cuDoubleComplex a) { return make_cuDoubleComplex(a.x, -a.y); }
__device__ __forceinline__ cuComplex conj(cuComplex a) { return make_cuComplex(a.x, -a.y); }
__device__ __forceinline__ complex_double2 conj(complex_double2 a) { return device::dd::make_complex_double2(a.real, device::dd::negate(a.imag)); }
__device__ __forceinline__ complex_float4 conj(complex_float4 a) { return device::qf::make_complex_float4(a.real, device::qf::negate(a.imag)); }

__device__ __forceinline__ void real_sqrt(double epi, double_idx x, double_idx& sq, double_idx& rsq) {
  double sqx = sqrt(x.real);
  sq = double_idx({ sqx, x.idx }); rsq = double_idx({ 1. / sqx, int32_t(!(epi <= sqx)) });
}
__device__ __forceinline__ void real_sqrt(float epi, float_idx x, float_idx& sq, float_idx& rsq) {
  float sqx = sqrtf(x.real);
  sq = float_idx({ sqx, x.idx }); rsq = float_idx({ 1.f / sqx, int32_t(!(epi <= sqx)) });
}
__device__ __forceinline__ void real_sqrt(double2 epi, double2_idx x, double2_idx& sq, double2_idx& rsq) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_double2(epi, sqx, less, par);
  sq = double2_idx({ sqx, x.idx }); rsq = double2_idx({ rsqx, int32_t(!(less || par)) });
}
__device__ __forceinline__ void real_sqrt(float4 epi, float4_idx x, float4_idx& sq, float4_idx& rsq) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx);
  bool less, par; device::cmp::cmp_float4(epi, sqx, less, par);
  sq = float4_idx({ sqx, x.idx }); rsq = float4_idx({ rsqx, int32_t(!(less || par)) });
}

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
    if constexpr(sizeof(real_t) < sizeof(matrix_t))
      A_col_i[j] = conj(A_col_j[i] = A[i]);
    else
      A_col_i[j] = A_col_j[i] = A[i];
    thread_x = cmp_max(thread_x, thread_c);
    D[i] = thread_c.real;
  }

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_x, cmp_max);
  if (gridDim.x == 1)
  { if (threadIdx.x == 0) real_sqrt(D[-M], thread_x, idx[0], idx[1]); return; }

  if (threadIdx.x == 0) {
    work[blockIdx.x] = thread_x;
    if (blockIdx.x == 0)
    { A[0] = sq; int32_t p = jpiv[0]; jpiv[0] = jpiv[j]; jpiv[j] = p; }
  } else { thread_x = idx_t(); }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, work[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); }
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
  if (gridDim.x == 1)
  { if (threadIdx.x == 0) real_sqrt(D[-M], thread_x, idx[0], idx[1]); return; }

  if (threadIdx.x == 0) {
    work[blockIdx.x] = thread_x;
    if (blockIdx.x == 0) { A[0] = sq; }
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < gridDim.x; i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, work[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(D[-M], thread_x, idx[0], idx[1]); }
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

template <class real_t, class matrix_t, class idx_t>
inline void gemv_pp_dispatcher(cudaStream_t stream, int32_t j, int32_t M, int32_t N, matrix_t sq, real_t rsq, matrix_t* A, int64_t lda, int32_t* jpiv, real_t* D, idx_t* idx) {
  int32_t device_sms = 0, device = -1; cudaGetDevice(&device); cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
  int32_t maxBlocksPerSM = 0;
  uint8_t* diag = &((uint8_t*)D)[int64_t(8192) + int64_t(M) * sizeof(real_t)]; A = &A[M]; jpiv = &jpiv[M];
  if (j) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), std::max(N + block_threads - 2, M + block_threads - 1) / block_threads);
    void* kernelArgs[]{ &j, &M, &N, &sq, &rsq, &A, &lda, &jpiv, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(gemv_pp_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  }
  else {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, gemv_pp_nopiv_kernel<block_threads, real_t, matrix_t, idx_t>, block_threads, 0);
    int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 2) / block_threads);
    void* kernelArgs[]{ &M, &N, &sq, &rsq, &A, &lda, &diag, &D, &idx };
    cudaLaunchCooperativeKernel(gemv_pp_nopiv_kernel<block_threads, real_t, matrix_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  }
}

void internal::Cholesky::gemv_pp_f64(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* D) {
  gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_f32(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* D) {
  gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_f128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_f128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  gemv_pp_dispatcher(stream, j, M, N, scale[0].real, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_cf64(cudaStream_t stream, double_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, double* D) {
  cuDoubleComplex sqc = make_cuDoubleComplex(scale[0].real, 0.);
  gemv_pp_dispatcher(stream, j, M, N, sqc, scale[1].real, (cuDoubleComplex*)A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_cf32(cudaStream_t stream, float_idx* scale, int32_t j, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, float* D) {
  cuComplex sqc = make_cuComplex(scale[0].real, 0.f);
  gemv_pp_dispatcher(stream, j, M, N, sqc, scale[1].real, (cuComplex*)A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_cf128_dd(cudaStream_t stream, double2_idx* scale, int32_t j, int32_t M, int32_t N, complex_double2* A, int32_t lda, int32_t* jpiv, double2* D) {
  complex_double2 sqc = device::dd::make_complex_double2(scale[0].real, make_double2(0., 0.));
  gemv_pp_dispatcher(stream, j, M, N, sqc, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::gemv_pp_cf128_qf(cudaStream_t stream, float4_idx* scale, int32_t j, int32_t M, int32_t N, complex_float4* A, int32_t lda, int32_t* jpiv, float4* D) {
  complex_float4 sqc = device::qf::make_complex_float4(scale[0].real, make_float4(0.f, 0.f, 0.f, 0.f));
  gemv_pp_dispatcher(stream, j, M, N, sqc, scale[1].real, A, int64_t(lda), jpiv, D, scale);
  cudaStreamSynchronize(stream);
}
