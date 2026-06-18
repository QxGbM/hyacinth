
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>
#include <float_max.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>
#include <cooperative_groups.h>

__device__ __forceinline__ void real_sqrt(double epi, double& epi_d, double_idx x, double_idx& sq, double_idx& rsq) {
  double sqx = sqrt(x.real); epi_d = epi = epi * sqx;
  sq = double_idx({ sqx, x.idx }); rsq = double_idx({ 1. / sqx, int32_t(sqx < epi) });
}
__device__ __forceinline__ void real_sqrt(float epi, float& epi_d, float_idx x, float_idx& sq, float_idx& rsq) {
  float sqx = sqrtf(x.real); epi_d = epi = epi * sqx;
  sq = float_idx({ sqx, x.idx }); rsq = float_idx({ 1.f / sqx, int32_t(sqx < epi) });
}
__device__ __forceinline__ void real_sqrt(double2 epi, double2& epi_d, double2_idx x, double2_idx& sq, double2_idx& rsq) {
  double2 sqx, rsqx; device::dd::frsqrt(x.real, sqx, rsqx);
  epi_d = epi = device::dd::mul(epi, sqx);
  bool less, par; device::cmp::cmp_double2(sqx, epi, less, par);
  sq = double2_idx({ sqx, x.idx }); rsq = double2_idx({ rsqx, int32_t(less) });
}
__device__ __forceinline__ void real_sqrt(float4 epi, float4& epi_d, float4_idx x, float4_idx& sq, float4_idx& rsq) {
  float4 sqx, rsqx; device::qf::frsqrt(x.real, sqx, rsqx);
  epi_d = epi = device::qf::mul(epi, sqx);
  bool less, par; device::cmp::cmp_float4(sqx, epi, less, par);
  sq = float4_idx({ sqx, x.idx }); rsq = float4_idx({ rsqx, int32_t(less) });
}

template <int32_t BLOCK_THREADS, class real_t, class idx_t>
__global__ void imax_kernel(real_t epi, int32_t N, const real_t* __restrict__ X, int64_t incx, int32_t* __restrict__ jpiv, real_t* __restrict__ D, idx_t* __restrict__ idx) {
  const int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS, elements = int32_t(gridDim.x) * BLOCK_THREADS;
  idx_t thread_x = idx_t();

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
  device::cmp::idx_max cmp_max;

  for (int32_t i = block_offset + int32_t(threadIdx.x); i < N; i += elements)
    thread_x = cmp_max(thread_x, idx_t({ D[i] = X[int64_t(i) * incx], jpiv[i] = i + 1 }));

  thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(thread_x, cmp_max);
  if (int32_t(gridDim.x) == 1)
  { if (threadIdx.x == 0) real_sqrt(epi, D[0], thread_x, idx[0], idx[1]); return; }

  if (threadIdx.x == 0) idx[blockIdx.x] = thread_x;
    else thread_x = idx_t();
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    for (int32_t i = threadIdx.x; i < int32_t(gridDim.x); i += BLOCK_THREADS)
      thread_x = cmp_max(thread_x, idx[i]);
    thread_x = cub::BlockReduce<idx_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(thread_x, cmp_max);
    if (threadIdx.x == 0) { real_sqrt(epi, D[0], thread_x, idx[0], idx[1]); }
  }
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

template <class real_t, class idx_t>
inline void imax_dispatcher(cudaStream_t stream, real_t epi, int32_t N, const real_t* X, int64_t incx, int32_t* jpiv, real_t* D, idx_t* idx) {
  int32_t device_sms = 0, device = -1;
  cudaGetDevice(&device); cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
  int32_t maxBlocksPerSM = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, imax_kernel<block_threads, real_t, idx_t>, block_threads, 0);
  int32_t grid = std::min(std::min(grid_blocks, device_sms * maxBlocksPerSM), (N + block_threads - 1) / block_threads);
  void* kernelArgs[]{ &epi, &N, &X, &incx, &jpiv, &D, &idx };
  cudaLaunchCooperativeKernel(imax_kernel<block_threads, real_t, idx_t>, grid, block_threads, kernelArgs, 0, stream);
  cudaStreamSynchronize(stream);
}

void internal::Cholesky::imax_f64(cudaStream_t stream, double epi, int32_t N, const double* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale) {
  imax_dispatcher(stream, std::min(1., std::max(0., std::abs(epi))), N, X, int64_t(incx), jpiv, D, scale);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, double epi, int32_t N, const float* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale) {
  imax_dispatcher(stream, float(std::min(1., std::max(0., std::abs(epi)))), N, X, int64_t(incx), jpiv, D, scale);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, double epi, int32_t N, const double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale) {
  imax_dispatcher(stream, device::dd::double2dd(std::min(1., std::max(0., std::abs(epi)))), N, X, int64_t(incx), jpiv, D, scale);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, double epi, int32_t N, const float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale) {
  imax_dispatcher(stream, device::qf::double2qf(std::min(1., std::max(0., std::abs(epi)))), N, X, int64_t(incx), jpiv, D, scale);
}

void internal::Cholesky::imax_cf64(cudaStream_t stream, double epi, int32_t N, const std::complex<double>* X, int32_t incx, int32_t* jpiv, double* D, double_idx* scale) {
  imax_dispatcher(stream, std::min(1., std::max(0., std::abs(epi))), N, (const double*)X, int64_t(incx) << 1, jpiv, D, scale);
}

void internal::Cholesky::imax_cf32(cudaStream_t stream, double epi, int32_t N, const std::complex<float>* X, int32_t incx, int32_t* jpiv, float* D, float_idx* scale) {
  imax_dispatcher(stream, float(std::min(1., std::max(0., std::abs(epi)))), N, (const float*)X, int64_t(incx) << 1, jpiv, D, scale);
}

void internal::Cholesky::imax_cf128_dd(cudaStream_t stream, double epi, int32_t N, const complex_double2* X, int32_t incx, int32_t* jpiv, double2* D, double2_idx* scale) {
  imax_dispatcher(stream, device::dd::double2dd(std::min(1., std::max(0., std::abs(epi)))), N, (const double2*)X, int64_t(incx) << 1, jpiv, D, scale);
}

void internal::Cholesky::imax_cf128_qf(cudaStream_t stream, double epi, int32_t N, const complex_float4* X, int32_t incx, int32_t* jpiv, float4* D, float4_idx* scale) {
  imax_dispatcher(stream, device::qf::double2qf(std::min(1., std::max(0., std::abs(epi)))), N, (const float4*)X, int64_t(incx) << 1, jpiv, D, scale);
}
