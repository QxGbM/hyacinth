
#include <internal.hpp>
#include <float_max.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }
};

template <class idx_t, class idx_ptr, class real_ptr, class real_const_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS>
__global__ void imax_kernel(int32_t N, real_const_ptr X, int64_t incx, real_ptr D, idx_ptr idx) {
  constexpr int32_t elements = GRID_BLOCKS *BLOCK_THREADS;
  int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS;
  idx_t thread_x = idx_t();

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  real_max cmp_max;

  for (int32_t i = block_offset + int32_t(threadIdx.x); i < N; i += elements)
    thread_x = cmp_max(thread_x, idx_t({ D[i] = X[int64_t(i) * incx], i + 1 }));

  thread_x = block_reduce.Reduce(thread_x, cmp_max);
  if (threadIdx.x == 0)
    idx[blockIdx.x] = thread_x;
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, const double* X, int32_t incx, double* D, double* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<double_idx, double_idx* __restrict__, double* __restrict__, const double* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, X, int64_t(incx), D, (double_idx*)scale);
  imax_f64_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, const float* X, int32_t incx, float* D, float* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<float_idx, float_idx* __restrict__, float* __restrict__, const float* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, X, int64_t(incx), D, (float_idx*)scale);
  imax_f32_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, const double2* X, int32_t incx, double2* D, double2* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<double2_idx, double2_idx* __restrict__, double2* __restrict__, const double2* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, X, int64_t(incx), D, (double2_idx*)scale);
  imax_f128_dd_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, const float4* X, int32_t incx, float4* D, float4* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<float4_idx, float4_idx* __restrict__, float4* __restrict__, const float4* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, X, int64_t(incx), D, (float4_idx*)scale);
  imax_f128_qf_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_cf64(cudaStream_t stream, int32_t N, const std::complex<double>* X, int32_t incx, double* D, double* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<double_idx, double_idx* __restrict__, double* __restrict__, const double* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, (const double*)X, int64_t(incx) << 1, D, (double_idx*)scale);
  imax_f64_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_cf32(cudaStream_t stream, int32_t N, const std::complex<float>* X, int32_t incx, float* D, float* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<float_idx, float_idx* __restrict__, float* __restrict__, const float* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, (const float*)X, int64_t(incx) << 1, D, (float_idx*)scale);
  imax_f32_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_cf128_dd(cudaStream_t stream, int32_t N, const complex_double2* X, int32_t incx, double2* D, double2* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<double2_idx, double2_idx* __restrict__, double2* __restrict__, const double2* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, (const double2*)X, int64_t(incx) << 1, D, (double2_idx*)scale);
  imax_f128_dd_host_sync(stream, N + 1, grid, scale);
}

void internal::Cholesky::imax_cf128_qf(cudaStream_t stream, int32_t N, const complex_float4* X, int32_t incx, float4* D, float4* scale) {
  int32_t grid = std::min(grid_blocks, (N + block_threads - 1) / block_threads);
  imax_kernel<float4_idx, float4_idx* __restrict__, float4* __restrict__, const float4* __restrict__, grid_blocks, block_threads>
    <<< grid, block_threads, 0, stream >>> (N, (const float4*)X, int64_t(incx) << 1, D, (float4_idx*)scale);
  imax_f128_qf_host_sync(stream, N + 1, grid, scale);
}
