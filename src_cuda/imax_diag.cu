
#include <internal.hpp>
#include <float_max.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct real_max {
  __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }

  __host__ __device__ __forceinline__ void init(double_idx& a) { a = double_idx({ 0., -1 }); }
  __host__ __device__ __forceinline__ void init(float_idx& a) { a = float_idx({ 0.f, -1 }); }
  __host__ __device__ __forceinline__ void init(double2_idx& a) { a = double2_idx({ make_double2(0., 0.), -1 }); }
  __host__ __device__ __forceinline__ void init(float4_idx& a) { a = float4_idx({ make_float4(0.f, 0.f, 0.f, 0.f), -1 }); }
};

template <class idx_t, class idx_ptr, class real_const_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS>
__global__ void imax_kernel(int32_t N, real_const_ptr X, idx_ptr idx) {
  constexpr int32_t elements = GRID_BLOCKS *BLOCK_THREADS;
  int32_t block_offset = int32_t(blockIdx.x) * BLOCK_THREADS;
  idx_t thread_x;

  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  real_max cmp_max;
  cmp_max.init(thread_x);

  for (int32_t i = block_offset + int32_t(threadIdx.x); i < N; i += elements)
    thread_x = cmp_max(thread_x, idx_t({ X[i], i }));

  if (block_offset < N)
    thread_x = block_reduce.Reduce(thread_x, cmp_max);

  if (threadIdx.x == 0)
    idx[blockIdx.x] = thread_x;
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, const double* X, double* scale) {
  constexpr int32_t items_per_thread = 1;
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <double_idx, double_idx* __restrict__, const double* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (double_idx*)scale);
  imax_f64_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, const float* X, float* scale) {
  constexpr int32_t items_per_thread = 1;
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <float_idx, float_idx* __restrict__, const float* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (float_idx*)scale);
  imax_f32_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, const double2* X, double2* scale) {
  constexpr int32_t items_per_thread = 1;
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <double2_idx, double2_idx* __restrict__, const double2* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (double2_idx*)scale);
  imax_f128_dd_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, const float4* X, float4* scale) {
  constexpr int32_t items_per_thread = 1;
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <float4_idx, float4_idx* __restrict__, const float4* __restrict__, grid_blocks, block_threads>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (float4_idx*)scale);
  imax_f128_qf_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}
