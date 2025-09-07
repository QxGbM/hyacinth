
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

template <class real_t, class real_const_ptr, class idx_t, class idx_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void imax_kernel(int32_t N, real_const_ptr X, idx_ptr idx) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  constexpr int32_t block_mask = ~(elements_block - 1) & (elements - 1);

  int32_t block_offset = int32_t(blockIdx.x) * elements_block;
  int32_t N2 = N & (elements_block - 1), N1 = N - N2;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<idx_t, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t thread_x[ITEMS_PER_THREAD]; idx_t thread_i[ITEMS_PER_THREAD]; int32_t thread_locs[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load);
  cub::BlockReduce<idx_t, BLOCK_THREADS> block_reduce(temp_reduce);
  real_max cmp_max;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_locs[i] = int32_t(threadIdx.x) + i * BLOCK_THREADS;

  for (int32_t k = block_offset; k < N1; k += elements) {
    block_load.Load(&X[k], thread_x);

    if (k == block_offset) {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_i[i] = idx_t({ thread_x[i], k + thread_locs[i] });
    }
    else {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_i[i] = cmp_max(thread_i[i], idx_t({ thread_x[i], k + thread_locs[i] }));
    }
  }

  if (0 < N2 && block_offset == (N1 & block_mask)) {
    block_load.Load(&X[N1], thread_x, N2, real_t());

    if (N1 == block_offset) {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_i[i] = idx_t({ thread_x[i], N1 + thread_locs[i] });
    }
    else {
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        thread_i[i] = cmp_max(thread_i[i], idx_t({ thread_x[i], N1 + thread_locs[i] }));
    }
  }

  idx_t block_res; cmp_max.init(block_res);
  if (block_offset < N)
    block_res = block_reduce.Reduce(thread_i, cmp_max);

  if (threadIdx.x == 0)
    idx[blockIdx.x] = block_res;
}

constexpr int32_t grid_blocks = 256;
constexpr int32_t block_threads = 256;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, const double* X, double* scale) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <double, const double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (double_idx*)scale);
  imax_f64_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, const float* X, float* scale) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <float, const float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (float_idx*)scale);
  imax_f32_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, const double2* X, double2* scale) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <double2, const double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (double2_idx*)scale);
  imax_f128_dd_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, const float4* X, float4* scale) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  constexpr int32_t elements_block = block_threads * items_per_thread;
  imax_kernel <float4, const float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, (float4_idx*)scale);
  imax_f128_qf_host_sync(stream, N, std::min(grid_blocks, (N + elements_block - 1) / elements_block), scale);
}
