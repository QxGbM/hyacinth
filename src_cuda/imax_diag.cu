
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <numeric>
#include <cub/cub.cuh>
#include <cuComplex.h>

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

  idx_t block_res;
  cmp_max.init(block_res);
  if (block_offset < N)
    block_res = block_reduce.Reduce(thread_i, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0)
    idx[blockIdx.x] = block_res;
}

constexpr int32_t grid_blocks = 128;
constexpr int32_t block_threads = 128;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, double* X, double* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  double_idx* p = (double_idx*)diag_piv, init({ 0., -1 });
  imax_kernel <double, const double* __restrict__, double_idx, double_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, p);
  cudaStreamSynchronize(stream);
  double_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
  p[0] = (0 <= res.idx && res.idx < N) ? res : init;
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, float* X, float* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  float_idx* p = (float_idx*)diag_piv, init({ 0.f, -1 });
  imax_kernel <float, const float* __restrict__, float_idx, float_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, p);
  cudaStreamSynchronize(stream);
  float_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
  p[0] = (0 <= res.idx && res.idx < N) ? res : init;
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, double2* X, double2* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  double2_idx* p = (double2_idx*)diag_piv, init({ make_double2(0., 0.), -1 });
  imax_kernel <double2, const double2* __restrict__, double2_idx, double2_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, p);
  cudaStreamSynchronize(stream);
  double2_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
  p[0] = (0 <= res.idx && res.idx < N) ? res : init;
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, float4* X, float4* diag_piv) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  float4_idx* p = (float4_idx*)diag_piv, init({ make_float4(0.f, 0.f, 0.f, 0.f), -1 });
  imax_kernel <float4, const float4* __restrict__, float4_idx, float4_idx* __restrict__, grid_blocks, block_threads, items_per_thread>
    <<< grid_blocks, block_threads, 0, stream >>> (N, X, p);
  cudaStreamSynchronize(stream);
  float4_idx res = std::reduce(p, &p[grid_blocks], init, real_max());
  p[0] = (0 <= res.idx && res.idx < N) ? res : init;
}
