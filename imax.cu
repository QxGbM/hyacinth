
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

template <class real_t> struct real_pair {
  real_t first;
  int32_t second;
};

template <class real_t> struct real_pair_max {
  struct less_real {
    __device__ __forceinline__ bool operator()(double a, double b) { return a < b; }
    __device__ __forceinline__ bool operator()(float a, float b) { return a < b; }
    __device__ __forceinline__ bool operator()(double2 a, double2 b) { return device::dd::a_less_than_b(a, b); }
    __device__ __forceinline__ bool operator()(float4 a, float4 b) { return device::qf::a_less_than_b(a, b); }
  };

  struct eq_real {
    __device__ __forceinline__ bool operator()(double a, double b) { return a == b; }
    __device__ __forceinline__ bool operator()(float a, float b) { return a == b; }
    __device__ __forceinline__ bool operator()(double2 a, double2 b) { return device::dd::a_eq_to_b(a, b); }
    __device__ __forceinline__ bool operator()(float4 a, float4 b) { return device::qf::a_eq_to_b(a, b); }
  };

  __device__ __forceinline__ real_pair<real_t> operator()(real_pair<real_t> e1, real_pair<real_t> e2) const {
    less_real cmp_less; eq_real cmp_eq;
    real_pair<real_t> val = cmp_less(e1.first, e2.first) ? e2 : e1;
    int32_t id_tie = min(e1.second, e2.second);
    return real_pair<real_t>({ val.first, cmp_eq(e1.first, e2.first) ? id_tie : val.second });
  }
};

struct rsqrt_real {
  __device__ __forceinline__ double operator()(double f) { return rsqrt(f); }
  __device__ __forceinline__ float operator()(float f) { return rsqrtf(f); }
  __device__ __forceinline__ double2 operator()(double2 f) { return device::dd::frsqrt(f); }
  __device__ __forceinline__ float4 operator()(float4 f) { return device::qf::frsqrt(f); }
};

template <class real_t, class real_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void reduce_real(int32_t N, real_ptr X, int32_t* i_out, real_ptr rsq_out) {
  using BlockLoad = cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockReduce = cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>;
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename BlockLoad::TempStorage temp_load;
  __shared__ typename BlockReduce::TempStorage temp_reduce;

  real_t thread_data[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];
  real_pair_max<real_t> cmp_max;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_pair[i] = real_pair<real_t>({ real_t(), -1 });

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + i;
    BlockLoad(temp_load).Load(&X[i], thread_data, num_items, real_t());

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], thread_loc + j }));
  }

  real_pair<real_t> block_res = BlockReduce(temp_reduce).Reduce(thread_pair, cmp_max);

  if (threadIdx.x == 0) {
    rsqrt_real rsqrt_func;
    *i_out = block_res.second;
    *rsq_out = rsqrt_func(block_res.first);
    X[block_res.second] = X[0];
  }
}

constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_double(cudaStream_t stream, int32_t N, double* X, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  reduce_real <double, double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, rsq);
}

void internal::Cholesky::imax_float(cudaStream_t stream, int32_t N, float* X, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  reduce_real <float, float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, rsq);
}

void internal::Cholesky::imax_double2(cudaStream_t stream, int32_t N, double2* X, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  reduce_real <double2, double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, rsq);
}

void internal::Cholesky::imax_float4(cudaStream_t stream, int32_t N, float4* X, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  reduce_real <float4, float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, rsq);
}
