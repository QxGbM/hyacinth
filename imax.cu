
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

template <class real_t> struct real_pair { real_t real; int32_t idx; };

template <class real_t> struct real_pair_max {
  struct cmp_less_w_parity {
    __device__ __forceinline__ void operator()(double a, double b, bool& less, bool& par) {
      less = a < b; par = a == b; }
    __device__ __forceinline__ void operator()(float a, float b, bool& less, bool& par) {
      less = a < b; par = a == b; }
    __device__ __forceinline__ void operator()(double2 a, double2 b, bool& less, bool& par) {
      bool l1 = a.x < b.x, l2 = a.y < b.y;
      bool p1 = a.x == b.x;
      less = l1 || (p1 && l2); par = p1 && (a.y == b.y); 
    }
    __device__ __forceinline__ void operator()(float4 a, float4 b, bool& less, bool& par) {
      bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w;
      bool p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z);
      less = l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); par = p3 && (a.w == b.w);
    }
  };

  __device__ __forceinline__ real_pair<real_t> operator()(real_pair<real_t> e1, real_pair<real_t> e2) const {
    cmp_less_w_parity cmp_func; bool less, par;
    cmp_func(e1.real, e2.real, less, par);
    real_pair<real_t> val = less ? e2 : e1;
    int32_t id_tie = min(e1.idx, e2.idx);
    return real_pair<real_t>({ val.real, par ? id_tie : val.idx });
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
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t thread_data[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS> block_reduce(temp_reduce);
  real_pair_max<real_t> cmp_max;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_pair[i] = real_pair<real_t>({ real_t(), -1 });

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + i;
    block_load.Load(&X[i], thread_data, num_items, real_t());

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], thread_loc + j }));
  }

  real_pair<real_t> block_res = block_reduce.Reduce(thread_pair, cmp_max);

  if (threadIdx.x == 0) {
    rsqrt_real rsqrt_func;
    *i_out = block_res.idx;
    *rsq_out = rsqrt_func(block_res.real);
    X[block_res.idx] = X[0];
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
