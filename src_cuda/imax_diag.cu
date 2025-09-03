
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

template <class real_t> struct real_pair { real_t real; int32_t idx; };

template <class real_t> struct real_pair_max {
  __host__ __device__ __forceinline__ void cmp_less_w_parity(double a, double b, bool& less, bool& par) {
    less = a < b; par = a == b; }
  __host__ __device__ __forceinline__ void cmp_less_w_parity(float a, float b, bool& less, bool& par) {
    less = a < b; par = a == b; }
  __host__ __device__ __forceinline__ void cmp_less_w_parity(double2 a, double2 b, bool& less, bool& par) {
    bool l1 = a.x < b.x, l2 = a.y < b.y;
    bool p1 = a.x == b.x;
    less = l1 || (p1 && l2); par = p1 && (a.y == b.y); 
  }
  __host__ __device__ __forceinline__ void cmp_less_w_parity(float4 a, float4 b, bool& less, bool& par) {
    bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w;
    bool p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z);
    less = l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); par = p3 && (a.w == b.w);
  }

  __host__ __device__ __forceinline__ real_pair<real_t> operator()(real_pair<real_t> e1, real_pair<real_t> e2) {
    bool less, par;
    cmp_less_w_parity(e1.real, e2.real, less, par);
    real_pair<real_t> val = less ? e2 : e1;
    int32_t id_tie = min(e1.idx, e2.idx);
    return real_pair<real_t>({ val.real, par ? id_tie : val.idx });
  }
};

template <class real_t, class real_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void imax_kernel(int32_t N, real_ptr X, int32_t* __restrict__ i_out, real_ptr d_out) {
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t thread_data[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];
  int32_t thread_locs[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load);
  cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS> block_reduce(temp_reduce);
  real_pair_max<real_t> cmp_max;

  block_load.Load(X, thread_data, N1, real_t());
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    int32_t thread_loc_i = int32_t(threadIdx.x) + i * BLOCK_THREADS;
    thread_pair[i] = real_pair<real_t>({ thread_data[i], thread_loc_i });
    thread_locs[i] = thread_loc_i;
  }

  for (int32_t i = elements; i < N1; i += elements) {
    block_load.Load(&X[i], thread_data);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], i + thread_locs[j] }));
  }

  if (0 < N2) {
    block_load.Load(&X[N1], thread_data, N2, real_t());

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], N1 + thread_locs[j] }));
  }

  real_pair<real_t> block_res = block_reduce.Reduce(thread_pair, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0 && block_res.idx < N) {
    *i_out = block_res.idx;
    *d_out = block_res.real;
    X[block_res.idx] = X[0];
  }
  else if (threadIdx.x == 0)
    *i_out = -1;
}

constexpr int32_t block_threads = 512;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, double* X, int32_t* piv, double* diag) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  imax_kernel <double, double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, diag);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, float* X, int32_t* piv, float* diag) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  imax_kernel <float, float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, diag);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, double2* X, int32_t* piv, double2* diag) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  imax_kernel <double2, double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, diag);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, float4* X, int32_t* piv, float4* diag) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  imax_kernel <float4, float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, piv, diag);
}
