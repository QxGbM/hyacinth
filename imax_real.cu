
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

struct minus_norm {
  __device__ __forceinline__ double operator()(double a, double c) { return fma(-a, a, c); }
  __device__ __forceinline__ float operator()(float a, float c) { return fmaf(-a, a, c); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 c) { return device::dd::fma(device::dd::negate(a), a, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 c) { return device::qf::fma(device::qf::negate(a), a, c); }
};

template <class real_t, class real_ptr, class real_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void update_reduce_real(int32_t N, real_const_ptr A, real_ptr X, real_ptr C, int32_t ldc, int32_t* i_out, real_ptr rsq_out) {
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  __shared__ typename cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t thread_A[ITEMS_PER_THREAD], thread_X[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS> block_reduce(temp_reduce);

  real_pair_max<real_t> cmp_max;
  minus_norm fma_func;

  int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD;
  block_load.Load(A, thread_A, N1, real_t());
  block_load.Load(X, thread_X, N1, real_t());

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    real_t ax = fma_func(thread_A[i], thread_X[i]);
    thread_pair[i] = real_pair<real_t>({ ax, thread_loc + i });
    thread_X[i] = ax;
  }

  block_store.Store(X, thread_X, N1);

  for (int32_t i = elements; i < N1; i += elements) {
    int32_t thread_loc_i = thread_loc + i;
    block_load.Load(&A[i], thread_A);
    block_load.Load(&X[i], thread_X);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
      real_t ax = fma_func(thread_A[j], thread_X[j]);
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ ax, thread_loc_i + j }));
      thread_X[j] = ax;
    }

    block_store.Store(&X[i], thread_X);
  }

  if (0 < N2) {
    int32_t thread_loc_n = thread_loc + N1;
    block_load.Load(&A[N1], thread_A, N2, real_t());
    block_load.Load(&X[N1], thread_X, N2, real_t());

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      real_t ax = fma_func(thread_A[i], thread_X[i]);
      thread_pair[i] = cmp_max(thread_pair[i], real_pair<real_t>({ ax, thread_loc_n + i }));
      thread_X[i] = ax;
    }

    block_store.Store(&X[N1], thread_X, N2);
  }

  real_pair<real_t> block_res = block_reduce.Reduce(thread_pair, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0) {
    real_t Cp = C[block_res.idx], Dp = C[block_res.idx * (ldc + 1)], D0 = C[0];
    C[0] = Cp;
    C[block_res.idx] = D0;
    C[block_res.idx * ldc] = Dp;
    C[block_res.idx * (ldc + 1)] = Cp;

    rsqrt_real rsqrt_func;
    *i_out = block_res.idx;
    *rsq_out = rsqrt_func(block_res.real);
    X[block_res.idx] = X[0];
  }
}

template <class real_t, class real_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void reduce_real(int32_t N, real_ptr X, real_ptr C, int32_t ldc, int32_t* i_out, real_ptr rsq_out) {
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;
  int32_t rem = N & (elements - 1), div = N - rem;
  int32_t N1 = max(div, rem), N2 = min(div, rem);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>::TempStorage temp_reduce;
  real_t thread_data[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];

  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS> block_reduce(temp_reduce);
  real_pair_max<real_t> cmp_max;

  int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD;
  block_load.Load(X, thread_data, N1, real_t());

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_pair[i] = real_pair<real_t>({ thread_data[i], thread_loc + i });

  for (int32_t i = elements; i < N1; i += elements) {
    int32_t thread_loc_i = thread_loc + i;
    block_load.Load(&X[i], thread_data);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], thread_loc_i + j }));
  }

  if (0 < N2) {
    int32_t thread_loc_n = thread_loc + N1;
    block_load.Load(&X[N1], thread_data, N2, real_t());

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], thread_loc_n + j }));
  } 

  real_pair<real_t> block_res = block_reduce.Reduce(thread_pair, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0) {
    real_t Cp = C[block_res.idx], Dp = C[block_res.idx * (ldc + 1)], D0 = C[0];
    C[0] = Cp;
    C[block_res.idx] = D0;
    C[block_res.idx * ldc] = Dp;
    C[block_res.idx * (ldc + 1)] = Cp;

    rsqrt_real rsqrt_func;
    *i_out = block_res.idx;
    *rsq_out = rsqrt_func(block_res.real);
    X[block_res.idx] = X[0];
  }
}

constexpr int32_t block_threads = 16 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_double(cudaStream_t stream, int32_t N, const double* A, double* X, double* C, int32_t ldc, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);

  if (A == nullptr)
    reduce_real <double, double* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
  else
    update_reduce_real <double, double* __restrict__, const double* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, A, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_float(cudaStream_t stream, int32_t N, const float* A, float* X, float* C, int32_t ldc, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);

  if (A == nullptr)
    reduce_real <float, float* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
  else
    update_reduce_real <float, float* __restrict__, const float* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, A, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_double2(cudaStream_t stream, int32_t N, const double2* A, double2* X, double2* C, int32_t ldc, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);

  if (A == nullptr)
    reduce_real <double2, double2* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
  else
    update_reduce_real <double2, double2* __restrict__, const double2* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, A, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_float4(cudaStream_t stream, int32_t N, const float4* A, float4* X, float4* C, int32_t ldc, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);

  if (A == nullptr)
    reduce_real <float4, float4* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
  else
    update_reduce_real <float4, float4* __restrict__, const float4* __restrict__, block_threads, items_per_thread>
      <<< 1, block_threads, 0, stream >>> (N, A, X, C, ldc, piv, rsq);
}
