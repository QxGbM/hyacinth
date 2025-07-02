
#include <internal.hpp>
#include <float4.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

struct minus_norm {
  __device__ __forceinline__ double operator()(double a, double c) { return fma(-a, a, c); }
  __device__ __forceinline__ float operator()(float a, float c) { return fmaf(-a, a, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 c) { return device::f4::fma(device::f4::negate(a), a, c); }

  __device__ __forceinline__ double operator()(cuDoubleComplex a, double c) {
    return fma(-a.x, a.x, fma(-a.y, a.y, c)); }
  __device__ __forceinline__ float operator()(cuComplex a, float c) {
    return fmaf(-a.x, a.x, fmaf(-a.y, a.y, c)); }
  __device__ __forceinline__ float4 operator()(complex_float4 a, float4 c) { 
    return device::f4::fma(device::f4::negate(a.real), a.real, device::f4::fma(device::f4::negate(a.imag), a.imag, c)); }
};

struct init_load {
  __device__ __forceinline__ operator double() { return 0.; }
  __device__ __forceinline__ operator float() { return 0.f; }
  __device__ __forceinline__ operator float4() { return make_float4(0.f, 0.f, 0.f, 0.f); }

  __device__ __forceinline__ operator cuDoubleComplex() { return make_cuDoubleComplex(0., 0.); }
  __device__ __forceinline__ operator cuComplex() { return make_cuComplex(0.f, 0.f); }
  __device__ __forceinline__ operator complex_float4() { return device::f4::make_complex_float4(make_float4(0.f, 0.f, 0.f, 0.f), make_float4(0.f, 0.f, 0.f, 0.f)); }
};

template <class real_t> struct real_pair {
  real_t first;
  int32_t second;
};

template <class real_t> struct real_pair_max {
  struct less_real {
    __device__ __forceinline__ bool operator()(double a, double b) { return a < b; }
    __device__ __forceinline__ bool operator()(float a, float b) { return a < b; }
    __device__ __forceinline__ bool operator()(float4 a, float4 b) { return device::f4::a_less_than_b(a, b); }
  };

  struct eq_real {
    __device__ __forceinline__ bool operator()(double a, double b) { return a == b; }
    __device__ __forceinline__ bool operator()(float a, float b) { return a == b; }
    __device__ __forceinline__ bool operator()(float4 a, float4 b) { return device::f4::a_eq_to_b(a, b); }
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
  __device__ __forceinline__ float4 operator()(float4 f) { return device::f4::frsqrt(f); }
};

template <class real_t, class real_ptr, class complex_t, class complex_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void update_reduce_general(int32_t N, complex_const_ptr A, real_ptr X, int32_t* i_out, real_ptr rsq_out) {
  using BlockLoadA = cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockLoadX = cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockStore = cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockReduce = cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>;
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename BlockLoadA::TempStorage temp_loadA;
  __shared__ typename BlockLoadX::TempStorage temp_loadX;
  __shared__ typename BlockStore::TempStorage temp_store;
  __shared__ typename BlockReduce::TempStorage temp_reduce;

  complex_t thread_A[ITEMS_PER_THREAD], init_complex = init_load();
  real_t thread_X[ITEMS_PER_THREAD], init_real = init_load();
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];
  real_pair_max<real_t> cmp_max;
  minus_norm fma_func;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_pair[i] = real_pair<real_t>({ init_real, -1 });

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + i;
    BlockLoadA(temp_loadA).Load(&A[i], thread_A, num_items, init_complex);
    BlockLoadX(temp_loadX).Load(&X[i], thread_X, num_items, init_real);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
      real_t ax = fma_func(thread_A[j], thread_X[j]);
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ ax, thread_loc + j }));
      thread_X[j] = ax;
    }

    BlockStore(temp_store).Store(&X[i], thread_X, num_items);
  }

  real_pair<real_t> block_res = BlockReduce(temp_reduce).Reduce(thread_pair, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0) {
    rsqrt_real rsqrt_func;
    *i_out = block_res.second;
    *rsq_out = rsqrt_func(block_res.first);
    X[block_res.second] = X[0];
  }
}

constexpr int32_t block_threads = 8 * 32;

void imax_update_double(cudaStream_t stream, int32_t N, const double* A, double* X, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = 4;
  update_reduce_general <double, double* __restrict__, double, const double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void imax_update_float(cudaStream_t stream, int32_t N, const float* A, float* X, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = 8;
  update_reduce_general <float, float* __restrict__, float, const float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void imax_update_float4(cudaStream_t stream, int32_t N, const float4* A, float4* X, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = 4;
  update_reduce_general <float4, float4* __restrict__, float4, const float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void imax_update_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, double* X, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = 2;
  update_reduce_general <double, double* __restrict__, cuDoubleComplex, const cuDoubleComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, (const cuDoubleComplex*)A, X, piv, rsq);
}

void imax_update_float_complex(cudaStream_t stream, int32_t N, const std::complex<float>* A, float* X, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = 4;
  update_reduce_general <float, float* __restrict__, cuComplex, const cuComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, (const cuComplex*)A, X, piv, rsq);
}

void imax_update_float4_complex(cudaStream_t stream, int32_t N, const complex_float4* A, float4* X, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = 1;
  update_reduce_general <float4, float4* __restrict__, complex_float4, const complex_float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}
