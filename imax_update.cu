
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>

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

  __device__ __forceinline__ double operator()(cuDoubleComplex a, double c) {
    return fma(-a.x, a.x, fma(-a.y, a.y, c)); }
  __device__ __forceinline__ float operator()(cuComplex a, float c) {
    return fmaf(-a.x, a.x, fmaf(-a.y, a.y, c)); }
  __device__ __forceinline__ double2 operator()(complex_double2 a, double2 c) { 
    return device::dd::fma(device::dd::negate(a.real), a.real, device::dd::fma(device::dd::negate(a.imag), a.imag, c)); }
  __device__ __forceinline__ float4 operator()(complex_float4 a, float4 c) { 
    return device::qf::fma(device::qf::negate(a.real), a.real, device::qf::fma(device::qf::negate(a.imag), a.imag, c)); }
};

template <class real_t, class real_ptr, class complex_t, class complex_const_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void update_reduce_general(int32_t N, complex_const_ptr A, real_ptr X, int32_t* i_out, real_ptr rsq_out) {
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_loadA;
  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_loadX;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  __shared__ typename cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS>::TempStorage temp_reduce;
  complex_t thread_A[ITEMS_PER_THREAD];
  real_t thread_X[ITEMS_PER_THREAD];
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];

  cub::BlockLoad<complex_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load_a(temp_loadA);
  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load_x(temp_loadX);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  cub::BlockReduce<real_pair<real_t>, BLOCK_THREADS> block_reduce(temp_reduce);

  real_pair_max<real_t> cmp_max;
  minus_norm fma_func;

  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    thread_pair[i] = real_pair<real_t>({ real_t(), -1 });

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    int32_t thread_loc = threadIdx.x * ITEMS_PER_THREAD + i;
    block_load_a.Load(&A[i], thread_A, num_items, complex_t());
    block_load_x.Load(&X[i], thread_X, num_items, real_t());

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j) {
      real_t ax = fma_func(thread_A[j], thread_X[j]);
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ ax, thread_loc + j }));
      thread_X[j] = ax;
    }

    block_store.Store(&X[i], thread_X, num_items);
  }

  real_pair<real_t> block_res = block_reduce.Reduce(thread_pair, cmp_max);

  __syncthreads();
  if (threadIdx.x == 0) {
    rsqrt_real rsqrt_func;
    *i_out = block_res.idx;
    *rsq_out = rsqrt_func(block_res.real);
    X[block_res.idx] = X[0];
  }
}

constexpr int32_t block_threads = 8 * 32;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_update_double(cudaStream_t stream, int32_t N, const double* A, double* X, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  update_reduce_general <double, double* __restrict__, double, const double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void internal::Cholesky::imax_update_float(cudaStream_t stream, int32_t N, const float* A, float* X, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  update_reduce_general <float, float* __restrict__, float, const float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void internal::Cholesky::imax_update_double2(cudaStream_t stream, int32_t N, const double2* A, double2* X, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  update_reduce_general <double2, double2* __restrict__, double2, const double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void internal::Cholesky::imax_update_float4(cudaStream_t stream, int32_t N, const float4* A, float4* X, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  update_reduce_general <float4, float4* __restrict__, float4, const float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void internal::Cholesky::imax_update_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, double* X, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  update_reduce_general <double, double* __restrict__, cuDoubleComplex, const cuDoubleComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, (const cuDoubleComplex*)A, X, piv, rsq);
}

void internal::Cholesky::imax_update_float_complex(cudaStream_t stream, int32_t N, const std::complex<float>* A, float* X, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  update_reduce_general <float, float* __restrict__, cuComplex, const cuComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, (const cuComplex*)A, X, piv, rsq);
}

void internal::Cholesky::imax_update_double2_complex(cudaStream_t stream, int32_t N, const complex_double2* A, double2* X, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  update_reduce_general <double2, double2* __restrict__, complex_double2, const complex_double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}

void internal::Cholesky::imax_update_float4_complex(cudaStream_t stream, int32_t N, const complex_float4* A, float4* X, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  update_reduce_general <float4, float4* __restrict__, complex_float4, const complex_float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, A, X, piv, rsq);
}
