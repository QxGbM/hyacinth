
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

struct swap_real {
  __device__ __forceinline__ void operator()(double& c0, double& cp, double& d0, double& dp) {
    double t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = t2;
  }
  __device__ __forceinline__ void operator()(float& c0, float& cp, float& d0, float& dp) {
    float t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = t2;
  }
  __device__ __forceinline__ void operator()(double2& c0, double2& cp, double2& d0, double2& dp) {
    double2 t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = t2;
  }
  __device__ __forceinline__ void operator()(float4& c0, float4& cp, float4& d0, float4& dp) {
    float4 t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = t2;
  }
};

struct swap_complex {
  __device__ __forceinline__ void operator()(cuDoubleComplex& c0, cuDoubleComplex& cp, cuDoubleComplex& d0, cuDoubleComplex& dp) {
    cuDoubleComplex t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = make_cuDoubleComplex(t2.x, -t2.y);
  }
  __device__ __forceinline__ void operator()(cuComplex& c0, cuComplex& cp, cuComplex& d0, cuComplex& dp) {
    cuComplex t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = make_cuComplex(t2.x, -t2.y);
  }
  __device__ __forceinline__ void operator()(complex_double2& c0, complex_double2& cp, complex_double2& d0, complex_double2& dp) {
    complex_double2 t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = device::dd::make_complex_double2(t2.real, device::dd::negate(t2.imag));
  }
  __device__ __forceinline__ void operator()(complex_float4& c0, complex_float4& cp, complex_float4& d0, complex_float4& dp) {
    complex_float4 t1 = c0, t2 = cp, t3 = dp;
    c0 = t2; cp = t1; d0 = t3; dp = device::qf::make_complex_float4(t2.real, device::qf::negate(t2.imag));
  }
};

template <class real_t, class real_ptr, class complex_t, class complex_ptr, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void imax_kernel(int32_t N, real_ptr X, complex_ptr C, int32_t ldc, int32_t* __restrict__ i_out, real_ptr rsq_out) {
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
    if constexpr(sizeof(real_t) < sizeof(complex_t)) {
      swap_complex swap_f;
      swap_f(C[0], C[block_res.idx], C[block_res.idx * uint64_t(ldc)], C[block_res.idx * uint64_t(ldc + 1)]);
    }
    else {
      swap_real swap_f;
      swap_f(C[0], C[block_res.idx], C[block_res.idx * uint64_t(ldc)], C[block_res.idx * uint64_t(ldc + 1)]);
    }

    rsqrt_real rsqrt_func;
    *i_out = block_res.idx;
    *rsq_out = rsqrt_func(block_res.real);
    X[block_res.idx] = X[0];
  }
}

constexpr int32_t block_threads = 512;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::imax_f64(cudaStream_t stream, int32_t N, double* X, double* C, int32_t ldc, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double);
  imax_kernel <double, double* __restrict__, double, double* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_f32(cudaStream_t stream, int32_t N, float* X, float* C, int32_t ldc, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float);
  imax_kernel <float, float* __restrict__, float, float* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_f128_dd(cudaStream_t stream, int32_t N, double2* X, double2* C, int32_t ldc, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(double2);
  imax_kernel <double2, double2* __restrict__, double2, double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_f128_qf(cudaStream_t stream, int32_t N, float4* X, float4* C, int32_t ldc, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(float4);
  imax_kernel <float4, float4* __restrict__, float4, float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_cf64(cudaStream_t stream, int32_t N, double* X, std::complex<double>* C, int32_t ldc, int32_t* piv, double* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<double>);
  imax_kernel <double, double* __restrict__, cuDoubleComplex, cuDoubleComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, (cuDoubleComplex*)C, ldc, piv, rsq);
}

void internal::Cholesky::imax_cf32(cudaStream_t stream, int32_t N, float* X, std::complex<float>* C, int32_t ldc, int32_t* piv, float* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(std::complex<float>);
  imax_kernel <float, float* __restrict__, cuComplex, cuComplex* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, (cuComplex*)C, ldc, piv, rsq);
}

void internal::Cholesky::imax_cf128_dd(cudaStream_t stream, int32_t N, double2* X, complex_double2* C, int32_t ldc, int32_t* piv, double2* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_double2);
  imax_kernel <double2, double2* __restrict__, complex_double2, complex_double2* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}

void internal::Cholesky::imax_cf128_qf(cudaStream_t stream, int32_t N, float4* X, complex_float4* C, int32_t ldc, int32_t* piv, float4* rsq) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(complex_float4);
  imax_kernel <float4, float4* __restrict__, complex_float4, complex_float4* __restrict__, block_threads, items_per_thread>
    <<< 1, block_threads, 0, stream >>> (N, X, C, ldc, piv, rsq);
}
