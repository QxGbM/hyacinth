
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a + b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a + b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::add(a, b); }
};

struct add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) {
    return device::dd::make_complex_double2(device::dd::add(a.real, b.real), device::dd::add(a.imag, b.imag)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) {
    return device::qf::make_complex_float4(device::qf::add(a.real, b.real), device::qf::add(a.imag, b.imag)); }
};

template <int32_t COMPLEX, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ __forceinline__ void array_sum(matrix_t (&a)[ITEMS_PER_THREAD], matrix_t (&b)[ITEMS_PER_THREAD]) {
  if constexpr(COMPLEX) {
    add_complex add_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      a[i] = add_func(a[i], b[i]);
  }
  else {
    add_real add_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      a[i] = add_func(a[i], b[i]);
  }
}

template <class real_t, class matrix_t> struct fused_scal_a {
  real_t rsq; real_t* __restrict__ D; matrix_t x0; matrix_t* __restrict__ X; int64_t incx;

  fused_scal_a(real_t sq, real_t rsq, matrix_t* X, int64_t incx, real_t* D) :
    rsq(rsq), D(D), X(X), incx(incx) {
      union { real_t f[2]; matrix_t e;} d { sq, real_t() };
      x0 = d.e;
    }

  __device__ __forceinline__ void scal_a(double s, double a, double& c_conj, double& d) const {
    double e = c_conj = s * a; d = fma(-e, e, d);
  }
  __device__ __forceinline__ void scal_a(float s, float a, float& c_conj, float& d) const {
    float e = c_conj = s * a; d = fmaf(-e, e, d);
  }
  __device__ __forceinline__ void scal_a(double2 s, double2 a, double2& c_conj, double2& d) const {
    double2 e = c_conj = device::dd::mul(s, a);
    d = device::dd::add(device::dd::mul(device::dd::negate(e), e), d);
  }
  __device__ __forceinline__ void scal_a(float4 s, float4 a, float4& c_conj, float4& d) const {
    float4 e = c_conj = device::qf::mul(s, a);
    d = device::qf::add(device::qf::mul(device::qf::negate(e), e), d);
  }
  __device__ __forceinline__ void scal_a(double s, cuDoubleComplex a, cuDoubleComplex& c_conj, double& d) const {
    cuDoubleComplex e = c_conj = make_cuDoubleComplex(s * a.x, -s * a.y);
    d = fma(-e.x, e.x, fma(-e.y, e.y, d));
  }
  __device__ __forceinline__ void scal_a(float s, cuComplex a, cuComplex& c_conj, float& d) const {
    cuComplex e = c_conj = make_cuComplex(s * a.x, -s * a.y);
    d = fmaf(-e.x, e.x, fmaf(-e.y, e.y, d));
  }
  __device__ __forceinline__ void scal_a(double2 s, complex_double2 a, complex_double2& c_conj, double2& d) const {
    using device::dd::add, device::dd::mul, device::dd::negate;
    complex_double2 e = c_conj = device::dd::make_complex_double2(mul(s, a.real), mul(negate(s), a.imag));
    d = add(mul(negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }
  __device__ __forceinline__ void scal_a(float4 s, complex_float4 a, complex_float4& c_conj, float4& d) const {
    using device::qf::add, device::qf::mul, device::qf::negate;
    complex_float4 e = c_conj = device::qf::make_complex_float4(mul(s, a.real), mul(negate(s), a.imag));
    d = add(mul(negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }

  __device__ __forceinline__ void set_offdiag(matrix_t a, int64_t i) const { scal_a(rsq, a, X[i * incx], D[i]); }
  __device__ __forceinline__ void set_diag() const { X[0] = x0; }
};

template <class matrix_const_ptr, int32_t WARP_THREADS, int32_t ITEMS_PER_THREAD, int32_t N, class real_t, class matrix_t>
__global__ void fix_N_reduce_kernel(int32_t M, matrix_const_ptr A, int64_t lda, fused_scal_a<real_t, matrix_t> scal_func) {
  constexpr int32_t COMPLEX = (sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t block_warps = (N + 7) / 8;
  constexpr int32_t elements_thread = N < 8 ? N : 8;
  constexpr int32_t elements_warp = ITEMS_PER_THREAD * WARP_THREADS;
  int32_t small_load_pred = int32_t(blockIdx.x + 1 == gridDim.x && 0 < M);
  M = small_load_pred ? M : elements_warp;

  __shared__ typename cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::BlockStore<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store[block_warps];
  __shared__ typename cub::BlockLoad<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load_rl[block_warps];

  __shared__ matrix_t warpA[block_warps][elements_warp];
  matrix_t threadA[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];
  real_t threadD[ITEMS_PER_THREAD]; matrix_t threadX[ITEMS_PER_THREAD];

  cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load[threadIdx.y]);
  cub::BlockStore<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store[threadIdx.y]);
  cub::BlockLoad<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load_rl[threadIdx.y]);

  int32_t i = blockIdx.x * elements_warp;
  matrix_const_ptr A_i = &A[uint64_t(i) + uint64_t(threadIdx.y << 3) * lda];
  if (small_load_pred)
    block_load.Load(A_i, threadA, M);
  else
    block_load.Load(A_i, threadA);

  #pragma unroll
  for (int32_t k = 1; k < elements_thread; ++k) {
    block_load.Load(&A_i[uint64_t(k) * lda], threadB);
    array_sum<COMPLEX>(threadA, threadB);
  }

  if constexpr(8 < N) {
    block_store.Store(warpA[threadIdx.y], threadA);
    __syncthreads();

    if (threadIdx.y == 0) {
      #pragma unroll
      for (int32_t k = 1; k < block_warps; ++k) {
        block_load.Load(warpA[k], threadB);
        array_sum<COMPLEX>(threadA, threadB);
      }
    }
  }

  if (threadIdx.y == 0) {
    if (small_load_pred) {
      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k) {
        int32_t thread_k = (k * WARP_THREADS) + (1 + threadIdx.x);
        if (thread_k <= M)
          scal_func.set_offdiag(threadA[k], int64_t(i) + int64_t(thread_k));
      }
    }
    else {
      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k)
        scal_func.set_offdiag(threadA[k], int64_t(i) + int64_t((k * WARP_THREADS) + (1 + threadIdx.x)));
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
      scal_func.set_diag();
  }
}

constexpr int32_t warp_threads = 64;
constexpr int32_t thread_bytes = 32;

template <class matrix_const_ptr, class real_t, class matrix_t>
inline void reduce_scal_dispatcher(cudaStream_t stream, real_t* scale, int32_t M, int32_t N, matrix_const_ptr A, int32_t lda, matrix_t* X, int32_t incx, real_t* D) {
  fused_scal_a<real_t, matrix_t> scal_func(scale[0], scale[1], X, incx, D);

  if (1 < M) {
    constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);
    constexpr int32_t elements_warp = items_per_thread * warp_threads;
    int32_t grid = ((M - 1) + elements_warp - 1) / elements_warp;
    int32_t rem = (M - 1) & (elements_warp - 1);

    switch (N) {
      case 1: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 1>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 2: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 2>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 4: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 4>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 8: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 8>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 16: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 16>
        <<< grid, dim3(warp_threads, 2, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 32: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 32>
        <<< grid, dim3(warp_threads, 4, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      case 64: fix_N_reduce_kernel <matrix_const_ptr, warp_threads, items_per_thread, 64>
        <<< grid, dim3(warp_threads, 8, 1), 0, stream >>> (rem, A, lda, scal_func); break;
      default: break;
    }
  }
  else if (1 == M) {
    *((matrix_t*)scale) = scal_func.x0;
    cudaMemcpyAsync(X, scale, sizeof(matrix_t), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::reduce_scal_f64(cudaStream_t stream, double* scale, int32_t M, int32_t N, const double* A, int32_t lda, double* X, int32_t incx, double* D) {
  reduce_scal_dispatcher<const double* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

void internal::Cholesky::reduce_scal_f32(cudaStream_t stream, float* scale, int32_t M, int32_t N, const float* A, int32_t lda, float* X, int32_t incx, float* D) {
  reduce_scal_dispatcher<const float* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

void internal::Cholesky::reduce_scal_cf64(cudaStream_t stream, double* scale, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* X, int32_t incx, double* D) {
  reduce_scal_dispatcher<const cuDoubleComplex* __restrict__>(stream, scale, M, N, (const cuDoubleComplex*)A, lda, (cuDoubleComplex*)X, incx, D);
}

void internal::Cholesky::reduce_scal_cf32(cudaStream_t stream, float* scale, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* X, int32_t incx, float* D) {
  reduce_scal_dispatcher<const cuComplex* __restrict__>(stream, scale, M, N, (const cuComplex*)A, lda, (cuComplex*)X, incx, D);
}

void internal::Cholesky::reduce_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, const double2* A, int32_t lda, double2* X, int32_t incx, double2* D) {
  reduce_scal_dispatcher<const double2* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

void internal::Cholesky::reduce_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, const float4* A, int32_t lda, float4* X, int32_t incx, float4* D) {
  reduce_scal_dispatcher<const float4* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

void internal::Cholesky::reduce_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, const complex_double2* A, int32_t lda, complex_double2* X, int32_t incx, double2* D) {
  reduce_scal_dispatcher<const complex_double2* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

void internal::Cholesky::reduce_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, const complex_float4* A, int32_t lda, complex_float4* X, int32_t incx, float4* D) {
  reduce_scal_dispatcher<const complex_float4* __restrict__>(stream, scale, M, N, A, lda, X, incx, D);
}

