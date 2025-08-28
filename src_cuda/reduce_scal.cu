
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
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::add(a, b); }
};

struct scal_a_function {
  __device__ __forceinline__ void operator()(double s, double a, double& c, double& c_conj, double& d) {
    double e = c_conj = c = s * a; d = fma(-e, e, d);
  }
  __device__ __forceinline__ void operator()(float s, float a, float& c, float& c_conj, float& d) {
    float e = c_conj = c = s * a; d = fmaf(-e, e, d);
  }
  __device__ __forceinline__ void operator()(double2 s, double2 a, double2& c, double2& c_conj, double2& d) {
    double2 e = c_conj = c = device::dd::mul(s, a); d = device::dd::add(device::dd::mul(device::dd::negate(e), e), d);
  }
  __device__ __forceinline__ void operator()(float4 s, float4 a, float4& c, float4& c_conj, float4& d) {
    float4 e = c_conj = c = device::qf::mul(s, a); d = device::qf::add(device::qf::mul(device::qf::negate(e), e), d);
  }

  __device__ __forceinline__ void operator()(double s, cuDoubleComplex a, cuDoubleComplex& c, cuDoubleComplex& c_conj, double& d) {
    cuDoubleComplex e = c = make_cuDoubleComplex(s * a.x, s * a.y);
    c_conj = make_cuDoubleComplex(e.x, -e.y);
    d = fma(-e.x, e.x, fma(-e.y, e.y, d));
  }
  __device__ __forceinline__ void operator()(float s, cuComplex a, cuComplex& c, cuComplex& c_conj, float& d) {
    cuComplex e = c = make_cuComplex(s * a.x, s * a.y);
    c_conj = make_cuComplex(e.x, -e.y);
    d = fmaf(-e.x, e.x, fmaf(-e.y, e.y, d));
  }
  __device__ __forceinline__ void operator()(double2 s, complex_double2 a, complex_double2& c, complex_double2& c_conj, double2& d) {
    using device::dd::add, device::dd::mul, device::dd::negate;
    complex_double2 e = c = device::dd::make_complex_double2(mul(s, a.real), mul(s, a.imag));
    c_conj = device::dd::make_complex_double2(e.real, negate(e.imag));
    d = add(mul(negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }
  __device__ __forceinline__ void operator()(float4 s, complex_float4 a, complex_float4& c, complex_float4& c_conj, float4& d) {
    using device::qf::add, device::qf::mul, device::qf::negate;
    complex_float4 e = c = device::qf::make_complex_float4(mul(s, a.real), mul(s, a.imag));
    c_conj = device::qf::make_complex_float4(e.real, negate(e.imag));
    d = add(mul(device::qf::negate(e.real), e.real), add(mul(negate(e.imag), e.imag), d));
  }
};

template <int32_t COMPLEX, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_sum(matrix_t (&a)[ITEMS_PER_THREAD], matrix_t (&b)[ITEMS_PER_THREAD]) {
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

template <class real_t, class real_ptr, class matrix_t, class matrix_ptr, int32_t WARP_THREADS, int32_t ITEMS_PER_THREAD, int32_t N>
__global__ void fix_N_reduce_kernel(real_t sq, real_t rsq, int32_t M, matrix_ptr A, int32_t lda, real_ptr D) {
  constexpr int32_t COMPLEX = (sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t block_warps = (N + 7) / 8;
  constexpr int32_t elements_thread = N < 8 ? N : 8;
  constexpr int32_t elements_warp = ITEMS_PER_THREAD * WARP_THREADS;
  int32_t small_load_pred = int32_t(blockIdx.x + 1 == gridDim.x && 0 < M);
  M = small_load_pred ? M : elements_warp;
  matrix_ptr B = &A[int64_t((1 - N) + int32_t(threadIdx.y << 3)) * int64_t(lda)];

  __shared__ typename cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::BlockStore<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store[block_warps];
  __shared__ typename cub::BlockLoad<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load_rl[block_warps];
  __shared__ typename cub::BlockStore<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store_rl[block_warps];

  __shared__ matrix_t warpA[block_warps][elements_warp];
  matrix_t threadA[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];
  real_t threadD[ITEMS_PER_THREAD];

  cub::BlockLoad<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load[threadIdx.y]);
  cub::BlockStore<matrix_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store[threadIdx.y]);
  cub::BlockLoad<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_rl(temp_load_rl[threadIdx.y]);
  cub::BlockStore<real_t, WARP_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store_rl(temp_store_rl[threadIdx.y]);

  int32_t i = 1 + blockIdx.x * elements_warp;
  matrix_ptr A_i = &B[i];
  if (small_load_pred)
    block_load.Load(A_i, threadA, M);
  else
    block_load.Load(A_i, threadA);

  #pragma unroll
  for (int32_t k = 1; k < elements_thread; ++k) {
    block_load.Load(&A_i[uint64_t(k) * uint64_t(lda)], threadB);
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
    scal_a_function scal_func;
    if (small_load_pred)
      block_load_rl.Load(&D[i], threadD, M);
    else
      block_load_rl.Load(&D[i], threadD);

    #pragma unroll
    for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k)
      scal_func(rsq, threadA[k], threadA[k], threadB[k], threadD[k]);

    if (small_load_pred) {
      block_store_rl.Store(&D[i], threadD, M);

      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k) {
        int32_t thread_k = (k * WARP_THREADS) + threadIdx.x;
        if (thread_k < M)
          A[(uint64_t(i) + uint64_t(thread_k)) * uint64_t(lda)] = threadB[k];
      }
    }
    else {
      block_store_rl.Store(&D[i], threadD);

      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k) {
        int32_t thread_k = (k * WARP_THREADS) + threadIdx.x;
        A[(uint64_t(i) + uint64_t(thread_k)) * uint64_t(lda)] = threadB[k];
      }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) {
      ((real_t*)A)[0] = sq;
      if constexpr(COMPLEX)
        ((real_t*)A)[1] = real_t();
    }
  }
}

constexpr int32_t warp_threads = 64;
constexpr int32_t thread_bytes = 32;

template <class real_t, class real_ptr, class matrix_t, class matrix_ptr>
inline void reduce_scal_dispatcher(cudaStream_t stream, real_t* scale, int32_t M, int32_t N, matrix_ptr A, int32_t lda, real_ptr D) {
  if (1 < M) {
    constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);
    constexpr int32_t elements_warp = items_per_thread * warp_threads;
    int32_t grid = ((M - 1) + elements_warp - 1) / elements_warp;
    int32_t rem = (M - 1) & (elements_warp - 1);
    real_t sq = scale[0], rsq = scale[1];

    switch (N) {
      case 1: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 1>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 2: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 2>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 3: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 3>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 4: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 4>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 8: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 8>
        <<< grid, dim3(warp_threads, 1, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 16: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 16>
        <<< grid, dim3(warp_threads, 2, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 32: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 32>
        <<< grid, dim3(warp_threads, 4, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      case 64: fix_N_reduce_kernel <real_t, real_ptr, matrix_t, matrix_ptr, warp_threads, items_per_thread, 64>
        <<< grid, dim3(warp_threads, 8, 1), 0, stream >>> (sq, rsq, rem, A, lda, D); break;
      default: break;
    }
  }
  else if (1 == M) {
    std::memset(&scale[1], 0, sizeof(real_t));
    cudaMemcpyAsync(A, scale, sizeof(matrix_t), cudaMemcpyHostToDevice, stream);
  }
}

void internal::Cholesky::reduce_scal_f64(cudaStream_t stream, double* scale, int32_t M, int32_t N, double* A, int32_t lda, double* D) {
  reduce_scal_dispatcher<double, double* __restrict__, double, double* __restrict__>(stream, scale, M, N, A, lda, D);
}

void internal::Cholesky::reduce_scal_f32(cudaStream_t stream, float* scale, int32_t M, int32_t N, float* A, int32_t lda, float* D) {
  reduce_scal_dispatcher<float, float* __restrict__, float, float* __restrict__>(stream, scale, M, N, A, lda, D);
}

void internal::Cholesky::reduce_scal_f128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, double2* A, int32_t lda, double2* D) {
  reduce_scal_dispatcher<double2, double2* __restrict__, double2, double2* __restrict__>(stream, scale, M, N, A, lda, D);
}

void internal::Cholesky::reduce_scal_f128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, float4* A, int32_t lda, float4* D) {
  reduce_scal_dispatcher<float4, float4* __restrict__, float4, float4* __restrict__>(stream, scale, M, N, A, lda, D);
}

void internal::Cholesky::reduce_scal_cf64(cudaStream_t stream, double* scale, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, double* D) {
  reduce_scal_dispatcher<double, double* __restrict__, cuDoubleComplex, cuDoubleComplex* __restrict__>(stream, scale, M, N, (cuDoubleComplex*)A, lda, D);
}

void internal::Cholesky::reduce_scal_cf32(cudaStream_t stream, float* scale, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, float* D) {
  reduce_scal_dispatcher<float, float* __restrict__, cuComplex, cuComplex* __restrict__>(stream, scale, M, N, (cuComplex*)A, lda, D);
}

void internal::Cholesky::reduce_scal_cf128_dd(cudaStream_t stream, double2* scale, int32_t M, int32_t N, complex_double2* A, int32_t lda, double2* D) {
  reduce_scal_dispatcher<double2, double2* __restrict__, complex_double2, complex_double2* __restrict__>(stream, scale, M, N, A, lda, D);
}

void internal::Cholesky::reduce_scal_cf128_qf(cudaStream_t stream, float4* scale, int32_t M, int32_t N, complex_float4* A, int32_t lda, float4* D) {
  reduce_scal_dispatcher<float4, float4* __restrict__, complex_float4, complex_float4* __restrict__>(stream, scale, M, N, A, lda, D);
}
