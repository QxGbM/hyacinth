
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

struct mul_real {
  __device__ __forceinline__ double operator()(double a, double b) { return a * b; }
  __device__ __forceinline__ float operator()(float a, float b) { return a * b; }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b) { return device::dd::mul(a, b); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b) { return device::qf::mul(a, b); }
};

struct mul_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, double b) {
    return make_cuDoubleComplex(a.x * b, a.y * b); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, float b) {
    return make_cuComplex(a.x * b, a.y * b); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, double2 b) { 
    return device::dd::make_complex_double2(device::dd::mul(a.real, b), device::dd::mul(a.imag, b)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, float4 b) { 
    return device::qf::make_complex_float4(device::qf::mul(a.real, b), device::qf::mul(a.imag, b)); }
};

struct conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::conj(f); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::conj(f); }
};

template <int32_t COMPLEX, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_sum(matrix_t (&a)[ITEMS_PER_THREAD], matrix_t const (&b)[ITEMS_PER_THREAD]) {
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

template <int32_t COMPLEX, class real_t, class matrix_t, int32_t ITEMS_PER_THREAD>
__device__ void array_mul(real_t s, matrix_t (&a)[ITEMS_PER_THREAD], matrix_t (&b)[ITEMS_PER_THREAD]) {
  if constexpr(COMPLEX) {
    mul_complex mul_func;
    conj conj_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
      matrix_t e = a[i] = mul_func(a[i], s);
      b[i] = conj_func(e);
    }
  }
  else {
    mul_real mul_func;
    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      b[i] = a[i] = mul_func(a[i], s);
  }
}

template <class real_t, class matrix_t, class matrix_ptr, int32_t ITEMS_PER_THREAD, int32_t N>
__global__ void fix_N_reduce_kernel(real_t scale, int32_t M, matrix_ptr A, int32_t lda) {
  constexpr int32_t COMPLEX = (sizeof(real_t) < sizeof(matrix_t));
  constexpr int32_t block_warps = (N + 7) / 8;
  constexpr int32_t elements_thread = N < 8 ? N : 8;
  constexpr int32_t elements_block = ITEMS_PER_THREAD * 32;
  int32_t elements = gridDim.x * elements_block;
  int32_t M2 = M & (elements_block - 1), M1 = M - M2;
  matrix_ptr B = &A[int64_t((1 - N) + int32_t(threadIdx.y << 3)) * int64_t(lda)];

  __shared__ typename cub::WarpLoad<matrix_t, ITEMS_PER_THREAD, cub::WARP_LOAD_STRIPED>::TempStorage temp_load[block_warps];
  __shared__ typename cub::WarpStore<matrix_t, ITEMS_PER_THREAD, cub::WARP_STORE_STRIPED>::TempStorage temp_store[block_warps];
  __shared__ matrix_t warpA[block_warps][ITEMS_PER_THREAD * 32];
  matrix_t threadA[ITEMS_PER_THREAD], threadB[ITEMS_PER_THREAD];

  cub::WarpLoad<matrix_t, ITEMS_PER_THREAD, cub::WARP_LOAD_STRIPED> warp_load(temp_load[threadIdx.y]);
  cub::WarpStore<matrix_t, ITEMS_PER_THREAD, cub::WARP_STORE_STRIPED> warp_store(temp_store[threadIdx.y]);

  for (int32_t i = (blockIdx.x * elements_block); i < M1; i += elements) {
    matrix_ptr A_i = &B[i];
    warp_load.Load(A_i, threadA);

    #pragma unroll
    for (int32_t k = 1; k < elements_thread; ++k) {
      warp_load.Load(&A_i[uint64_t(k) * uint64_t(lda)], threadB);
      array_sum<COMPLEX>(threadA, threadB);
    }

    if constexpr(8 < N) {
      warp_store.Store(warpA[threadIdx.y], threadA);
      __syncthreads();

      #pragma unroll
      for (int32_t k = 1; k < block_warps; ++k) {
        warp_load.Load(warpA[threadIdx.y], threadB);
        array_sum<COMPLEX>(threadA, threadB);
      }
    }

    if (threadIdx.y == 0) {
      array_mul<COMPLEX>(scale, threadA, threadB);
      warp_store.Store(&A[i], threadA);
      
      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k) {
        int32_t thread_k = (k << 5) + threadIdx.x;
        A[uint64_t(i + thread_k) * uint64_t(lda)] = threadB[k];
      }
    }
  }

  if (0 < M2 && blockIdx.x == 0) {
    matrix_ptr A_i = &B[M1];
    warp_load.Load(A_i, threadA, M2);

    #pragma unroll
    for (int32_t k = 1; k < elements_thread; ++k) {
      warp_load.Load(&A_i[uint64_t(k) * uint64_t(lda)], threadB, M2);
      array_sum<COMPLEX>(threadA, threadB);
    }

    if constexpr(8 < N) {
      warp_store.Store(warpA[threadIdx.y], threadA);
      __syncthreads();

      #pragma unroll
      for (int32_t k = 1; k < block_warps; ++k) {
        warp_load.Load(warpA[threadIdx.y], threadB);
        array_sum<COMPLEX>(threadA, threadB);
      }
    }

    if (threadIdx.y == 0) {
      array_mul<COMPLEX>(scale, threadA, threadB);
      warp_store.Store(&A[M1], threadA, M2);

      #pragma unroll
      for (int32_t k = 0; k < ITEMS_PER_THREAD; ++k) {
        int32_t thread_k = (k << 5) + threadIdx.x;
        if (thread_k < M2)
          A[uint64_t(M1 + thread_k) * uint64_t(lda)] = threadB[k];
      }
    }
  }
}

constexpr int32_t thread_bytes = 32;

template <class real_t, class matrix_t, class matrix_ptr>
inline void reduce_scal_dispatcher(cudaStream_t stream, real_t scale, int32_t M, int32_t N, matrix_ptr A, int32_t lda) {
  constexpr int32_t items_per_thread = thread_bytes / sizeof(matrix_t);
  constexpr int32_t elements_block = items_per_thread * 32;
  int32_t grid = (M + elements_block - 1) / elements_block;

  switch (N) {
    case 1: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 1>
      <<< grid, 32, 0, stream >>> (scale, M, A, lda); break;
    case 4: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 4>
      <<< grid, 32, 0, stream >>> (scale, M, A, lda); break;
    case 8: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 8>
      <<< grid, 32, 0, stream >>> (scale, M, A, lda); break;
    case 16: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 16>
      <<< grid, 64, 0, stream >>> (scale, M, A, lda); break;
    case 32: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 32>
      <<< grid, 128, 0, stream >>> (scale, M, A, lda); break;
    case 64: fix_N_reduce_kernel <real_t, matrix_t, matrix_ptr, items_per_thread, 64>
      <<< grid, 256, 0, stream >>> (scale, M, A, lda); break;
    default: break;
  }
}

void internal::Cholesky::reduce_scal_f64(cudaStream_t stream, const double scale, int32_t M, int32_t N, double* A, int32_t lda) {
  reduce_scal_dispatcher<double, double, double* __restrict__>(stream, scale, M, N, A, lda);
}

void internal::Cholesky::reduce_scal_f32(cudaStream_t stream, const float scale, int32_t M, int32_t N, float* A, int32_t lda) {
  reduce_scal_dispatcher<float, float, float* __restrict__>(stream, scale, M, N, A, lda);
}

void internal::Cholesky::reduce_scal_f128_dd(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, double2* A, int32_t lda) {
  reduce_scal_dispatcher<double2, double2, double2* __restrict__>(stream, scale, M, N, A, lda);
}

void internal::Cholesky::reduce_scal_f128_qf(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, float4* A, int32_t lda) {
  reduce_scal_dispatcher<float4, float4, float4* __restrict__>(stream, scale, M, N, A, lda);
}

void internal::Cholesky::reduce_scal_cf64(cudaStream_t stream, const double scale, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  reduce_scal_dispatcher<double, cuDoubleComplex, cuDoubleComplex* __restrict__>(stream, scale, M, N, (cuDoubleComplex*)A, lda);
}

void internal::Cholesky::reduce_scal_cf32(cudaStream_t stream, const float scale, int32_t M, int32_t N, std::complex<float>* A, int32_t lda) {
  reduce_scal_dispatcher<float, cuComplex, cuComplex* __restrict__>(stream, scale, M, N, (cuComplex*)A, lda);
}

void internal::Cholesky::reduce_scal_cf128_dd(cudaStream_t stream, const double2 scale, int32_t M, int32_t N, complex_double2* A, int32_t lda) {
  reduce_scal_dispatcher<double2, complex_double2, complex_double2* __restrict__>(stream, scale, M, N, A, lda);
}

void internal::Cholesky::reduce_scal_cf128_qf(cudaStream_t stream, const float4 scale, int32_t M, int32_t N, complex_float4* A, int32_t lda) {
  reduce_scal_dispatcher<float4, complex_float4, complex_float4* __restrict__>(stream, scale, M, N, A, lda);
}
