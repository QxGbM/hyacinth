
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

template <int32_t SET> struct acc_i32 {
  __device__ __forceinline__ void operator()(double& f, int32_t const (&i)[1], int32_t expon) {
    if constexpr(SET)
      f = scalbn(double(i[0]), expon);
    else
      f += scalbn(double(i[0]), expon);
  }

  __device__ __forceinline__ void operator()(float& f, int32_t const (&i)[1], int32_t expon) {
    if constexpr (SET)
      f = scalbnf(float(i[0]), expon);
    else
      f += scalbnf(float(i[0]), expon);
  }

  __device__ __forceinline__ void operator()(double2& f, int32_t const (&i)[1], int32_t expon) {
    if constexpr (SET)
      f = make_double2(scalbn(double(i[0]), expon), 0.);
    else
      f = device::dd::add_sd(f, scalbn(double(i[0]), expon));
  }

  __device__ __forceinline__ void operator()(float4& f, int32_t const (&i)[1], int32_t expon) {
    float c1 = float(i[0]);
    float c2 = float(i[0] - int32_t(c1));
    if constexpr (SET)
      f = make_float4(scalbnf(c1, expon), scalbnf(c2, expon), 0.f, 0.f);
    else
      f = device::qf::add_df(f, make_float2(scalbnf(c1, expon), scalbnf(c2, expon)));
  }

  __device__ __forceinline__ void operator()(cuDoubleComplex& f, int32_t const (&i)[2], int32_t expon) {
    operator()(f.x, *(int32_t const(*)[1])(&i[0]), expon);
    operator()(f.y, *(int32_t const(*)[1])(&i[1]), expon);
  }

  __device__ __forceinline__ void operator()(cuComplex& f, int32_t const (&i)[2], int32_t expon) {
    operator()(f.x, *(int32_t const(*)[1])(&i[0]), expon);
    operator()(f.y, *(int32_t const(*)[1])(&i[1]), expon);
  }

  __device__ __forceinline__ void operator()(complex_double2& f, int32_t const (&i)[2], int32_t expon) {
    operator()(f.real, *(int32_t const(*)[1])(&i[0]), expon);
    operator()(f.imag, *(int32_t const(*)[1])(&i[1]), expon);
  }

  __device__ __forceinline__ void operator()(complex_float4& f, int32_t const (&i)[2], int32_t expon) {
    operator()(f.real, *(int32_t const(*)[1])(&i[0]), expon);
    operator()(f.imag, *(int32_t const(*)[1])(&i[1]), expon);
  }
};

template <int32_t SET, class real_t, int32_t ITEMS_PER_THREAD, int32_t COMPLEX>
__device__ __forceinline__ void decode_i32_array(real_t (&val)[ITEMS_PER_THREAD], int32_t const (&c)[COMPLEX][ITEMS_PER_THREAD], int32_t const (&expon)[ITEMS_PER_THREAD], int32_t expon_k) {
  acc_i32<SET> acc_f;
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    int32_t ci[COMPLEX];

    #pragma unroll
    for (int32_t j = 0; j < COMPLEX; ++j)
      ci[j] = c[j][i];
    acc_f(val[i], ci, expon[i] + expon_k);
  }
}

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE, int32_t COMPLEX>
__global__ void decode_kernel(int32_t k_begin, int32_t k_len, int32_t N, const int32_t* __restrict__ vec_expon, const int32_t* __restrict__ A, const int32_t* __restrict__ B, int32_t lda, uint64_t strideA, real_ptr C, int32_t ldc) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  int32_t row_expon[ITEMS_PER_THREAD], c[COMPLEX][ITEMS_PER_THREAD];
  real_t val[ITEMS_PER_THREAD];

  for (int32_t col = blockIdx.x; col < N; col += GRID_BLOCKS) {
    uint64_t A_col = uint64_t(col) * uint64_t(lda);
    real_ptr C_col = &C[uint64_t(col) * uint64_t(ldc)];
    int32_t col_expon = vec_expon[col] + k_begin;

    for (int32_t i = 0; i < N; i += elements) {
      uint64_t A_ij = uint64_t(i) + A_col;
      int32_t num_items = min(elements, N - i);
      block_load.Load(&vec_expon[i], row_expon, num_items, 0);

      #pragma unroll
      for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
        row_expon[j] = BASE * (row_expon[j] + col_expon);

      block_load.Load(&A[A_ij], c[0], num_items, 0);
      if constexpr(1 < COMPLEX)
        block_load.Load(&B[A_ij], c[1], num_items, 0);
      decode_i32_array<1>(val, c, row_expon, 0);

      for (int32_t k = 1; k < k_len; ++k) {
        uint64_t A_k = A_ij + uint64_t(k) * strideA;

        block_load.Load(&A[A_k], c[0], num_items, 0);
        if constexpr(1 < COMPLEX)
          block_load.Load(&B[A_k], c[1], num_items, 0);
        decode_i32_array<0>(val, c, row_expon, BASE * k);
      }

      block_store.Store(&C_col[i], val, num_items);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

void internal::int8::decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <double, double* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, nullptr, lda, strideA, C, ldc);
}

void internal::int8::decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <float, float* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, nullptr, lda, strideA, C, ldc);
}

void internal::int8::decode_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, double2* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, nullptr, lda, strideA, C, ldc);
}

void internal::int8::decode_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, float4* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, nullptr, lda, strideA, C, ldc);
}

void internal::int8::decode_cf64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, std::complex<double>* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <cuDoubleComplex, cuDoubleComplex* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 2>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, &A[uint64_t(order) * strideA], lda, strideA, (cuDoubleComplex*)C, ldc);
}

void internal::int8::decode_cf32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, std::complex<float>* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <cuComplex, cuComplex* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 2>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, &A[uint64_t(order) * strideA], lda, strideA, (cuComplex*)C, ldc);
}

void internal::int8::decode_complex_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, complex_double2* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <complex_double2, complex_double2* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 2>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, &A[uint64_t(order) * strideA], lda, strideA, C, ldc);
}

void internal::int8::decode_complex_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, const int32_t* vec_expon, const int32_t* A, int32_t lda, complex_float4* C, int32_t ldc) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  decode_kernel <complex_float4, complex_float4* __restrict__, grid_blocks, block_threads, items_per_thread, exp_base, 2>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, N, vec_expon, A, &A[uint64_t(order) * strideA], lda, strideA, C, ldc);
}

