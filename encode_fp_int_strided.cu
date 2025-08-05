
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <cub/cub.cuh>

template <int32_t base, int32_t order> struct encode_fp_int {
  __device__ __forceinline__ void operator()(double f, int32_t vec_e, uint32_t (&code)[order]) {
    int32_t e;
    device::int8::encode_double<base>(f, e, *(uint32_t(*)[4])(&code[0]));
    device::int8::align_expon<order>(code, e - vec_e);
  }

  __device__ __forceinline__ void operator()(float f, int32_t vec_e, uint32_t (&code)[order]) {
    int32_t e;
    device::int8::encode_float<base>(f, e, *(uint32_t(*)[2])(&code[0]));
    device::int8::align_expon<order>(code, e - vec_e);
  }
};

// a (a0, a1, a2, a3) -->> a (byte0[a0:3], byte1[a0:3], byte2[a0:3], byte3[a0:3])
__device__ __forceinline__ void transpose_uint32(uint32_t (&a)[4]) {
  uint32_t a01_lo = __byte_perm(a[0], a[1], 0x5140);
  uint32_t a01_hi = __byte_perm(a[0], a[1], 0x7362);
  uint32_t a23_lo = __byte_perm(a[2], a[3], 0x5140);
  uint32_t a23_hi = __byte_perm(a[2], a[3], 0x7362);

  a[0] = __byte_perm(a01_lo, a23_lo, 0x5410);
  a[1] = __byte_perm(a01_lo, a23_lo, 0x7632);
  a[2] = __byte_perm(a01_hi, a23_hi, 0x5410);
  a[3] = __byte_perm(a01_hi, a23_hi, 0x7632);
}

template <int32_t BASE, int32_t ORDER, class real_t>
__device__ __forceinline__ void encode_fp_array(uint32_t (&code)[4 * ORDER], real_t const (&val)[4], int32_t expon) {
  encode_fp_int<BASE, ORDER> encode_f;

  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    uint32_t c[ORDER]{};
    encode_f(val[i], expon, c);

    #pragma unroll
    for (int32_t k = 0; k < ORDER; ++k)
      code[i + (k << 2)] = c[k];
  }

  #pragma unroll
  for (int32_t k = 0; k < ORDER; ++k)
    transpose_uint32(*(uint32_t(*)[4])(&code[k << 2]));
}

template <class real_t, class real_const_ptr, int32_t GRID_X, int32_t GRID_Y, int32_t BLOCK_THREADS, int32_t COMPLEX, int32_t BASE, int32_t ORDER>
__global__ void encode_kernel(int32_t order, int32_t M, int32_t N, real_const_ptr C, int32_t ldc, const int32_t* __restrict__ vec_expon, uint32_t* __restrict__ A, uint32_t* __restrict__ B, int32_t lda, uint64_t strideA) {
  constexpr int32_t items = 4 << COMPLEX;
  constexpr int32_t elements_block = items * BLOCK_THREADS;
  constexpr int32_t elements = GRID_Y * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t M2 = M & (elements_block - 1), M1 = M - M2, i8_M2 = (M2 + items - 1) >> (2 + COMPLEX);

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, items>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<uint32_t, BLOCK_THREADS, 1>::TempStorage temp_store;
  real_t val[items];

  cub::BlockLoad<real_t, BLOCK_THREADS, items> block_load(temp_load);
  cub::BlockStore<uint32_t, BLOCK_THREADS, 1> block_store(temp_store);

  for (int32_t col = blockIdx.y; col < N; col += GRID_X) {
    real_const_ptr C_i = &C[uint64_t(col) * uint64_t(ldc)];
    uint64_t A_i = uint64_t(col) * uint64_t(lda);
    int32_t vec_e = vec_expon[col];

    for (int32_t i = block_offset; i < M1; i += elements) {
      block_load.Load(&C_i[i], val);

      if constexpr(COMPLEX) {
        real_t C_rl[4]{ val[0], val[2], val[4], val[6] };
        real_t C_im[4]{ val[1], val[3], val[5], val[7] };
        uint32_t A_rl[4 * ORDER], A_im[4 * ORDER];

        encode_fp_array<BASE, ORDER>(A_rl, C_rl, vec_e);
        encode_fp_array<BASE, ORDER>(A_im, C_im, vec_e);

        uint64_t A_row = A_i + (i >> (2 + COMPLEX));
        for (int32_t k = 0; k < order; ++k) {
          uint64_t A_k = A_row + uint64_t(k) * strideA;
          block_store.Store(&A[A_k], *(uint32_t(*)[1])(&A_rl[k]));
          block_store.Store(&B[A_k], *(uint32_t(*)[1])(&A_im[k]));
        }
      }
      else {
        uint32_t A_rl[4 * ORDER];
        encode_fp_array<BASE, ORDER>(A_rl, val, vec_e);
        uint64_t A_row = A_i + (i >> (2 + COMPLEX));
        for (int32_t k = 0; k < order; ++k)
          block_store.Store(&A[A_row + uint64_t(k) * strideA], *(uint32_t(*)[1])(&A_rl[k]));
      }
    }

    if (0 < M2 && blockIdx.x == 0) {
      block_load.Load(&C_i[M1], val, M2, real_t());

      if constexpr(COMPLEX) {
        real_t C_rl[4]{ val[0], val[2], val[4], val[6] };
        real_t C_im[4]{ val[1], val[3], val[5], val[7] };
        uint32_t A_rl[4 * ORDER], A_im[4 * ORDER];

        encode_fp_array<BASE, ORDER>(A_rl, C_rl, vec_e);
        encode_fp_array<BASE, ORDER>(A_im, C_im, vec_e);

        uint64_t A_row = A_i + (M1 >> (2 + COMPLEX));
        for (int32_t k = 0; k < order; ++k) {
          uint64_t A_k = A_row + uint64_t(k) * strideA;
          block_store.Store(&A[A_k], *(uint32_t(*)[1])(&A_rl[k]), i8_M2);
          block_store.Store(&B[A_k], *(uint32_t(*)[1])(&A_im[k]), i8_M2);
        }
      }
      else {
        uint32_t A_rl[4 * ORDER];
        encode_fp_array<BASE, ORDER>(A_rl, val, vec_e);
        uint64_t A_row = A_i + (M1 >> (2 + COMPLEX));
        for (int32_t k = 0; k < order; ++k)
          block_store.Store(&A[A_row + uint64_t(k) * strideA], *(uint32_t(*)[1])(&A_rl[k]), i8_M2);
      }
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_x = 32;
constexpr int32_t grid_y = 64;

void internal::int8::encode_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  lda = lda >> 2;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  encode_kernel <double, const double* __restrict__, grid_x, grid_y, block_threads, 0, exp_base, order_max>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const double*)C, ldc, vec_expon, (uint32_t*)A, nullptr, lda, strideA);
}

void internal::int8::encode_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  lda = lda >> 2;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  encode_kernel <float, const float* __restrict__, grid_x, grid_y, block_threads, 0, exp_base, order_max>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const float*)C, ldc, vec_expon, (uint32_t*)A, nullptr, lda, strideA);
}

void internal::int8::encode_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  lda = lda >> 2;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  uint32_t* C_im = &((uint32_t*)A)[uint64_t(order) * strideA];
  encode_kernel <double, const double* __restrict__, grid_x, grid_y, block_threads, 1, exp_base, order_max>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M * 2, N, (const double*)C, ldc * 2, vec_expon, (uint32_t*)A, C_im, lda, strideA);
}

void internal::int8::encode_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  lda = lda >> 2;
  uint64_t strideA = uint64_t(N) * uint64_t(lda);
  uint32_t* C_im = &((uint32_t*)A)[uint64_t(order) * strideA];
  encode_kernel <float, const float* __restrict__, grid_x, grid_y, block_threads, 1, exp_base, order_max>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M * 2, N, (const float*)C, ldc * 2, vec_expon, (uint32_t*)A, C_im, lda, strideA);
}
