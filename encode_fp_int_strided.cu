
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <cub/cub.cuh>

template <int order> struct encode_fp_int {
  __device__ __forceinline__ void operator()(double f, int32_t vec_e, uint32_t (&code)[order]) {
    int32_t e;
    device::int8::encode_double_exp7_9xi8(f, e, *reinterpret_cast<uint32_t(*)[3]>(&code[0]));
    device::int8::align_expon<order>(code, e - vec_e);
  }

  __device__ __forceinline__ void operator()(float f, int32_t vec_e, uint32_t (&code)[order]) {
    int32_t e;
    device::int8::encode_float_exp7_5xi8(f, e, *reinterpret_cast<uint32_t(*)[2]>(&code[0]));
    device::int8::align_expon<order>(code, e - vec_e);
  }
};

struct uint_transpose {
  // a (a0, a1, a2, a3) -->> a (byte0[a0:3], byte1[a0:3], byte2[a0:3], byte3[a0:3])
  __device__ __forceinline__ void operator()(uint32_t (&a)[4]) {
    uint32_t a01_lo = __byte_perm(a[0], a[1], 0x5140);
    uint32_t a01_hi = __byte_perm(a[0], a[1], 0x7362);
    uint32_t a23_lo = __byte_perm(a[2], a[3], 0x5140);
    uint32_t a23_hi = __byte_perm(a[2], a[3], 0x7362);

    a[0] = __byte_perm(a01_lo, a23_lo, 0x5410);
    a[1] = __byte_perm(a01_lo, a23_lo, 0x7632);
    a[2] = __byte_perm(a01_hi, a23_hi, 0x5410);
    a[3] = __byte_perm(a01_hi, a23_hi, 0x7632);
  }

  // a (r0, i0, r1, i1, r2, i2, r3, i3) -->> a (byte0[r0:3], byte0[i0:3], byte1[r0:3] ...)
  __device__ __forceinline__ void operator()(uint32_t (&a)[8]) {
    uint32_t r01_lo = __byte_perm(a[0], a[2], 0x5140);
    uint32_t r01_hi = __byte_perm(a[0], a[2], 0x7362);
    uint32_t i01_lo = __byte_perm(a[1], a[3], 0x5140);
    uint32_t i01_hi = __byte_perm(a[1], a[3], 0x7362);
    uint32_t r23_lo = __byte_perm(a[4], a[6], 0x5140);
    uint32_t r23_hi = __byte_perm(a[4], a[6], 0x7362);
    uint32_t i23_lo = __byte_perm(a[5], a[7], 0x5140);
    uint32_t i23_hi = __byte_perm(a[5], a[7], 0x7362);

    a[0] = __byte_perm(r01_lo, r23_lo, 0x5410);
    a[1] = __byte_perm(i01_lo, i23_lo, 0x5410);
    a[2] = __byte_perm(r01_lo, r23_lo, 0x7632);
    a[3] = __byte_perm(i01_lo, i23_lo, 0x7632);
    a[4] = __byte_perm(r01_hi, r23_hi, 0x5410);
    a[5] = __byte_perm(i01_hi, i23_hi, 0x5410);
    a[6] = __byte_perm(r01_hi, r23_hi, 0x7632);
    a[7] = __byte_perm(i01_hi, i23_hi, 0x7632);
  }
};

template <class real_t, class real_const_ptr, int32_t GRID_X, int32_t GRID_Y, int32_t BLOCK_THREADS, int32_t COMPLEX, int32_t ORDER>
__global__ void encode_fp_strided_i8(int32_t order, int32_t M, int32_t N, real_const_ptr A, int32_t lda, const int32_t* __restrict__ vec_expon, uint32_t* __restrict__ inA, int32_t ldi, int32_t stride) {
  constexpr int32_t items = 4 << COMPLEX;
  constexpr int32_t elements = items * BLOCK_THREADS * GRID_Y;
  constexpr int32_t elements_block = items * BLOCK_THREADS;

  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, items>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<uint32_t, BLOCK_THREADS, 1>::TempStorage temp_store;
  real_t threadA[items];
  uint32_t threadB[items * ORDER];

  cub::BlockLoad<real_t, BLOCK_THREADS, items> block_load(temp_load);
  cub::BlockStore<uint32_t, BLOCK_THREADS, 1> block_store(temp_store);
  encode_fp_int<ORDER> encode_f;
  uint_transpose trans_f;

  for (int32_t col = blockIdx.y; col < N; col += GRID_X) {
    real_const_ptr A_i = &A[col * lda];
    int32_t vec_e = vec_expon[col];

    for (int32_t i = blockIdx.x * elements_block; i < M; i += elements) {
      int32_t num_items = min(elements_block, M - i);
      uint32_t* __restrict__ inA_i = &inA[(i >> 2) + col * ldi];
      block_load.Load(&A_i[i], threadA, num_items, real_t());

      #pragma unroll
      for (int32_t j = 0; j < items; ++j) {
        uint32_t code[ORDER]{};
        encode_f(threadA[j], vec_e, code);

        #pragma unroll
        for (int32_t k = 0; k < ORDER; ++k)
          threadB[j + k * items] = code[k];
      }

      #pragma unroll
      for (int32_t k = 0; k < ORDER; ++k)
        trans_f(*reinterpret_cast<uint32_t(*)[items]>(&threadB[k * items]));
      
      num_items = (num_items + 3) >> 2;
      for (int32_t k = 0; k < order; ++k)
        block_store.Store(&inA_i[k * stride], *reinterpret_cast<uint32_t(*)[1]>(&threadB[k]), num_items);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_x = 32;
constexpr int32_t grid_y = 64;
constexpr int32_t order_max = 5;

void internal::int8::encode_f64_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <double, const double* __restrict__, grid_x, grid_y, block_threads, 0, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const double*)A, lda, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_f32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <float, const float* __restrict__, grid_x, grid_y, block_threads, 0, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const float*)A, lda, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_cf64_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <double, const double* __restrict__, grid_x, grid_y, block_threads, 1, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order * 2, M * 2, N, (const double*)A, lda * 2, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_cf32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <float, const float* __restrict__, grid_x, grid_y, block_threads, 1, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order * 2, M * 2, N, (const float*)A, lda * 2, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}
