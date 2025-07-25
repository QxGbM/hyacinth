
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>

/*template <int order> struct decode {
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

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t PACK_SIZE>
__global__ void decode_strided_i32_real(int32_t order, int32_t N, const int32_t* __restrict__ vec_expon, const int32_t* __restrict__ A, int32_t lda, int32_t strideA, real_ptr C, int32_t ldc) {
  constexpr int32_t elements = ITEMS_PER_THREAD * BLOCK_THREADS;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  int32_t threadA[ITEMS_PER_THREAD * PACK_SIZE];
  real_t threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  encode_fp_int<ORDER> encode_f;
  uint_transpose trans_f;

  for (int32_t col = blockIdx.x; col < N; col += GRID_BLOCKS) {
    real_const_ptr A_i = &A[col * lda];
    int32_t vec_e = vec_expon[col];

    for (int32_t i = 0; i < N; i += elements) {
      int32_t num_items = min(elements, N - i);
      uint32_t* __restrict__ inA_i = &inA[(i >> COMPLEX) + col * ldi];
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
      
      num_items = (num_items + items - 1) >> COMPLEX;
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
  encode_fp_strided_i8 <double, const double* __restrict__, grid_x, grid_y, block_threads, 2, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const double*)A, lda, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_f32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <float, const float* __restrict__, grid_x, grid_y, block_threads, 2, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order, M, N, (const float*)A, lda, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_cf64_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <double, const double* __restrict__, grid_x, grid_y, block_threads, 3, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order * 2, M * 2, N, (const double*)A, lda * 2, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}

void internal::int8::encode_cf32_order20(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, const int32_t* vec_expon, int8_t* inA, int32_t ldi, int32_t stride) {
  encode_fp_strided_i8 <float, const float* __restrict__, grid_x, grid_y, block_threads, 3, order_max>
  <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (order * 2, M * 2, N, (const float*)A, lda * 2, vec_expon, (uint32_t*)inA, ldi >> 2, stride >> 2);
}*/
