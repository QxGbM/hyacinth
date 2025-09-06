
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <int32_t order, class real_t, class matrix_t> struct encode_func {
  const matrix_t* __restrict__ A;
  const int32_t* __restrict__ vec_expon;
  int8_t* __restrict__ B;
  int64_t M, lda, ldb, strideB;
  encode_func(int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vec_expon, int8_t* B, int32_t ldb) :
    A(A), vec_expon(vec_expon), B(B), M(M), lda(lda), ldb(ldb), strideB(int64_t(N) * int64_t(ldb)) {}

  __device__ __forceinline__ void encode(double f, int32_t vec_e, uint32_t (&code)[4]) {
    device::int8::encode_double_align<device::Config::exp_base>(f, vec_e, code);
  }

  __device__ __forceinline__ void encode(float f, int32_t vec_e, uint32_t (&code)[2]) {
    device::int8::encode_float_align<device::Config::exp_base>(f, vec_e, code);
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr int32_t code_words = sizeof(real_t) / 2;
    constexpr int32_t code_bytes = sizeof(real_t) * 2;
    union { uint32_t code[code_words]; int8_t bytes[code_bytes]; } c;
    int64_t x = i / M, y = i - M * x;
    int32_t expon = vec_expon[x];
    int8_t* B_rl = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    if constexpr(sizeof(real_t) < sizeof(matrix_t)) {
      union { uint32_t code[code_words]; int8_t bytes[code_bytes]; } c_im;
      int8_t* B_im = &B_rl[strideB * int64_t(order)];
      encode(A_i.x, expon, c.code);
      encode(A_i.y, expon, c_im.code);

      #pragma unroll
      for (int32_t k = 0; k < order; ++k) {
        B_rl[int64_t(k) * strideB] = c.bytes[k];
        B_im[int64_t(k) * strideB] = c_im.bytes[k];
      }
    }
    else {
      encode(A_i, expon, c.code);
      #pragma unroll
      for (int32_t k = 0; k < order; ++k)
        B_rl[int64_t(k) * strideB] = c.bytes[k];
    }
  }
};

template <class real_t, class matrix_t>
inline void encode_dispatcher(cudaStream_t stream, int32_t order, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, const int32_t* vec_expon, int8_t* A, int64_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t stride = M * N;

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<1, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<2, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<3, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<4, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<5, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<6, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<7, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<8, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<9, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<10, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
    default: break;
  }

  if constexpr (device::Config::exp_base < 7) {
    switch (order) {
      case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<11, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 12: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<12, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 13: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<13, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      default: break;
    }
  }

  if constexpr (device::Config::exp_base < 5) {
    switch (order) {
      case 14: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<14, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 15: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<15, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      case 16: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, encode_func<16, real_t, matrix_t>(M, N, C, ldc, vec_expon, A, lda)); break;
      default: break;
    }
  }
}

void internal::int8::encode_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  encode_dispatcher<double, double>(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::encode_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  encode_dispatcher<float, float>(stream, order, M, N, C, ldc, vec_expon, A, lda);
}

void internal::int8::encode_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  encode_dispatcher<double, double2>(stream, order, M, N, (double2*)C, ldc, vec_expon, A, lda);
}

void internal::int8::encode_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  encode_dispatcher<float, float2>(stream, order, M, N, (float2*)C, ldc, vec_expon, A, lda);
}
