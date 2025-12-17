
#include <hyacin.hpp>
#include <internal.hpp>
#include <int_fp_quantize.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t ORDER, class real_t, class matrix_t> struct quantize_func {
  const matrix_t* __restrict__ A;
  const uint64_t* __restrict__ vec_expon;
  int8_t* __restrict__ B;
  int32_t umax;
  int64_t M, lda, ldb, strideB;
  quantize_func(int64_t M, int64_t N, const matrix_t* A, int64_t lda, int32_t umax, const uint64_t* vec_expon, int8_t* B, int64_t ldb) :
    A(A), vec_expon(vec_expon), B(B), umax(umax), M(M), lda(lda), ldb(ldb), strideB(int64_t(N) * int64_t(ldb)) {}

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr int32_t COMPLEX = int32_t(sizeof(real_t) < sizeof(matrix_t));
    int64_t x = i / M, y = i - M * x;
    int32_t expon = -int32_t(vec_expon[x]);
    int8_t* B_i = &B[y + x * ldb];
    matrix_t A_i = A[y + x * lda];

    uint32_t code[3]; int8_t* bytes = (int8_t*)&code[0];
    if constexpr(COMPLEX)
      device::int8::quantize_f64_i8limbs(double(A_i.x), expon, umax, code);
    else
      device::int8::quantize_f64_i8limbs(double(A_i), expon, umax, code);

    int64_t iter = 0;
    #pragma unroll
    for (uint32_t k = 0; k < ORDER; ++k)
    { B_i[iter] = bytes[k]; iter += strideB; }

    if constexpr(COMPLEX) {
      device::int8::quantize_f64_i8limbs(double(A_i.y), expon, umax, code);
      #pragma unroll
      for (uint32_t k = 0; k < ORDER; ++k)
      { B_i[iter] = bytes[k]; iter += strideB; }
    }
  }
};

template <class real_t, class matrix_t>
inline void quantize_dispatcher(cudaStream_t stream, int32_t order, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int64_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  int64_t stride = M * N;

  switch (order) {
    case 1: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<1, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 2: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<2, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 3: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<3, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 4: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<4, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 5: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<5, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 6: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<6, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 7: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<7, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 8: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<8, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 9: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<9, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 10: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<10, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    case 11: thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, stride, quantize_func<11, real_t, matrix_t>(M, N, C, ldc, umax, vec_expon, A, lda)); break;
    default: break;
  }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double>(stream, order, M, N, C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<double>(stream, order, M, N, (double2*)C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float>(stream, order, M, N, C, ldc, umax, vec_expon, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int8_t* A, int32_t lda) {
  quantize_dispatcher<float>(stream, order, M, N, (float2*)C, ldc, umax, vec_expon, A, lda);
}
