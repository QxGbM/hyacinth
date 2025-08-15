
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <class real_t, class matrix_t> struct encode_func {
  const matrix_t* A;
  const int32_t* vec_expon;
  int8_t* B;
  int32_t order;
  int64_t M, lda, ldb, strideB;
  encode_func(int32_t order, int32_t M, int32_t N, const matrix_t* A, int32_t lda, const int32_t* vec_expon, int8_t* B, int32_t ldb) :
    A(A), vec_expon(vec_expon), B(B), order(order), M(M), lda(lda), ldb(ldb), strideB(int64_t(N) * int64_t(ldb)) {}

  __device__ __forceinline__ void encode(double f, int32_t vec_e, uint32_t (&code)[4]) {
    int32_t e;
    device::int8::encode_double<device::Config::exp_base>(f, e, code);
    device::int8::align_expon(code, e - vec_e);
  }

  __device__ __forceinline__ void encode(float f, int32_t vec_e, uint32_t (&code)[2]) {
    int32_t e;
    device::int8::encode_float<device::Config::exp_base>(f, e, code);
    device::int8::align_expon(code, e - vec_e);
  }

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr int32_t code_words = sizeof(real_t) / 2;
    constexpr int32_t code_bytes = sizeof(real_t) * 2;
    union { uint32_t code[code_words]; int8_t bytes[code_bytes]; } c;
    int64_t x = i / M, y = i - M * x;
    int32_t expon = vec_expon[x];

    if constexpr(sizeof(real_t) < sizeof(matrix_t)) {
      union { uint32_t code[code_words]; int8_t bytes[code_bytes]; } c_im;
      union { matrix_t comp; real_t real[2]; } A_i{A[y + x * lda]};
      int8_t* B_rl = &B[y + x * ldb], *B_im = &B_rl[strideB * int64_t(order)];
      encode(A_i.real[0], expon, c.code);
      encode(A_i.real[1], expon, c_im.code);

      #pragma unroll
      for (int32_t k = 0; k < code_bytes; ++k) {
        if (k < order) {
          B_rl[int64_t(k) * strideB] = c.bytes[k];
          B_im[int64_t(k) * strideB] = c_im.bytes[k];
        }
      }
    }
    else {
      real_t A_i = A[y + x * lda];
      int8_t* B_i = &B[y + x * ldb];
      encode(A_i, expon, c.code);

      #pragma unroll
      for (int32_t k = 0; k < code_bytes; ++k) {
        if (k < order)
          B_i[int64_t(k) * strideB] = c.bytes[k];
      }
    }
  }
};

void internal::int8::encode_f64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const double* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  encode_func<double, double> encode(order, M, N, C, ldc, vec_expon, A, lda);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), encode);
}

void internal::int8::encode_f32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const float* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  encode_func<float, float> encode(order, M, N, C, ldc, vec_expon, A, lda);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), encode);
}

void internal::int8::encode_cf64(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  encode_func<double, double2> encode(order, M, N, (double2*)C, ldc, vec_expon, A, lda);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), encode);
}

void internal::int8::encode_cf32(cudaStream_t stream, int32_t order, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, const int32_t* vec_expon, int8_t* A, int32_t lda) {
  thrust::counting_iterator<int64_t> iter(0);
  encode_func<float, float2> encode(order, M, N, (float2*)C, ldc, vec_expon, A, lda);
  thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, uint64_t(M) * uint64_t(N), encode);
}
