
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

template<uint32_t ITEMS> struct __align__(32) i126 { uint64_t e[ITEMS][2]{}; };

struct i126_add {
  template <uint32_t ITEMS>
  __device__ __forceinline__ i126<ITEMS> operator()(i126<ITEMS> a, i126<ITEMS> b) {
    #pragma unroll
    for (uint32_t i = 0; i < ITEMS; ++i) {
      device::int8::add_shifted(a.e[i], int64_t(b.e[i][0]), uint32_t(0));
      device::int8::add_shifted(a.e[i], int64_t(b.e[i][1]), uint32_t(63));
    }
    return a;
  }
};

template <class matrix_t, class matrix_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int32_t M, matrix_const_ptr A, int64_t lda, int64_t c_lo, int64_t c_hi, const uint32_t* __restrict__ scale, const double* __restrict__ z, uint64_t* __restrict__ vec_sum, int32_t incv) {
  constexpr uint32_t ITEMS = uint32_t(COMPLEX + 2);
  __shared__ typename cub::BlockReduce<i126<ITEMS>, BLOCK_THREADS>::TempStorage temp_reduce;
  cub::BlockReduce<i126<ITEMS>, BLOCK_THREADS> block_reduce(temp_reduce);

  matrix_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  i126<ITEMS> threadA; int32_t sgn = 0, expon = 0;
  device::int8::extract_scale(scale[blockIdx.x], sgn, expon);
  device::int8::combine_zc(z[blockIdx.x], c_lo, c_hi);
  expon = -expon;

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    int64_t q; uint32_t sft; matrix_t x = A_i[i];
    if constexpr(COMPLEX) {
      device::int8::round_f64_i64s(scalbn(double(sgn ? -x.x : x.x), expon), q, sft);
      device::int8::add_shifted(threadA.e[0], q, sft);
      device::int8::round_f64_i64s(scalbn(double(sgn ? -x.y : x.y), expon), q, sft);
      device::int8::add_shifted(threadA.e[1], q, sft);
      device::int8::add_shifted(threadA.e[2], c_lo, uint32_t(1));
      device::int8::add_shifted(threadA.e[2], c_hi, uint32_t(64));
    }
    else {
      device::int8::round_f64_i64s(scalbn(double(sgn ? -x : x), expon), q, sft);
      device::int8::add_shifted(threadA.e[0], q, sft);
      device::int8::add_shifted(threadA.e[1], c_lo, uint32_t(0));
      device::int8::add_shifted(threadA.e[1], c_hi, uint32_t(63));
    }
  }

  if (0 < M)
    threadA = block_reduce.Reduce(threadA, i126_add());

  if (threadIdx.x == 0)
    if constexpr(COMPLEX) {
      int64_t im_lo = threadA.e[1][0], im_hi = threadA.e[1][1];
      threadA.e[1][0] = threadA.e[0][0];
      threadA.e[1][1] = threadA.e[0][1];

      device::int8::add_shifted(threadA.e[0], im_lo, uint32_t(0));
      device::int8::add_shifted(threadA.e[1], -im_lo, uint32_t(0));
      device::int8::add_shifted(threadA.e[0], im_hi, uint32_t(63));
      device::int8::add_shifted(threadA.e[1], -im_hi, uint32_t(63));
      device::int8::add_shifted(threadA.e[2], threadA.e[0][0], uint32_t(0));
      device::int8::add_shifted(threadA.e[2], threadA.e[0][1], uint32_t(63));

      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadA.e[0][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[0][1];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1][1];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[2][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[2][1];
    }
    else {
      device::int8::add_shifted(threadA.e[1], threadA.e[0][0], uint32_t(0));
      device::int8::add_shifted(threadA.e[1], threadA.e[0][1], uint32_t(63));

      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadA.e[0][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[0][1];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1][1];
    }
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int64_t c_lo, int64_t c_hi, const uint32_t* scale, const double* z, uint64_t* vec_sum, int32_t incv) {
  vector_sum_kernel <double, const double* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, c_lo, c_hi, scale, z, vec_sum, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int64_t c_lo, int64_t c_hi, const uint32_t* scale, const double* z, uint64_t* vec_sum, int32_t incv) {
  vector_sum_kernel <float, const float* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, c_lo, c_hi, scale, z, vec_sum, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int64_t c_lo, int64_t c_hi, const uint32_t* scale, const double* z, uint64_t* vec_sum, int32_t incv) {
  vector_sum_kernel <double2, const double2* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (M, (double2*)A, lda, c_lo, c_hi, scale, z, vec_sum, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int64_t c_lo, int64_t c_hi, const uint32_t* scale, const double* z, uint64_t* vec_sum, int32_t incv) {
  vector_sum_kernel <float2, const float2* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (M, (float2*)A, lda, c_lo, c_hi, scale, z, vec_sum, incv);
}
