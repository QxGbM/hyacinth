
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct __align__(16) i126 { uint64_t e[2]; };
struct __align__(32) i126x2 { uint64_t e[2][2]; };

struct i126_add {
  __device__ __forceinline__ i126 operator()(i126 a, i126 b) {
    device::int8::add_shifted(a.e, int64_t(b.e[0]), uint32_t(0));
    device::int8::add_shifted(a.e, int64_t(b.e[1]), uint32_t(63));
    return a;
  }
  __device__ __forceinline__ i126x2 operator()(i126x2 a, i126x2 b) {
    device::int8::add_shifted(a.e[0], int64_t(b.e[0][0]), uint32_t(0));
    device::int8::add_shifted(a.e[0], int64_t(b.e[0][1]), uint32_t(63));
    device::int8::add_shifted(a.e[1], int64_t(b.e[1][0]), uint32_t(0));
    device::int8::add_shifted(a.e[1], int64_t(b.e[1][1]), uint32_t(63));
    return a;
  }
};

template <class real_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int32_t M, real_const_ptr A, int64_t lda, const uint32_t* __restrict__ scale, uint64_t* __restrict__ vec_sum, int32_t incv) {
  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  i126 threadA; threadA.e[0] = threadA.e[1] = uint64_t(0);
  int32_t sgn = 0, expon = 0;
  device::int8::extract_scale(scale[blockIdx.x], sgn, expon);
  uint32_t sft_init = uint32_t(-expon);

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    int64_t q; uint32_t sft = sft_init;
    device::int8::round_f64_i64s(double(A_i[i]), q, sft);
    device::int8::add_shifted(threadA.e, q, sft);
  }

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<i126x2, BLOCK_THREADS>::TempStorage temp_reduce;
    cub::BlockReduce<i126x2, BLOCK_THREADS> block_reduce(temp_reduce);

    union { i126 val[2]; i126x2 state; } threadB{ threadA, threadA };
    if (sgn)
      device::int8::negate_shifted(threadB.val[0].e);
    if (sgn ^ (int32_t(threadIdx.x) & 1))
      device::int8::negate_shifted(threadB.val[1].e);
    
    if (0 < M)
      threadB.state = block_reduce.Reduce(threadB.state, i126_add());

    if (threadIdx.x == 0) {
      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadB.state.e[0][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadB.state.e[0][1];
      vec_sum = &vec_sum[incv]; *vec_sum = threadB.state.e[1][0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadB.state.e[1][1];
    }
  }
  else {
    __shared__ typename cub::BlockReduce<i126, BLOCK_THREADS>::TempStorage temp_reduce;
    cub::BlockReduce<i126, BLOCK_THREADS> block_reduce(temp_reduce);

    if (sgn)
      device::int8::negate_shifted(threadA.e);

    if (0 < M)
      threadA = block_reduce.Reduce(threadA, i126_add());

    if (threadIdx.x == 0) {
      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadA.e[0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1];
    }
  }
}

template <int32_t BLOCK_THREADS>
__global__ void vector_correction_kernel(int64_t M, int32_t N, int64_t c_lo, int64_t c_hi, const double* __restrict__ z, const uint64_t* __restrict__ vec_sum, uint64_t* __restrict__ vec_crr_sum, int32_t incv) {
  int32_t i = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x);

  if (i < N) {
    uint64_t kz[2]{ vec_sum[i], vec_sum[i + incv] };
    device::int8::combine_zc(z[i], c_lo, c_hi);
    device::int8::ima_shifted(kz, c_lo, M, uint32_t(0));
    device::int8::ima_shifted(kz, c_hi, M, uint32_t(63));
    vec_crr_sum[i] = kz[0];
    vec_crr_sum[i + incv] = kz[1];
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int64_t c_lo, int64_t c_hi, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const double* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, (uint32_t*)vec_expon, vec_sum, incv);

  double* z = (double*)&vec_expon[incv];
  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(4)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(M), N, c_lo, c_hi, z, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int64_t c_lo, int64_t c_hi, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const float* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, (uint32_t*)vec_expon, vec_sum, incv);

  double* z = (double*)&vec_expon[incv];
  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(4)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(M), N, c_lo, c_hi, z, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int64_t c_lo, int64_t c_hi, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const double* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (2 * M, (double*)A, 2 * lda, (uint32_t*)vec_expon, vec_sum, incv);

  double* z = (double*)&vec_expon[incv];
  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(6)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(2 * M), N, c_lo, c_hi, z, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int64_t c_lo, int64_t c_hi, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const float* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (2 * M, (float*)A, 2 * lda, (uint32_t*)vec_expon, vec_sum, incv);

  double* z = (double*)&vec_expon[incv];
  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(6)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(2 * M), N, c_lo, c_hi, z, vec_sum, vec_crr_sum, incv);
}
