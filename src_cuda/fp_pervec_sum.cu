
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct i64x3 { int64_t e[3]; };

struct i64_add {
  __device__ __forceinline__ i64x3 operator()(i64x3 a, i64x3 b) {
    a.e[0] += b.e[0]; a.e[1] += b.e[1]; a.e[2] += b.e[2];
    return a;
  }
};

__device__ __forceinline__ void acc_batch(i64x3& acc, int64_t hi, int64_t mi, int64_t lo, int32_t sgn) {
  acc.e[0] = sgn ? -lo : lo; acc.e[1] = sgn ? -mi : mi; acc.e[2] = sgn ? -hi : hi;
}
__device__ __forceinline__ void acc_normalize(int64_t hi, int64_t& mi, int64_t& lo) {
  uint64_t s[2]{ uint64_t(lo) & device::int8::i63, -(uint64_t(lo) >> 63) };
  device::int8::add_shifted(s, mi, uint32_t(32));
  device::int8::add_shifted(s, hi, uint32_t(63));
  lo = int64_t(s[0]); mi = int64_t(s[1]);
}

template <class real_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int32_t M, real_const_ptr A, int64_t lda, const uint64_t* __restrict__ scale, uint64_t* __restrict__ vec_sum, int32_t incv) {
  real_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  int64_t acc_hi = 0; uint64_t acc_mi = 0, acc_lo = 0;
  int32_t sgn = 0, expon = 0;
  device::int8::extract_scale(uint32_t(scale[blockIdx.x]), sgn, expon);
  expon = -expon;

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS) {
    int32_t hi = 0; uint64_t lo = uint64_t(0);
    device::int8::quantize_f64(A_i[i], expon, hi, lo);
    acc_hi += hi; acc_mi += (lo >> 32); acc_lo += uint32_t(lo);
  }

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<i64x3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    cub::BlockReduce<i64x3, BLOCK_THREADS> block_reduce[2]{ temp_reduce[0], temp_reduce[1] };
    i64x3 threadA, threadB;
    acc_batch(threadA, acc_hi, int64_t(acc_mi), int64_t(acc_lo), sgn);
    acc_batch(threadB, acc_hi, int64_t(acc_mi), int64_t(acc_lo), sgn ^ (int32_t(threadIdx.x) & 1));
    threadA = block_reduce[0].Reduce(threadA, i64_add());
    threadB = block_reduce[1].Reduce(threadB, i64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      acc_normalize(threadB.e[2], threadB.e[1], threadB.e[0]);
      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadA.e[0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1];
      vec_sum = &vec_sum[incv]; *vec_sum = threadB.e[0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadB.e[1];
    }
  }
  else {
    __shared__ typename cub::BlockReduce<i64x3, BLOCK_THREADS>::TempStorage temp_reduce;
    cub::BlockReduce<i64x3, BLOCK_THREADS> block_reduce(temp_reduce);
    i64x3 threadA; acc_batch(threadA, acc_hi, int64_t(acc_mi), int64_t(acc_lo), sgn);
    threadA = block_reduce.Reduce(threadA, i64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      vec_sum = &vec_sum[blockIdx.x]; *vec_sum = threadA.e[0];
      vec_sum = &vec_sum[incv]; *vec_sum = threadA.e[1];
    }
  }
}

template <int32_t BLOCK_THREADS>
__global__ void vector_correction_kernel(int64_t M, int32_t N, const uint64_t* __restrict__ vec_expon, const uint64_t* __restrict__ vec_sum, uint64_t* __restrict__ vec_crr_sum, int32_t incv) {
  int32_t i = int32_t(blockIdx.x) * BLOCK_THREADS + int32_t(threadIdx.x);

  if (i < N) {
    int32_t sgn = int32_t(vec_expon[i] >> 31) & 1;
    int32_t z_hi = int32_t(vec_expon[i] >> 32);
    uint64_t z_lo = vec_expon[i + incv];
    M = sgn ? M : -M;

    uint64_t kz[2]{ vec_sum[i], vec_sum[i + incv] };
    device::int8::ima_shifted(kz, int64_t(z_lo), M, uint32_t(0));
    device::int8::ima_shifted(kz, int64_t(z_hi), M, uint32_t(63));
    vec_crr_sum[i] = kz[0];
    vec_crr_sum[i + incv] = kz[1];
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const double* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon, vec_sum, incv);

  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(4)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(M), N, vec_expon, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const float* __restrict__, 0, block_threads>
    <<< N, block_threads, 0, stream >>> (M, A, lda, vec_expon, vec_sum, incv);

  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(4)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(M), N, vec_expon, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const double* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (2 * M, (double*)A, 2 * lda, vec_expon, vec_sum, incv);

  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(6)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(2 * M), N, vec_expon, vec_sum, vec_crr_sum, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, uint64_t* vec_expon, int32_t incv) {
  uint64_t* vec_sum = &vec_expon[int64_t(incv) * int64_t(2)];
  vector_sum_kernel <const float* __restrict__, 1, block_threads>
    <<< N, block_threads, 0, stream >>> (2 * M, (float*)A, 2 * lda, vec_expon, vec_sum, incv);

  uint64_t* vec_crr_sum = &vec_expon[int64_t(incv) * int64_t(6)];
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  vector_correction_kernel <block_threads>
    <<< grid, block_threads, 0, stream >>> (int64_t(2 * M), N, vec_expon, vec_sum, vec_crr_sum, incv);
}
