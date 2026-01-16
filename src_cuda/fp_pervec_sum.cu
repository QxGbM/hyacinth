
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

__device__ __forceinline__ void accumulate(double x, int32_t expon, int64_t& acc_hi, int64_t& acc_mi, int64_t& acc_lo) {
  int64_t q = device::int8::round_f64(x, 0x8000000000000000llu, expon, expon);
  acc_hi -= int64_t(q >> (63 - expon)); uint64_t q_sft = uint64_t(q) << expon;
  acc_mi -= int64_t(uint32_t(q_sft >> 32) & device::int8::i31);
  acc_lo -= int64_t(uint32_t(q_sft));
}
__device__ __forceinline__ void acc_normalize(int64_t hi, int64_t& mi, int64_t& lo) {
  uint64_t s[2]{ uint64_t(lo) & device::int8::i63, -(uint64_t(lo) >> 63) };
  device::int8::add_shifted(s, mi, uint32_t(32));
  device::int8::add_shifted(s, hi, uint32_t(63));
  lo = int64_t(s[0]); mi = int64_t(s[1]);
}

template <class matrix_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int32_t M, matrix_const_ptr A, int64_t lda, int32_t umax, uint64_t* __restrict__ vec_expon, int32_t incv) {
  matrix_const_ptr A_i = &A[int64_t(blockIdx.x) * lda];
  i64x3 threadA; int64_t iter;
  int32_t expon = umax - int32_t(vec_expon[blockIdx.x]);
  threadA.e[0] = threadA.e[1] = threadA.e[2] = int64_t(0);

  for (int32_t i = threadIdx.x; i < M; i += BLOCK_THREADS)
    accumulate(A_i[i], expon, threadA.e[2], threadA.e[1], threadA.e[0]);

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<i64x3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    cub::BlockReduce<i64x3, BLOCK_THREADS> block_reduce[2]{ temp_reduce[0], temp_reduce[1] };
    i64x3 threadB; int32_t sgn = int32_t(threadIdx.x) & 1;
    threadB.e[0] = sgn ? threadA.e[0] : -threadA.e[0];
    threadB.e[1] = sgn ? threadA.e[1] : -threadA.e[1];
    threadB.e[2] = sgn ? threadA.e[2] : -threadA.e[2];
    threadA = block_reduce[0].Reduce(threadA, i64_add());
    threadB = block_reduce[1].Reduce(threadB, i64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      acc_normalize(threadB.e[2], threadB.e[1], threadB.e[0]);
      vec_expon[iter = int64_t(blockIdx.x) + int64_t(incv)] = threadA.e[0];
      vec_expon[iter += incv] = threadA.e[1];
      vec_expon[iter += incv] = threadB.e[0];
      vec_expon[iter += incv] = threadB.e[1];
    }
  }
  else {
    __shared__ typename cub::BlockReduce<i64x3, BLOCK_THREADS>::TempStorage temp_reduce;
    cub::BlockReduce<i64x3, BLOCK_THREADS> block_reduce(temp_reduce);
    threadA = block_reduce.Reduce(threadA, i64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      vec_expon[iter = int64_t(blockIdx.x) + int64_t(incv)] = threadA.e[0];
      vec_expon[iter += incv] = threadA.e[1];
    }
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  vector_sum_kernel<const double* __restrict__, 0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  vector_sum_kernel<const float* __restrict__, 0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, umax, vec_expon, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  vector_sum_kernel<const double* __restrict__, 1, block_threads> <<< N, block_threads, 0, stream >>> (2 * M, (double*)A, 2 * lda, umax, vec_expon, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  vector_sum_kernel<const float* __restrict__, 1, block_threads> <<< N, block_threads, 0, stream >>> (2 * M, (float*)A, 2 * lda, umax, vec_expon, incv);
}
