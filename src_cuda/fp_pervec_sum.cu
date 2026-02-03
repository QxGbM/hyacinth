
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct u64x3 { uint64_t e[3]; };

struct u64_add {
  __device__ __forceinline__ u64x3 operator()(u64x3 a, u64x3 b) 
  { a.e[0] += b.e[0]; a.e[1] += b.e[1]; a.e[2] += b.e[2]; return a; }
};

__device__ __forceinline__ void accumulate(double x, int32_t expon, uint64_t lo, uint32_t hi, uint64_t& acc_hi, uint64_t& acc_mi, uint64_t& acc_lo) {
  int64_t q = device::int8::round_f64(x, expon, expon);
  lo += (uint64_t(q) << expon) & device::int8::i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  acc_hi += hi; acc_mi += uint32_t(lo >> 32) & device::int8::i31; acc_lo += uint32_t(lo);
}
__device__ __forceinline__ void acc_normalize(uint64_t hi, uint64_t& mi, uint64_t& lo) {
  uint64_t s[2]{ lo, hi };
  device::int8::add_shifted(s, mi, uint32_t(32));
  lo = -s[0]; mi = -s[1];
}

template <class matrix_const_ptr, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int64_t M, matrix_const_ptr A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax, uint64_t* __restrict__ vec_expon, int64_t incv) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  int64_t iter = int64_t(blockIdx.x) * lda + int64_t(threadIdx.x), iter_end = iter + M;
  int32_t expon = umax - int32_t(vec_expon[blockIdx.x]);
  u64x3 threadA; threadA.e[0] = threadA.e[1] = threadA.e[2] = uint64_t(0);

  for (int64_t i = iter; i < iter_end; i += inci)
    accumulate(A[i], expon, lo, hi, threadA.e[2], threadA.e[1], threadA.e[0]);

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<u64x3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    cub::BlockReduce<u64x3, BLOCK_THREADS> block_reduce[2]{ temp_reduce[0], temp_reduce[1] };
    u64x3 threadB; int32_t sgn = int32_t(threadIdx.x) & 1;
    threadB.e[0] = sgn ? threadA.e[0] : -threadA.e[0];
    threadB.e[1] = sgn ? threadA.e[1] : -threadA.e[1];
    threadB.e[2] = sgn ? threadA.e[2] : -threadA.e[2];
    threadA = block_reduce[0].Reduce(threadA, u64_add());
    threadB = block_reduce[1].Reduce(threadB, u64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      acc_normalize(threadB.e[2], threadB.e[1], threadB.e[0]);
      vec_expon[iter = int64_t(blockIdx.x) + incv] = threadA.e[0];
      vec_expon[iter += incv] = threadA.e[1];
      vec_expon[iter += incv] = threadB.e[0];
      vec_expon[iter += incv] = threadB.e[1];
    }
  }
  else {
    __shared__ typename cub::BlockReduce<u64x3, BLOCK_THREADS>::TempStorage temp_reduce;
    cub::BlockReduce<u64x3, BLOCK_THREADS> block_reduce(temp_reduce);
    threadA = block_reduce.Reduce(threadA, u64_add());

    if (threadIdx.x == 0) {
      acc_normalize(threadA.e[2], threadA.e[1], threadA.e[0]);
      vec_expon[iter = int64_t(blockIdx.x) + incv] = threadA.e[0];
      vec_expon[iter += incv] = threadA.e[1];
    }
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  vector_sum_kernel<const double* __restrict__, 0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  vector_sum_kernel<const float* __restrict__, 0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  vector_sum_kernel<const double* __restrict__, 1, block_threads> <<< N, block_threads, 0, stream >>> (2 * M, (double*)A, 2 * lda, lo, hi, umax, vec_expon, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, uint64_t* vec_expon, int32_t incv) {
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  vector_sum_kernel<const float* __restrict__, 1, block_threads> <<< N, block_threads, 0, stream >>> (2 * M, (float*)A, 2 * lda, lo, hi, umax, vec_expon, incv);
}
