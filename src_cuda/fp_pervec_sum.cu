
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

template <class real_t, int32_t ORDER, int32_t COMPLEX, int32_t BLOCK_THREADS>
__global__ void vector_sum_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax, const int32_t* __restrict__ vec_expon, uint64_t* __restrict__ vec_sum, int64_t incv) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  int64_t iter = int64_t(blockIdx.x) * lda, iter_end = iter + M;
  int32_t expon = umax - vec_expon[blockIdx.x];
  u64x3 threadA; threadA.e[0] = threadA.e[1] = threadA.e[2] = uint64_t(0);

  for (iter += int64_t(threadIdx.x); iter < iter_end; iter += inci)
    accumulate(A[iter], expon, lo, hi, threadA.e[2], threadA.e[1], threadA.e[0]);
  iter = int64_t(blockIdx.x);

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<u64x3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    u64x3 threadB = threadA;
    if (int32_t(threadIdx.x) & 1) threadA.e[0] = threadA.e[1] = threadA.e[2] = uint64_t(0);
      else threadB.e[0] = threadB.e[1] = threadB.e[2] = uint64_t(0);
    threadA = cub::BlockReduce<u64x3, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadA, u64_add());
    threadB = cub::BlockReduce<u64x3, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadB, u64_add());

    if (threadIdx.x == 0) {
      uint64_t c[ORDER]; c[0] = threadA.e[0];
      if constexpr(1 < ORDER) c[1] = threadA.e[2];
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadA.e[1], uint32_t(32));

      #pragma unroll
      for (int32_t limb = 0; limb < ORDER; ++limb)
      { vec_sum[iter] = c[limb]; iter += incv; }

      c[0] = threadB.e[0];
      if constexpr(1 < ORDER) c[1] = threadB.e[2];
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadB.e[1], uint32_t(32));

      #pragma unroll
      for (int32_t limb = 0; limb < ORDER; ++limb)
      { vec_sum[iter] = c[limb]; iter += incv; }
    }
  }
  else {
    __shared__ typename cub::BlockReduce<u64x3, BLOCK_THREADS>::TempStorage temp_reduce;
    threadA = cub::BlockReduce<u64x3, BLOCK_THREADS>(temp_reduce).Reduce(threadA, u64_add());

    if (threadIdx.x == 0) {
      uint64_t c[ORDER]; c[0] = threadA.e[0];
      if constexpr(1 < ORDER) c[1] = threadA.e[2];
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadA.e[1], uint32_t(32));

      #pragma unroll
      for (int32_t limb = 0; limb < ORDER; ++limb)
      { vec_sum[iter] = c[limb]; iter += incv; }
    }
  }
}

template <int32_t COMPLEX, class real_t>
inline void vsum_dispatcher(cudaStream_t stream, int64_t M, int32_t N, const real_t* A, int64_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
  constexpr int32_t block_threads = 512;
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  switch(order) {
    case 1: vector_sum_kernel<real_t, 1, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); break;
    case 2: vector_sum_kernel<real_t, 2, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); break;
    case 3: vector_sum_kernel<real_t, 3, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); break;
    default: break;
  }
}

void internal::int8::vsum_f64(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
  vsum_dispatcher<0>(stream, int64_t(M), N, A, int64_t(lda), umax, vec_expon, order, vec_sum, incv);
}

void internal::int8::vsum_f32(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
  vsum_dispatcher<0>(stream, int64_t(M), N, A, int64_t(lda), umax, vec_expon, order, vec_sum, incv);
}

void internal::int8::vsum_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
  vsum_dispatcher<1>(stream, int64_t(M) << 1, N, (const double*)A, int64_t(lda) << 1, umax, vec_expon, order, vec_sum, incv);
}

void internal::int8::vsum_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
  vsum_dispatcher<1>(stream, int64_t(M) << 1, N, (const float*)A, int64_t(lda) << 1, umax, vec_expon, order, vec_sum, incv);
}
