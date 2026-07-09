
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct u64_add {__device__ __forceinline__ ulonglong3 operator()(ulonglong3 a, ulonglong3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }};

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, uint64_t lo, uint32_t hi, ulonglong3& acc) {
  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  constexpr uint32_t i31 = 0x7fffffff;

  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += (uint64_t(q) << expon) & i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  acc.z += hi; acc.y += uint32_t(lo >> 32) & i31; acc.x += uint32_t(lo);
}

template <int32_t ORDER, int32_t COMPLEX, int32_t BLOCK_THREADS, class real_t>
__global__ void vector_sum_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax_m62, const int32_t* __restrict__ vec_expon, uint64_t* __restrict__ vec_sum, int64_t incv) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  int64_t iter = int64_t(blockIdx.x) * lda, iter_end = iter + M;
  int32_t expon = umax_m62 - vec_expon[blockIdx.x];
  ulonglong3 threadA = ulonglong3();

  for (iter += int64_t(threadIdx.x); iter < iter_end; iter += inci)
    accumulate(A[iter], expon, lo, hi, threadA);
  iter = int64_t(blockIdx.x);

  if constexpr(COMPLEX) {
    __shared__ typename cub::BlockReduce<ulonglong3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    ulonglong3 threadB = threadA;
    if (int32_t(threadIdx.x) & 1) threadA = ulonglong3(); else threadB = ulonglong3();
    threadA = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadA, u64_add());
    threadB = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadB, u64_add());

    if (threadIdx.x == 0) {
      uint64_t c[ORDER]; c[0] = threadA.x;
      if constexpr(1 < ORDER) c[1] = threadA.z;
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadA.y, uint32_t(32));

      #pragma unroll
      for (int32_t limb = 0; limb < ORDER; ++limb)
      { vec_sum[iter] = c[limb]; iter += incv; }

      c[0] = threadB.x;
      if constexpr(1 < ORDER) c[1] = threadB.z;
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadB.y, uint32_t(32));

      #pragma unroll
      for (int32_t limb = 0; limb < ORDER; ++limb)
      { vec_sum[iter] = c[limb]; iter += incv; }
    }
  }
  else {
    __shared__ typename cub::BlockReduce<ulonglong3, BLOCK_THREADS>::TempStorage temp_reduce;
    threadA = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce).Reduce(threadA, u64_add());

    if (threadIdx.x == 0) {
      uint64_t c[ORDER]; c[0] = threadA.x;
      if constexpr(1 < ORDER) c[1] = threadA.z;
      if constexpr(2 < ORDER) c[2] = uint64_t(0);
      device::int8::add_shifted(c, threadA.y, uint32_t(32));

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
  umax = umax - 62;
  switch(order) {
    case 1: vector_sum_kernel<1, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); return;
    case 2: vector_sum_kernel<2, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); return;
    case 3: vector_sum_kernel<3, COMPLEX, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda, lo, hi, umax, vec_expon, vec_sum, incv); return;
    default: return;
  }
}

namespace internal::int8 {

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<0>(stream, int64_t(M), N, A, int64_t(lda), umax, vec_expon, order, vec_sum, incv);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<0>(stream, int64_t(M), N, A, int64_t(lda), umax, vec_expon, order, vec_sum, incv);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<0>(stream, int64_t(M), N, A, int64_t(lda), umax, vec_expon, order, vec_sum, incv);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<1>(stream, int64_t(M) << 1, N, (const double*)A, int64_t(lda) << 1, umax, vec_expon, order, vec_sum, incv);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<1>(stream, int64_t(M) << 1, N, (const float*)A, int64_t(lda) << 1, umax, vec_expon, order, vec_sum, incv);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vec_expon, int32_t order, uint64_t* vec_sum, int64_t incv) {
    vsum_dispatcher<1>(stream, int64_t(M) << 1, N, (const __half*)A, int64_t(lda) << 1, umax, vec_expon, order, vec_sum, incv);
  }

}
