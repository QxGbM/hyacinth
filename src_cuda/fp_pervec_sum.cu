
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>

struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b) { a.x += b.x; a.y += b.y; return a; }
  __device__ __forceinline__ ulonglong3 operator()(ulonglong3 a, ulonglong3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
};

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, uint64_t lo, uint32_t, ulonglong2& acc) {
  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += uint64_t(q) << expon;
  acc.x += uint32_t(lo); acc.y += uint32_t(lo >> 32);
}

template <class real_t>
__device__ __forceinline__ void accumulate(real_t x, int32_t expon, uint64_t lo, uint32_t hi, ulonglong3& acc) {
  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  constexpr uint32_t i31 = 0x7fffffff;

  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += (uint64_t(q) << expon) & i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  acc.x += uint32_t(lo); acc.y += uint32_t(lo >> 32) & i31; acc.z += hi;
}

template <int32_t ORDER, int32_t sign, class reduc_t>
__device__ __forceinline__ uint64_t* conv_acc(reduc_t acc, uint64_t* out, int64_t stride) {
  uint64_t a[ORDER]{ acc.x };
  if constexpr(1 < ORDER) { if constexpr(std::is_same_v<reduc_t, ulonglong3>) a[1] = acc.z; else a[1] = uint64_t(0); }
  if constexpr(2 < ORDER) { a[2] = uint64_t(0); }

  device::int8::add_shifted(a, acc.y, uint32_t(32)); 
  if constexpr(sign) { *out = -a[0]; } else { *out = a[0]; }
  if constexpr(1 < ORDER && sign) { *(out += stride) = -a[1]; } else if constexpr(1 < ORDER) { *(out += stride) = a[1]; }
  if constexpr(2 < ORDER && sign) { *(out += stride) = -a[2]; } else if constexpr(2 < ORDER) { *(out += stride) = a[2]; }
  return &out[stride];
}

template <int32_t ORDER, int32_t Complex, int32_t BLOCK_THREADS, class reduc_t, class real_t>
__global__ void vector_sum_kernel(int64_t M, const real_t* __restrict__ A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax, const int32_t* __restrict__ vexp, uint64_t* __restrict__ vec_sum) {
  constexpr int64_t inci = int64_t(BLOCK_THREADS);
  int64_t iter = int64_t(blockIdx.x) * lda, iter_end = iter + M;
  int32_t expon = umax - vexp[blockIdx.x];
  reduc_t threadA = reduc_t();

  for (iter += int64_t(threadIdx.x); iter < iter_end; iter += inci)
    accumulate(A[iter], expon, lo, hi, threadA);
  iter = int64_t(blockIdx.x);

  if constexpr(Complex) {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce[2];
    reduc_t threadB = threadA;
    if (int32_t(threadIdx.x) & 1) threadA = reduc_t(); else threadB = reduc_t();
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[0]).Reduce(threadA, u64_add());
    threadB = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce[1]).Reduce(threadB, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 0>(threadB, conv_acc<ORDER, 0>(threadA, &vec_sum[blockIdx.x], int64_t(gridDim.x)), int64_t(gridDim.x)); }
  }
  else {
    __shared__ typename cub::BlockReduce<reduc_t, BLOCK_THREADS>::TempStorage temp_reduce;
    threadA = cub::BlockReduce<reduc_t, BLOCK_THREADS>(temp_reduce).Reduce(threadA, u64_add());
    if (threadIdx.x == 0) { conv_acc<ORDER, 1>(threadA, &vec_sum[blockIdx.x], int64_t(gridDim.x)); }
  }
}

template <class real_t, class matrix_t>
inline void vsum_dispatcher(cudaStream_t stream, int32_t M, int32_t N, const matrix_t* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
  constexpr int32_t block_threads = 512, Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);
  int64_t M64 = int64_t(M), lda64 = int64_t(lda); if constexpr(Complex) { M64 <<= 1; lda64 <<= 1; }
  if (umax < 63) switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong2> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    default: return;
  } else switch(order) {
    case 1: vector_sum_kernel<1, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 2: vector_sum_kernel<2, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    case 3: vector_sum_kernel<3, Complex, block_threads, ulonglong3> <<< N, block_threads, 0, stream >>> (M64, (const real_t*)A, lda64, lo, hi, umax, vexp, vec_sum); return;
    default: return;
  }
}

namespace internal::int8 {

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<double>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<float>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<__half>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<double>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<float>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t umax, const int32_t* vexp, int32_t order, uint64_t* vec_sum) {
    vsum_dispatcher<__half>(stream, M, N, A, lda, umax, vexp, order, vec_sum);
  }

}
