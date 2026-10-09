
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>
#include <cub/cub.cuh>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr uint32_t i30 = std::numeric_limits<uint32_t>::max() >> 2, i31 = std::numeric_limits<uint32_t>::max() >> 1;
constexpr uint64_t i62 = std::numeric_limits<uint64_t>::max() >> 2, i63 = std::numeric_limits<uint64_t>::max() >> 1;
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

__device__ __forceinline__ uint3 round_i95(const uchar2& sft, const double& x, int32_t expon, ulonglong2& c) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogb(x), expon - 62, 0));
  uint64_t i = uint64_t(llrint(scalbn(x, expon - int32_t(e))));
  uint64_t q0 = ((uint64_t(1) << sft.x) & i63) + ((i << e) & i62); c.x += q0;
  uint32_t q1 = ((uint32_t(1) << sft.y) & i31) + (((-uint32_t(i >> 63)) << (uint32_t(2) + e)) | uint32_t(i >> (uint32_t(62) - e))); c.y += q1 + uint32_t(c.x >> 62); c.x &= i62;
  return uint3({ uint32_t(q0), uint32_t(q0 >> 32) & i30, q1 + uint32_t(q0 >> 62) });
}

__device__ __forceinline__ uint3 round_i95(const uchar2& sft, const float& x, int32_t expon, ulonglong2& c) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogbf(x), expon - 62, 0));
  uint64_t i = uint64_t(llrintf(scalbnf(x, expon - int32_t(e))));
  uint64_t q0 = ((uint64_t(1) << sft.x) & i63) + ((i << e) & i62); c.x += q0;
  uint32_t q1 = ((uint32_t(1) << sft.y) & i31) + (((-uint32_t(i >> 63)) << (uint32_t(2) + e)) | uint32_t(i >> (uint32_t(62) - e))); c.y += q1 + uint32_t(c.x >> 62); c.x &= i62;
  return uint3({ uint32_t(q0), uint32_t(q0 >> 32) & i30, q1 + uint32_t(q0 >> 62) });
}

__device__ __forceinline__ uint3 round_i95(const uchar2& sft, const __half& x, int32_t expon, ulonglong2& c) {
  return round_i95(sft, __half2float(x), expon, c);
}

template <int32_t x> __device__ __forceinline__ int8_t* remainder(const uint3& rl, int8_t* A, int64_t strideA) {
  constexpr uint32_t MO = U8CRT::mo[x], R32 = U8CRT::rem_e32[x], R62 = U8CRT::rem_e62[x], minus_MO = -MO;
  uint32_t r = device::barrett_reduc<MO>(device::mulx_reduc<uint32_t(1), MO>(rl.x) + device::mulx_reduc<R32, MO>(rl.y) + device::mulx_reduc<R62, MO>(rl.z));
  *A = int8_t(r + ((-uint32_t(127u < r)) & minus_MO)); return &A[strideA];
}

template <int32_t x> __device__ __forceinline__ int8_t* remainder(const uint3& rl, const uint3& im, int8_t* A, int64_t strideA) {
  constexpr uint32_t MO = U8CRT::mo[x], R32 = U8CRT::rem_e32[x], R62 = U8CRT::rem_e62[x], minus_MO = -MO;
  uint32_t r = device::barrett_reduc<MO>(device::mulx_reduc<uint32_t(1), MO>(rl.x) + device::mulx_reduc<R32, MO>(rl.y) + device::mulx_reduc<R62, MO>(rl.z));
  uint32_t i = device::barrett_reduc<MO>(device::mulx_reduc<uint32_t(1), MO>(im.x) + device::mulx_reduc<R32, MO>(im.y) + device::mulx_reduc<R62, MO>(im.z));
  uint32_t s = r + i; s = __viaddmin_u32(minus_MO, s, s);
  *A = int8_t(s + ((-uint32_t(127u < s)) & minus_MO)); *(A += strideA) = int8_t(r + ((-uint32_t(127u < r)) & minus_MO)); *(A += strideA) = int8_t(i + ((-uint32_t(127u < i)) & minus_MO));
  return &A[strideA];
}

template <int32_t ORDER> __device__ __forceinline__ void quantize_i8(const uint3& rl, int8_t* A, int64_t strideA) {
  if constexpr(0 < ORDER) { A = remainder<0>(rl, A, strideA); } else { return A; }
  if constexpr(1 < ORDER) { A = remainder<1>(rl, A, strideA); }
  if constexpr(2 < ORDER) { A = remainder<2>(rl, A, strideA); }
  if constexpr(3 < ORDER) { A = remainder<3>(rl, A, strideA); }
  if constexpr(4 < ORDER) { A = remainder<4>(rl, A, strideA); }
  if constexpr(5 < ORDER) { A = remainder<5>(rl, A, strideA); }
  if constexpr(6 < ORDER) { A = remainder<6>(rl, A, strideA); }
  if constexpr(7 < ORDER) { A = remainder<7>(rl, A, strideA); }
  if constexpr(8 < ORDER) { A = remainder<8>(rl, A, strideA); }
  if constexpr(9 < ORDER) { A = remainder<9>(rl, A, strideA); }
  if constexpr(10 < ORDER) { A = remainder<10>(rl, A, strideA); }
  if constexpr(11 < ORDER) { A = remainder<11>(rl, A, strideA); }
  if constexpr(12 < ORDER) { A = remainder<12>(rl, A, strideA); }
  if constexpr(13 < ORDER) { A = remainder<13>(rl, A, strideA); }
  if constexpr(14 < ORDER) { A = remainder<14>(rl, A, strideA); }
  if constexpr(15 < ORDER) { A = remainder<15>(rl, A, strideA); }
  if constexpr(16 < ORDER) { A = remainder<16>(rl, A, strideA); }
  if constexpr(17 < ORDER) { A = remainder<17>(rl, A, strideA); }
  if constexpr(18 < ORDER) { A = remainder<18>(rl, A, strideA); }
  if constexpr(19 < ORDER) { A = remainder<19>(rl, A, strideA); }
  if constexpr(20 < ORDER) { A = remainder<20>(rl, A, strideA); }
  if constexpr(21 < ORDER) { A = remainder<21>(rl, A, strideA); }
  if constexpr(22 < ORDER) { A = remainder<22>(rl, A, strideA); }
}

template <int32_t ORDER> __device__ __forceinline__ void quantize_i8(const uint3& rl, const uint3& im, int8_t* A, int64_t strideA) {
  if constexpr(0 < ORDER) { A = remainder<0>(rl, im, A, strideA); } else { return A; }
  if constexpr(1 < ORDER) { A = remainder<1>(rl, im, A, strideA); }
  if constexpr(2 < ORDER) { A = remainder<2>(rl, im, A, strideA); }
  if constexpr(3 < ORDER) { A = remainder<3>(rl, im, A, strideA); }
  if constexpr(4 < ORDER) { A = remainder<4>(rl, im, A, strideA); }
  if constexpr(5 < ORDER) { A = remainder<5>(rl, im, A, strideA); }
  if constexpr(6 < ORDER) { A = remainder<6>(rl, im, A, strideA); }
  if constexpr(7 < ORDER) { A = remainder<7>(rl, im, A, strideA); }
  if constexpr(8 < ORDER) { A = remainder<8>(rl, im, A, strideA); }
  if constexpr(9 < ORDER) { A = remainder<9>(rl, im, A, strideA); }
  if constexpr(10 < ORDER) { A = remainder<10>(rl, im, A, strideA); }
  if constexpr(11 < ORDER) { A = remainder<11>(rl, im, A, strideA); }
  if constexpr(12 < ORDER) { A = remainder<12>(rl, im, A, strideA); }
  if constexpr(13 < ORDER) { A = remainder<13>(rl, im, A, strideA); }
  if constexpr(14 < ORDER) { A = remainder<14>(rl, im, A, strideA); }
  if constexpr(15 < ORDER) { A = remainder<15>(rl, im, A, strideA); }
  if constexpr(16 < ORDER) { A = remainder<16>(rl, im, A, strideA); }
  if constexpr(17 < ORDER) { A = remainder<17>(rl, im, A, strideA); }
  if constexpr(18 < ORDER) { A = remainder<18>(rl, im, A, strideA); }
  if constexpr(19 < ORDER) { A = remainder<19>(rl, im, A, strideA); }
  if constexpr(20 < ORDER) { A = remainder<20>(rl, im, A, strideA); }
  if constexpr(21 < ORDER) { A = remainder<21>(rl, im, A, strideA); }
  if constexpr(22 < ORDER) { A = remainder<22>(rl, im, A, strideA); }
}

struct u64_add { __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b) { a.x += b.x; a.y += b.y + (a.x >> 62); a.x &= i62; return a; }};

template <int32_t ORDER, int32_t BLOCK_THREADS, class matrix_t, class sum_t>
__global__ void quantize_crt_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, const uchar2 init, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB, sum_t* __restrict__ vsum) {
  __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage rl_reduce;
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb]; vsum = &vsum[blockIdx.x];
  if (expon == int_max) {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { if constexpr(Complex) { write_zeros<ORDER * 3>(&B[i], strideB); } else { write_zeros<ORDER>(&B[i], strideB); }}
    if (int32_t(threadIdx.x) == 0) { *vsum = sum_t(); }
  } else if constexpr(Complex) {
    ulonglong2 threadR = make_ulonglong2(0llu, 0llu), threadI = make_ulonglong2(0llu, 0llu); u64_add acc;
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { matrix_t A_i = A[i]; quantize_i8<ORDER>(round_i95(init, A_i.x, expon, threadR), round_i95(init, A_i.y, expon, threadI), &B[i], strideB); }

    __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage im_reduce;
    threadR = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(rl_reduce).Reduce(threadR, acc);
    threadI = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(im_reduce).Reduce(threadI, acc);
    if (int32_t(threadIdx.x) == 0) { *vsum = make_ulonglong4_32a(threadR.x, threadR.y, threadI.x, threadI.y); }
  } else {
    ulonglong2 threadR = make_ulonglong2(0llu, 0llu); u64_add acc;
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { quantize_i8<ORDER>(round_i95(init, A[i], expon, threadR), &B[i], strideB); }

    threadR = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(rl_reduce).Reduce(threadR, acc);
    if (int32_t(threadIdx.x) == 0) { *vsum = threadR; }
  }
};

template <class matrix_t, class sum_t>
inline void quantize_crt_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, sum_t* vsum) {
  constexpr int32_t block_threads = 256; int64_t lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
  uchar2 init = corr < 63u ? make_uchar2(uint8_t(corr), uint8_t(31)) : make_uchar2(uint8_t(63), uint8_t(corr - 63u));

  switch (orderA) {
    case 2: quantize_crt_kernel<2, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 3: quantize_crt_kernel<3, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 4: quantize_crt_kernel<4, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 5: quantize_crt_kernel<5, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 6: quantize_crt_kernel<6, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 7: quantize_crt_kernel<7, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 8: quantize_crt_kernel<8, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 9: quantize_crt_kernel<9, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 10: quantize_crt_kernel<10, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 11: quantize_crt_kernel<11, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 12: quantize_crt_kernel<12, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 13: quantize_crt_kernel<13, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 14: quantize_crt_kernel<14, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 15: quantize_crt_kernel<15, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 16: quantize_crt_kernel<16, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 17: quantize_crt_kernel<17, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 18: quantize_crt_kernel<18, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 19: quantize_crt_kernel<19, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 20: quantize_crt_kernel<20, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 21: quantize_crt_kernel<21, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 22: quantize_crt_kernel<22, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    case 23: quantize_crt_kernel<23, block_threads> <<< N, block_threads, 0, stream >>> (M, A, lda64, init, vexp, B, ldb64, strideB, vsum); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, vsum); }

};
