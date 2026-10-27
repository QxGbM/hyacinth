
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr uint32_t i31 = std::numeric_limits<uint32_t>::max() >> 1;
template <int32_t ORDER> static __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

static __device__ __forceinline__ uint3 round_i95(const double& x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogb(x), expon - 62, 0));
  uint64_t i = uint64_t(llrint(scalbn(x, expon - int32_t(e)))), q0 = i << e;
  return uint3({ uint32_t(q0), uint32_t(q0 >> 32) & i31, ((-uint32_t(i >> 63)) << (uint32_t(1) + e)) | uint32_t(i >> (uint32_t(63) - e)) });
}

static __device__ __forceinline__ uint3 round_i95(const float& x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogbf(x), expon - 62, 0));
  uint64_t i = uint64_t(llrintf(scalbnf(x, expon - int32_t(e)))), q0 = i << e;
  return uint3({ uint32_t(q0), uint32_t(q0 >> 32) & i31, ((-uint32_t(i >> 63)) << (uint32_t(1) + e)) | uint32_t(i >> (uint32_t(63) - e)) });
}

static __device__ __forceinline__ uint3 round_i95(const __half& x, int32_t expon) {
  return round_i95(__half2float(x), expon);
}

template <int32_t x> static __device__ __forceinline__ int8_t* remainder(const uint3& rl, int8_t* A, int64_t strideA) {
  constexpr uint32_t MO = U8CRT::mo[x], R32 = U8CRT::rem_e32[x], R63 = U8CRT::rem_e63[x], minus_MO = -MO;
  uint32_t r = device::barrett_reduc<MO>(device::mulx_reduc<0, uint32_t(1), MO>(rl.x) + device::mulx_reduc<0, R32, MO>(rl.y) + device::mulx_reduc<1, R63, MO>(rl.z));
  *A = int8_t(r + ((-uint32_t(127u < r)) & minus_MO)); return &A[strideA];
}

template <int32_t ORDER> static __device__ __forceinline__ void quantize_i8(const uint3& rl, int8_t* A, int64_t strideA) {
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
  if constexpr(23 < ORDER) { A = remainder<23>(rl, A, strideA); }
}

template <int32_t ORDER, class matrix_t>
__global__ void quantize_crt_real_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  int32_t i = (int32_t(blockIdx.x) << 8) + int32_t(threadIdx.x), expon = vexp[blockIdx.y]; A = &A[int64_t(blockIdx.y) * lda]; B = &B[int64_t(blockIdx.y) * ldb];
  if (i < M) { if (expon == int_max) { write_zeros<ORDER>(&B[i], strideB); } else { quantize_i8<ORDER>(round_i95(A[i], expon), &B[i], strideB); }}
};

template <class matrix_t> static inline void quantize_crt_real_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 256; int64_t lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
  dim3 gridx(uint32_t(M + 255) >> 8, uint32_t(N));
  switch (orderA) {
    case 2: quantize_crt_real_kernel<2> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 3: quantize_crt_real_kernel<3> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 4: quantize_crt_real_kernel<4> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 5: quantize_crt_real_kernel<5> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 6: quantize_crt_real_kernel<6> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 7: quantize_crt_real_kernel<7> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 8: quantize_crt_real_kernel<8> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 9: quantize_crt_real_kernel<9> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 10: quantize_crt_real_kernel<10> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 11: quantize_crt_real_kernel<11> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 12: quantize_crt_real_kernel<12> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 13: quantize_crt_real_kernel<13> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 14: quantize_crt_real_kernel<14> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 15: quantize_crt_real_kernel<15> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 16: quantize_crt_real_kernel<16> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 17: quantize_crt_real_kernel<17> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 18: quantize_crt_real_kernel<18> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 19: quantize_crt_real_kernel<19> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 20: quantize_crt_real_kernel<20> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 21: quantize_crt_real_kernel<21> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 22: quantize_crt_real_kernel<22> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 23: quantize_crt_real_kernel<23> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 24: quantize_crt_real_kernel<24> <<< gridx, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_crt_real_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_crt_real_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_crt_real_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

};
