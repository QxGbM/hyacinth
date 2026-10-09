
#include <internal.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr uint64_t i63 = std::numeric_limits<uint64_t>::max() >> 1;
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

__device__ __forceinline__ void round_i95(const double& x, int32_t expon, uint64_t& lo, uint32_t& hi) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogb(x), expon - 62, 0));
  uint64_t i = uint64_t(llrint(scalbn(x, expon - int32_t(e))));
  lo = (i << e) & i63; hi = ((-uint32_t(i >> 63)) << (uint32_t(1) + e)) | uint32_t(i >> (uint32_t(63) - e));
}

__device__ __forceinline__ void round_i95(const float& x, int32_t expon, uint64_t& lo, uint32_t& hi) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogbf(x), expon - 62, 0));
  uint64_t i = uint64_t(llrintf(scalbnf(x, expon - int32_t(e))));
  lo = (i << e) & i63; hi = ((-uint32_t(i >> 63)) << (uint32_t(1) + e)) | uint32_t(i >> (uint32_t(63) - e));
}

__device__ __forceinline__ void round_i95(const __half& x, int32_t expon, uint64_t& lo, uint32_t& hi) {
  round_i95(__half2float(x), expon, lo, hi);
}

template <int32_t ORDER> __device__ __forceinline__ int8_t* quantize_i8(const uint64_t& lo, const uint32_t& hi, int8_t* A, int64_t strideA) {
  uint32_t a;
  if constexpr(0 < ORDER) { a = uint8_t(lo); *A = int8_t(a); } else { return A; }
  if constexpr(1 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 8); *(A += strideA) = int8_t(a); }
  if constexpr(2 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 16); *(A += strideA) = int8_t(a); }
  if constexpr(3 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 24); *(A += strideA) = int8_t(a); }
  if constexpr(4 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 32); *(A += strideA) = int8_t(a); }
  if constexpr(5 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 40); *(A += strideA) = int8_t(a); }
  if constexpr(6 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(lo >> 48); *(A += strideA) = int8_t(a); }
  if constexpr(7 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t((hi << 7) | (lo >> 56)); *(A += strideA) = int8_t(a); }
  if constexpr(8 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(hi >> 1); *(A += strideA) = int8_t(a); }
  if constexpr(9 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(hi >> 9); *(A += strideA) = int8_t(a); }
  if constexpr(10 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(hi >> 17); *(A += strideA) = int8_t(a); }
  if constexpr(11 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(hi >> 25); *(A += strideA) = int8_t(a); }
  return &A[strideA];
}

template <int32_t ORDER, class matrix_t>
__global__ void quantize_limbs_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb];
  if (expon == int_max) {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += int32_t(blockDim.x))
    { if constexpr(Complex) { write_zeros<ORDER * 3>(&B[i], strideB); } else { write_zeros<ORDER>(&B[i], strideB); }}
  } else if constexpr(Complex) {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += int32_t(blockDim.x)) {
      matrix_t A_i = A[i]; uint64_t rl_lo, im_lo; uint32_t rl_hi, im_hi;
      round_i95(A_i.x, expon, rl_lo, rl_hi); round_i95(A_i.y, expon, im_lo, im_hi); uint64_t lo = rl_lo + im_lo; uint32_t hi = rl_hi + im_hi + uint32_t(lo >> 63); lo &= i63;
      quantize_i8<ORDER>(im_lo, im_hi, quantize_i8<ORDER>(rl_lo, rl_hi, quantize_i8<ORDER>(lo, hi, &B[i], strideB), strideB), strideB);
    }
  } else {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += int32_t(blockDim.x))
    { uint64_t lo; uint32_t hi; round_i95(A[i], expon, lo, hi); quantize_i8<ORDER>(lo, hi, &B[i], strideB); }
  }
};

template <class matrix_t>
inline void quantize_limbs_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 256; int64_t lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
  switch (orderA) {
    case 1: quantize_limbs_kernel<1> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 2: quantize_limbs_kernel<2> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 3: quantize_limbs_kernel<3> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 4: quantize_limbs_kernel<4> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 5: quantize_limbs_kernel<5> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 6: quantize_limbs_kernel<6> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 7: quantize_limbs_kernel<7> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 8: quantize_limbs_kernel<8> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 9: quantize_limbs_kernel<9> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 10: quantize_limbs_kernel<10> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 11: quantize_limbs_kernel<11> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    case 12: quantize_limbs_kernel<12> <<< N, block_threads, 0, stream >>> (M, A, lda64, vexp, B, ldb64, strideB); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

  void quantize_limbs(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb)
  { quantize_limbs_dispatcher(stream, M, N, orderA, A, lda, vexp, B, ldb); }

};
