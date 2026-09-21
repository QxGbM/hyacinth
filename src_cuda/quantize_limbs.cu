
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

template <int32_t ORDER> __device__ __forceinline__ int8_t* quantize_i8(uint64_t lo, uint32_t hi, int8_t* A, int64_t strideA) {
  uint32_t c = 0; constexpr uint32_t u31 = uint32_t(0x80000000);
  if constexpr(0 < ORDER) {
    uint32_t code = device::int8::conv_u8i8(uint32_t(lo), c);
    *A = int8_t(code);
    if constexpr(1 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(2 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(3 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  } else { return A; }

  if constexpr(4 < ORDER) {
    uint32_t code = device::int8::conv_u8i8(uint32_t(lo >> 32) | (hi << 31), c);
    *(A += strideA) = int8_t(code);
    if constexpr(5 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(6 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(7 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  if constexpr(8 < ORDER) {
    uint32_t code = device::int8::conv_u8i8((hi & u31) | (hi >> 1), c);
    *(A += strideA) = int8_t(code);
    if constexpr(9 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(10 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(11 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  return &A[strideA];
}

template <int32_t ORDER, class matrix_t>
__global__ void quantize_limbs_kernel(int64_t M, const matrix_t* __restrict__ A, int64_t lda, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb];
  if (expon == int_max) {
    for (int64_t i = int64_t(threadIdx.x); i < M; i += int64_t(blockDim.x))
    { if constexpr(Complex) { write_zeros<ORDER * 3>(&B[i], strideB); } else { write_zeros<ORDER>(&B[i], strideB); }}
  } else if constexpr(Complex) {
    for (int64_t i = int64_t(threadIdx.x); i < M; i += int64_t(blockDim.x)) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}, A_im[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i.x, expon, e); device::int8::add_shifted(A_rl, q_rl, e);
      int64_t q_im = device::int8::round_i64(A_i.y, expon, e); device::int8::add_shifted(A_im, q_im, e);

      int8_t* B_i = quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), &B[i], strideB);
      device::int8::add_shifted(A_rl, q_im, e);
      quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), quantize_i8<ORDER>(A_im[0], uint32_t(A_im[1]), B_i, strideB), strideB);
    }
  } else {
    for (int64_t i = int64_t(threadIdx.x); i < M; i += int64_t(blockDim.x)) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i, expon, e); device::int8::add_shifted(A_rl, q_rl, e);
      quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), &B[i], strideB);
    }
  }
};

template <class matrix_t>
inline void quantize_limbs_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 512;
  int64_t M64 = int64_t(M), lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
  switch (orderA) {
    case 1: quantize_limbs_kernel<1> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 2: quantize_limbs_kernel<2> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 3: quantize_limbs_kernel<3> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 4: quantize_limbs_kernel<4> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 5: quantize_limbs_kernel<5> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 6: quantize_limbs_kernel<6> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 7: quantize_limbs_kernel<7> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 8: quantize_limbs_kernel<8> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 9: quantize_limbs_kernel<9> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 10: quantize_limbs_kernel<10> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 11: quantize_limbs_kernel<11> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
    case 12: quantize_limbs_kernel<12> <<< N, block_threads, 0, stream >>> (M64, A, lda64, vexp, B, ldb64, strideB); return;
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

}
