
#include <internal.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr uint64_t i63 = 0x7fffffffffffffffllu;
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

struct __align__(16) lint95_t { uint64_t x; uint32_t y; };
__device__ __forceinline__ lint95_t round_i95(double x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogb(x), expon - 62, 0));
  uint64_t i = uint64_t(llrint(scalbn(x, expon - int32_t(e))));
  uint64_t m0 = -uint64_t(e < uint32_t(63));
  uint32_t m1 = ~uint32_t(m0), rem = (e - (uint32_t(63) & m1));
  uint64_t q0 = i << rem;
  uint32_t sign = -uint32_t(i >> 63), q1 = (sign << (uint32_t(1) + rem)) | uint32_t(i >> (uint32_t(63) - rem));
  return lint95_t({ q0 & m0 & i63, (q1 & uint32_t(m0)) | (uint32_t(q0) & m1) });
}

__device__ __forceinline__ lint95_t round_i95(float x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogbf(x), expon - 62, 0));
  uint64_t i = uint64_t(llrintf(scalbnf(x, expon - int32_t(e))));
  uint64_t m0 = -uint64_t(e < uint32_t(63));
  uint32_t m1 = ~uint32_t(m0), rem = (e - (uint32_t(63) & m1));
  uint64_t q0 = i << rem;
  uint32_t sign = -uint32_t(i >> 63), q1 = (sign << (uint32_t(1) + rem)) | uint32_t(i >> (uint32_t(63) - rem));
  return lint95_t({ q0 & m0 & i63, (q1 & uint32_t(m0)) | (uint32_t(q0) & m1) });
}

__device__ __forceinline__ lint95_t round_i95(__half x, int32_t expon) {
  return round_i95(__half2float(x), expon);
}

template <int32_t ORDER> __device__ __forceinline__ int8_t* quantize_i8(lint95_t i, int8_t* A, int64_t strideA) {
  uint32_t a;
  if constexpr(0 < ORDER) { a = uint8_t(i.x); *A = int8_t(a); } else { return A; }
  if constexpr(1 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 8); *(A += strideA) = int8_t(a); }
  if constexpr(2 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 16); *(A += strideA) = int8_t(a); }
  if constexpr(3 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 24); *(A += strideA) = int8_t(a); }
  if constexpr(4 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 32); *(A += strideA) = int8_t(a); }
  if constexpr(5 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 40); *(A += strideA) = int8_t(a); }
  if constexpr(6 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.x >> 48); *(A += strideA) = int8_t(a); }
  if constexpr(7 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t((i.y << 7) | (i.x >> 56)); *(A += strideA) = int8_t(a); }
  if constexpr(8 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.y >> 1); *(A += strideA) = int8_t(a); }
  if constexpr(9 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.y >> 9); *(A += strideA) = int8_t(a); }
  if constexpr(10 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.y >> 17); *(A += strideA) = int8_t(a); }
  if constexpr(11 < ORDER) { a = (a >> 8) + ((a >> 7) & uint32_t(1)) + uint8_t(i.y >> 25); *(A += strideA) = int8_t(a); }
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
      matrix_t A_i = A[i]; lint95_t A_rl = round_i95(A_i.x, expon), A_im = round_i95(A_i.y, expon);
      int8_t* B_i = quantize_i8<ORDER>(A_rl, &B[i], strideB);
      A_rl.x += A_im.x; A_rl.y += A_im.y + uint32_t(A_rl.x >> 63); A_rl.x &= i63;
      quantize_i8<ORDER>(A_rl, quantize_i8<ORDER>(A_im, B_i, strideB), strideB);
    }
  } else {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += int32_t(blockDim.x))
    { quantize_i8<ORDER>(round_i95(A[i], expon), &B[i], strideB); }
  }
};

template <class matrix_t>
inline void quantize_limbs_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, const int32_t* vexp, int8_t* B, int32_t ldb) {
  constexpr int32_t block_threads = 512;
  int64_t lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
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

}
