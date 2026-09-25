
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>
#include <cub/cub.cuh>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
constexpr uint64_t i63 = uint64_t(std::numeric_limits<int64_t>::max());
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

struct __align__(16) lint95_t { uint64_t x; uint32_t y; };
__device__ __forceinline__ lint95_t round_i95(lint95_t a, double x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogb(x), expon - 62, 0));
  uint64_t i = uint64_t(llrint(scalbn(x, expon - int32_t(e))));
  uint64_t m0 = -uint64_t(e < uint32_t(63));
  uint32_t m1 = ~uint32_t(m0), rem = (e - (uint32_t(63) & m1));
  uint64_t q0 = i << rem;
  uint32_t sign = -uint32_t(i >> 63), q1 = (sign << (uint32_t(1) + rem)) | uint32_t(i >> (uint32_t(63) - rem));
  a.x += q0 & m0 & i63; a.y += ((q1 & uint32_t(m0)) | (uint32_t(q0) & m1)) + uint32_t(a.x >> 63); a.x &= i63;
  return a;
}

__device__ __forceinline__ lint95_t round_i95(lint95_t a, float x, int32_t expon) {
  uint32_t e = uint32_t(__viaddmax_s32(ilogbf(x), expon - 62, 0));
  uint64_t i = uint64_t(llrintf(scalbnf(x, expon - int32_t(e))));
  uint64_t m0 = -uint64_t(e < uint32_t(63));
  uint32_t m1 = ~uint32_t(m0), rem = (e - (uint32_t(63) & m1));
  uint64_t q0 = i << rem;
  uint32_t sign = -uint32_t(i >> 63), q1 = (sign << (uint32_t(1) + rem)) | uint32_t(i >> (uint32_t(63) - rem));
  a.x += q0 & m0 & i63; a.y += ((q1 & uint32_t(m0)) | (uint32_t(q0) & m1)) + uint32_t(a.x >> 63); a.x &= i63;
  return a;
}

__device__ __forceinline__ lint95_t round_i95(lint95_t a, __half x, int32_t expon) {
  return round_i95(a, __half2float(x), expon);
}

template <int32_t x> __device__ __forceinline__ int8_t remainder(uint32_t lo, uint32_t mi, uint32_t hi) {
  using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
  constexpr uint32_t MO = mo[x], R32 = rem_e32[x], R63 = rem_e63[x];
  uint32_t r = device::barrett_reduc<MO>(device::mulx_reduc<uint32_t(1), MO>(lo) + device::mulx_reduc<R32, MO>(mi) + device::mulx_reduc<R63, MO>(hi));
  uint32_t i8_mask = -uint32_t(uint32_t(127) < r);
  return int8_t(r + (i8_mask & (-MO)));
}

template <int32_t ORDER> __device__ __forceinline__ int8_t* quantize_i8(lint95_t i, int8_t* A, int64_t strideA) {
  uint32_t lo = uint32_t(i.x), mi = uint32_t(i.x >> 32);
  if constexpr(0 < ORDER) { *A = remainder<0>(lo, mi, i.y); } else { return A; }
  if constexpr(1 < ORDER) { *(A += strideA) = remainder<1>(lo, mi, i.y); }
  if constexpr(2 < ORDER) { *(A += strideA) = remainder<2>(lo, mi, i.y); }
  if constexpr(3 < ORDER) { *(A += strideA) = remainder<3>(lo, mi, i.y); }
  if constexpr(4 < ORDER) { *(A += strideA) = remainder<4>(lo, mi, i.y); }
  if constexpr(5 < ORDER) { *(A += strideA) = remainder<5>(lo, mi, i.y); }
  if constexpr(6 < ORDER) { *(A += strideA) = remainder<6>(lo, mi, i.y); }
  if constexpr(7 < ORDER) { *(A += strideA) = remainder<7>(lo, mi, i.y); }
  if constexpr(8 < ORDER) { *(A += strideA) = remainder<8>(lo, mi, i.y); }
  if constexpr(9 < ORDER) { *(A += strideA) = remainder<9>(lo, mi, i.y); }
  if constexpr(10 < ORDER) { *(A += strideA) = remainder<10>(lo, mi, i.y); }
  if constexpr(11 < ORDER) { *(A += strideA) = remainder<11>(lo, mi, i.y); }
  if constexpr(12 < ORDER) { *(A += strideA) = remainder<12>(lo, mi, i.y); }
  if constexpr(13 < ORDER) { *(A += strideA) = remainder<13>(lo, mi, i.y); }
  if constexpr(14 < ORDER) { *(A += strideA) = remainder<14>(lo, mi, i.y); }
  if constexpr(15 < ORDER) { *(A += strideA) = remainder<15>(lo, mi, i.y); }
  if constexpr(16 < ORDER) { *(A += strideA) = remainder<16>(lo, mi, i.y); }
  if constexpr(17 < ORDER) { *(A += strideA) = remainder<17>(lo, mi, i.y); }
  if constexpr(18 < ORDER) { *(A += strideA) = remainder<18>(lo, mi, i.y); }
  if constexpr(19 < ORDER) { *(A += strideA) = remainder<19>(lo, mi, i.y); }
  if constexpr(20 < ORDER) { *(A += strideA) = remainder<20>(lo, mi, i.y); }
  if constexpr(21 < ORDER) { *(A += strideA) = remainder<21>(lo, mi, i.y); }
  if constexpr(22 < ORDER) { *(A += strideA) = remainder<22>(lo, mi, i.y); }
  return &A[strideA];
}

struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b)
  { a.x += b.x; a.y += b.y + (a.x >> 63); a.x &= i63; return a; }
  __device__ __forceinline__ ulonglong4_32a operator()(ulonglong4_32a a, ulonglong4_32a b)
  { a.x += b.x; a.z += b.z; a.y += b.y + (a.x >> 63); a.w += b.w + (a.z >> 63); a.x &= i63; a.z &= i63; return a; }
};

template <int32_t ORDER, int32_t BLOCK_THREADS, class matrix_t, class sum_t>
__global__ void quantize_crt_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, lint95_t init, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB, sum_t* __restrict__ vsum) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb]; vsum = &vsum[blockIdx.x];
  if (expon == int_max) {
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { if constexpr(Complex) { write_zeros<ORDER * 3>(&B[i], strideB); } else { write_zeros<ORDER>(&B[i], strideB); }}
    if (int32_t(threadIdx.x) == 0) { *vsum = sum_t(); }
  } else if constexpr(Complex) {
    __shared__ ulonglong2 rl[BLOCK_THREADS], im[BLOCK_THREADS]; u64_add acc;
    rl[threadIdx.x] = im[threadIdx.x] = make_ulonglong2(0llu, 0llu);
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS) {
      matrix_t A_i = A[i]; lint95_t A_rl = round_i95(init, A_i.x, expon), A_im = round_i95(init, A_i.y, expon);
      rl[threadIdx.x] = acc(rl[threadIdx.x], make_ulonglong2(A_rl.x, uint64_t(A_rl.y)));
      im[threadIdx.x] = acc(im[threadIdx.x], make_ulonglong2(A_im.x, uint64_t(A_im.y)));

      int8_t* B_i = quantize_i8<ORDER>(A_rl, &B[i], strideB);
      A_rl.x += A_im.x; A_rl.y += A_im.y + uint32_t(A_rl.x >> 63); A_rl.x &= i63;
      quantize_i8<ORDER>(A_rl, quantize_i8<ORDER>(A_im, B_i, strideB), strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong4_32a, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong4_32a threadA = cub::BlockReduce<ulonglong4_32a, BLOCK_THREADS>(temp_reduce).Reduce(make_ulonglong4_32a(rl[threadIdx.x].x, rl[threadIdx.x].y, im[threadIdx.x].x, im[threadIdx.x].y), acc);
    if (int32_t(threadIdx.x) == 0) { *vsum = threadA; }
  } else {
    __shared__ ulonglong2 rl[BLOCK_THREADS]; u64_add acc;
    rl[threadIdx.x] = make_ulonglong2(0llu, 0llu);
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS) {
      lint95_t A_rl = round_i95(init, A[i], expon);
      rl[threadIdx.x] = acc(rl[threadIdx.x], make_ulonglong2(A_rl.x, uint64_t(A_rl.y)));
      quantize_i8<ORDER>(A_rl, &B[i], strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong2 threadA = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(temp_reduce).Reduce(rl[threadIdx.x], acc);
    if (int32_t(threadIdx.x) == 0) { *vsum = threadA; }
  }
};

template <class matrix_t, class sum_t>
inline void quantize_crt_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, sum_t* vsum) {
  constexpr int32_t block_threads = 512;
  int64_t lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;
  lint95_t init = corr < uint32_t(63) ? lint95_t({ 1llu << corr, 0u }) : lint95_t({ 0llu, 1u << (corr - uint32_t(63)) }); 

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

}
