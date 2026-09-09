
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>
#include <cub/cub.cuh>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
template <int32_t ORDER> __device__ __forceinline__ void write_zeros(int8_t* A, int64_t strideA) {
  constexpr int8_t zero = int8_t(0);
  if constexpr(0 < ORDER) { *A = zero; }
  #pragma unroll
  for (int32_t i = 1; i < ORDER; ++i) { *(A += strideA) = zero; }
}

template <int32_t ORDER> __device__ __forceinline__ int8_t* quantize_i8(uint64_t lo, uint32_t hi, int8_t* A, int64_t strideA) {
  uint32_t lo_32 = uint32_t(lo), mi = uint32_t(lo >> 32);
  using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
  if constexpr(0 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[0]) | (uint64_t(mo[1]) << 16) | (uint64_t(mo[2]) << 32) | (uint64_t(mo[3]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[0]) | (uint64_t(rem_e32[1]) << 16) | (uint64_t(rem_e32[2]) << 32) | (uint64_t(rem_e32[3]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[0]) | (uint64_t(rem_e63[1]) << 16) | (uint64_t(rem_e63[2]) << 32) | (uint64_t(rem_e63[3]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *A = int8_t(code);
    if constexpr(1 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(2 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(3 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  } else { return A; }

  if constexpr(4 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[4]) | (uint64_t(mo[5]) << 16) | (uint64_t(mo[6]) << 32) | (uint64_t(mo[7]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[4]) | (uint64_t(rem_e32[5]) << 16) | (uint64_t(rem_e32[6]) << 32) | (uint64_t(rem_e32[7]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[4]) | (uint64_t(rem_e63[5]) << 16) | (uint64_t(rem_e63[6]) << 32) | (uint64_t(rem_e63[7]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *(A += strideA) = int8_t(code);
    if constexpr(5 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(6 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(7 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  if constexpr(8 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[8]) | (uint64_t(mo[9]) << 16) | (uint64_t(mo[10]) << 32) | (uint64_t(mo[11]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[8]) | (uint64_t(rem_e32[9]) << 16) | (uint64_t(rem_e32[10]) << 32) | (uint64_t(rem_e32[11]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[8]) | (uint64_t(rem_e63[9]) << 16) | (uint64_t(rem_e63[10]) << 32) | (uint64_t(rem_e63[11]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *(A += strideA) = int8_t(code);
    if constexpr(9 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(10 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(11 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  if constexpr(12 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[12]) | (uint64_t(mo[13]) << 16) | (uint64_t(mo[14]) << 32) | (uint64_t(mo[15]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[12]) | (uint64_t(rem_e32[13]) << 16) | (uint64_t(rem_e32[14]) << 32) | (uint64_t(rem_e32[15]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[12]) | (uint64_t(rem_e63[13]) << 16) | (uint64_t(rem_e63[14]) << 32) | (uint64_t(rem_e63[15]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *(A += strideA) = int8_t(code);
    if constexpr(13 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(14 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(15 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  if constexpr(16 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[16]) | (uint64_t(mo[17]) << 16) | (uint64_t(mo[18]) << 32) | (uint64_t(mo[19]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[16]) | (uint64_t(rem_e32[17]) << 16) | (uint64_t(rem_e32[18]) << 32) | (uint64_t(rem_e32[19]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[16]) | (uint64_t(rem_e63[17]) << 16) | (uint64_t(rem_e63[18]) << 32) | (uint64_t(rem_e63[19]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *(A += strideA) = int8_t(code);
    if constexpr(17 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(18 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(19 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  if constexpr(20 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[20]) | (uint64_t(mo[21]) << 16) | (uint64_t(mo[22]) << 32) | (uint64_t(mo[23]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[20]) | (uint64_t(rem_e32[21]) << 16) | (uint64_t(rem_e32[22]) << 32) | (uint64_t(rem_e32[23]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[20]) | (uint64_t(rem_e63[21]) << 16) | (uint64_t(rem_e63[22]) << 32) | (uint64_t(rem_e63[23]) << 48);
    uint32_t code = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    *(A += strideA) = int8_t(code);
    if constexpr(21 < ORDER) { *(A += strideA) = int8_t(code >> 8); }
    if constexpr(22 < ORDER) { *(A += strideA) = int8_t(code >> 16); }
    if constexpr(23 < ORDER) { *(A += strideA) = int8_t(code >> 24); }
  }

  return &A[strideA];
}

struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b)
  { a.x += b.x; a.y += b.y + (a.x >> 63); a.x &= 0x7fffffffffffffffllu; return a; }
  __device__ __forceinline__ ulonglong4_32a operator()(ulonglong4_32a a, ulonglong4_32a b)
  { a.x += b.x; a.z += b.z; a.y += b.y + (a.x >> 63); a.w += b.w + (a.z >> 63); a.x &= 0x7fffffffffffffffllu; a.z &= 0x7fffffffffffffffllu; return a; }
};

template <int32_t beta>
__device__ __forceinline__ void conv_acc(ulonglong2 acc, int64_t M, uint32_t corr, ulonglong2* out) {
  if constexpr (beta) { acc = u64_add().operator()(acc, *out); }
  uint64_t a[2]{ uint64_t(acc.x), uint64_t(acc.y) }; device::int8::add_shifted(a, M, corr);
  *out = make_ulonglong2(a[0], a[1]);
}

template <int32_t beta>
__device__ __forceinline__ void conv_acc(ulonglong4_32a acc, int64_t M, uint32_t corr, ulonglong4_32a* out) {
  if constexpr (beta) { acc = u64_add().operator()(acc, *out); }
  uint64_t r[2]{ uint64_t(acc.x), uint64_t(acc.y) }, i[2]{ uint64_t(acc.z), uint64_t(acc.w) };
  device::int8::add_shifted(r, M, corr); device::int8::add_shifted(i, M, corr);
  *out = make_ulonglong4_32a(r[0], r[1], i[0], i[1]);
}

template <int32_t ORDER, int32_t beta, int32_t BLOCK_THREADS, class matrix_t, class sum_t>
__global__ void quantize_crt_kernel(int64_t M, const matrix_t* __restrict__ A, int64_t lda, uint32_t corr, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB, sum_t* __restrict__ vsum) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  constexpr int64_t BLOCK_THREADS_64 = int64_t(BLOCK_THREADS);
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb]; vsum = &vsum[blockIdx.x];
  if (expon == int_max) {
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64)
    { if constexpr(Complex) { write_zeros<ORDER * 3>(&B[i], strideB); } else { write_zeros<ORDER>(&B[i], strideB); }}

    if (int32_t(threadIdx.x) == 0)
    { if constexpr(!beta) { *vsum = sum_t(); }}
  } else if constexpr(Complex) {
    __shared__ ulonglong2 rl[BLOCK_THREADS], im[BLOCK_THREADS]; u64_add acc;
    rl[threadIdx.x] = im[threadIdx.x] = make_ulonglong2(0llu, 0llu);
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}, A_im[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i.x, expon, e); device::int8::add_shifted(A_rl, q_rl, e); rl[threadIdx.x] = acc(rl[threadIdx.x], make_ulonglong2(A_rl[0], A_rl[1]));
      int64_t q_im = device::int8::round_i64(A_i.y, expon, e); device::int8::add_shifted(A_im, q_im, e); im[threadIdx.x] = acc(im[threadIdx.x], make_ulonglong2(A_im[0], A_im[1]));
      device::int8::add_shifted(A_rl, int64_t(1), corr); device::int8::add_shifted(A_im, int64_t(1), corr);

      int8_t* B_i = quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), &B[i], strideB);
      A_rl[0] += A_im[0]; A_rl[1] += A_im[1] + (A_rl[0] >> 63); A_rl[0] &= 0x7fffffffffffffffllu;
      quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), quantize_i8<ORDER>(A_im[0], uint32_t(A_im[1]), B_i, strideB), strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong4_32a, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong4_32a threadA = cub::BlockReduce<ulonglong4_32a, BLOCK_THREADS>(temp_reduce).Reduce(make_ulonglong4_32a(rl[threadIdx.x].x, rl[threadIdx.x].y, im[threadIdx.x].x, im[threadIdx.x].y), acc);
    if (int32_t(threadIdx.x) == 0) { conv_acc<beta>(threadA, M, corr, vsum); }
  } else {
    __shared__ ulonglong2 rl[BLOCK_THREADS]; u64_add acc;
    rl[threadIdx.x] = make_ulonglong2(0llu, 0llu);
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i, expon, e); device::int8::add_shifted(A_rl, q_rl, e); rl[threadIdx.x] = acc(rl[threadIdx.x], make_ulonglong2(A_rl[0], A_rl[1]));
      device::int8::add_shifted(A_rl, int64_t(1), corr);
      quantize_i8<ORDER>(A_rl[0], uint32_t(A_rl[1]), &B[i], strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong2 threadA = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(temp_reduce).Reduce(rl[threadIdx.x], acc);
    if (int32_t(threadIdx.x) == 0) { conv_acc<beta>(threadA, M, corr, vsum); }
  }
};

template <class matrix_t, class sum_t>
inline void quantize_crt_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, sum_t* vsum) {
  constexpr int32_t block_threads = 512;
  int64_t M64 = int64_t(M), lda64 = int64_t(lda), ldb64 = int64_t(ldb), strideB = int64_t(N) * ldb64;

  if (beta) switch (orderA) {
    case 2: quantize_crt_kernel<2, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 3: quantize_crt_kernel<3, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 4: quantize_crt_kernel<4, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 5: quantize_crt_kernel<5, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 6: quantize_crt_kernel<6, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 7: quantize_crt_kernel<7, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 8: quantize_crt_kernel<8, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 9: quantize_crt_kernel<9, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 10: quantize_crt_kernel<10, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 11: quantize_crt_kernel<11, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 12: quantize_crt_kernel<12, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 13: quantize_crt_kernel<13, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 14: quantize_crt_kernel<14, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 15: quantize_crt_kernel<15, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 16: quantize_crt_kernel<16, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 17: quantize_crt_kernel<17, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 18: quantize_crt_kernel<18, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 19: quantize_crt_kernel<19, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 20: quantize_crt_kernel<20, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 21: quantize_crt_kernel<21, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 22: quantize_crt_kernel<22, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 23: quantize_crt_kernel<23, 1, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    default: return;
  } else switch (orderA) {
    case 2: quantize_crt_kernel<2, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 3: quantize_crt_kernel<3, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 4: quantize_crt_kernel<4, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 5: quantize_crt_kernel<5, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 6: quantize_crt_kernel<6, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 7: quantize_crt_kernel<7, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 8: quantize_crt_kernel<8, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 9: quantize_crt_kernel<9, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 10: quantize_crt_kernel<10, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 11: quantize_crt_kernel<11, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 12: quantize_crt_kernel<12, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 13: quantize_crt_kernel<13, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 14: quantize_crt_kernel<14, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 15: quantize_crt_kernel<15, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 16: quantize_crt_kernel<16, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 17: quantize_crt_kernel<17, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 18: quantize_crt_kernel<18, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 19: quantize_crt_kernel<19, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 20: quantize_crt_kernel<20, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 21: quantize_crt_kernel<21, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 22: quantize_crt_kernel<22, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    case 23: quantize_crt_kernel<23, 0, block_threads> <<< N, block_threads, 0, stream >>> (M64, A, lda64, corr, vexp, B, ldb64, strideB, vsum); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong2* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, ulonglong4_32a* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr += uint32_t(corr ? -1 : 0), vexp, B, ldb, beta, vsum); }

}
