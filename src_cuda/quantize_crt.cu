
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>
#include <cub/cub.cuh>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
template <int32_t orderi8, int32_t ORDER> __device__ __forceinline__ void quantize_i8(uint64_t lo, uint32_t hi, uint32_t (&code)[ORDER]) {
  uint32_t lo_32 = uint32_t(lo), mi = uint32_t(lo >> 32);
  using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
  if constexpr(0 < orderi8 && 0 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[0]) | (uint64_t(mo[1]) << 16) | (uint64_t(mo[2]) << 32) | (uint64_t(mo[3]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[0]) | (uint64_t(rem_e32[1]) << 16) | (uint64_t(rem_e32[2]) << 32) | (uint64_t(rem_e32[3]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[0]) | (uint64_t(rem_e63[1]) << 16) | (uint64_t(rem_e63[2]) << 32) | (uint64_t(rem_e63[3]) << 48);
    code[0] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(0 < ORDER) { code[0] = uint32_t(0); }

  if constexpr(4 < orderi8 && 1 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[4]) | (uint64_t(mo[5]) << 16) | (uint64_t(mo[6]) << 32) | (uint64_t(mo[7]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[4]) | (uint64_t(rem_e32[5]) << 16) | (uint64_t(rem_e32[6]) << 32) | (uint64_t(rem_e32[7]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[4]) | (uint64_t(rem_e63[5]) << 16) | (uint64_t(rem_e63[6]) << 32) | (uint64_t(rem_e63[7]) << 48);
    code[1] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(1 < ORDER) { code[1] = uint32_t(0); }

  if constexpr(8 < orderi8 && 2 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[8]) | (uint64_t(mo[9]) << 16) | (uint64_t(mo[10]) << 32) | (uint64_t(mo[11]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[8]) | (uint64_t(rem_e32[9]) << 16) | (uint64_t(rem_e32[10]) << 32) | (uint64_t(rem_e32[11]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[8]) | (uint64_t(rem_e63[9]) << 16) | (uint64_t(rem_e63[10]) << 32) | (uint64_t(rem_e63[11]) << 48);
    code[2] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(2 < ORDER) { code[2] = uint32_t(0); }

  if constexpr(12 < orderi8 && 3 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[12]) | (uint64_t(mo[13]) << 16) | (uint64_t(mo[14]) << 32) | (uint64_t(mo[15]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[12]) | (uint64_t(rem_e32[13]) << 16) | (uint64_t(rem_e32[14]) << 32) | (uint64_t(rem_e32[15]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[12]) | (uint64_t(rem_e63[13]) << 16) | (uint64_t(rem_e63[14]) << 32) | (uint64_t(rem_e63[15]) << 48);
    code[3] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(3 < ORDER) { code[3] = uint32_t(0); }

  if constexpr(16 < orderi8 && 4 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[16]) | (uint64_t(mo[17]) << 16) | (uint64_t(mo[18]) << 32) | (uint64_t(mo[19]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[16]) | (uint64_t(rem_e32[17]) << 16) | (uint64_t(rem_e32[18]) << 32) | (uint64_t(rem_e32[19]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[16]) | (uint64_t(rem_e63[17]) << 16) | (uint64_t(rem_e63[18]) << 32) | (uint64_t(rem_e63[19]) << 48);
    code[4] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(4 < ORDER) { code[4] = uint32_t(0); }

  if constexpr(20 < orderi8 && 5 < ORDER) {
    constexpr uint64_t m = uint64_t(mo[20]) | (uint64_t(mo[21]) << 16) | (uint64_t(mo[22]) << 32) | (uint64_t(mo[23]) << 48);
    constexpr uint64_t r32 = uint64_t(rem_e32[20]) | (uint64_t(rem_e32[21]) << 16) | (uint64_t(rem_e32[22]) << 32) | (uint64_t(rem_e32[23]) << 48);
    constexpr uint64_t r63 = uint64_t(rem_e63[20]) | (uint64_t(rem_e63[21]) << 16) | (uint64_t(rem_e63[22]) << 32) | (uint64_t(rem_e63[23]) << 48);
    code[5] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
  } else if constexpr(5 < ORDER) { code[5] = uint32_t(0); }
}

template <int32_t orderi8, int32_t ORDER>
__device__ __forceinline__ int8_t* write_i8(const uint32_t (&code)[ORDER], int8_t* A, int64_t strideA) {
  if constexpr(0 < orderi8 && 0 < ORDER) { *A = int8_t(code[0]); } else { return A; }
  if constexpr(1 < orderi8 && 0 < ORDER) { *(A += strideA) = int8_t(code[0] >> 8); }
  if constexpr(2 < orderi8 && 0 < ORDER) { *(A += strideA) = int8_t(code[0] >> 16); }
  if constexpr(3 < orderi8 && 0 < ORDER) { *(A += strideA) = int8_t(code[0] >> 24); }
  if constexpr(4 < orderi8 && 1 < ORDER) { *(A += strideA) = int8_t(code[1]); }
  if constexpr(5 < orderi8 && 1 < ORDER) { *(A += strideA) = int8_t(code[1] >> 8); }
  if constexpr(6 < orderi8 && 1 < ORDER) { *(A += strideA) = int8_t(code[1] >> 16); }
  if constexpr(7 < orderi8 && 1 < ORDER) { *(A += strideA) = int8_t(code[1] >> 24); }
  if constexpr(8 < orderi8 && 2 < ORDER) { *(A += strideA) = int8_t(code[2]); }
  if constexpr(9 < orderi8 && 2 < ORDER) { *(A += strideA) = int8_t(code[2] >> 8); }
  if constexpr(10 < orderi8 && 2 < ORDER) { *(A += strideA) = int8_t(code[2] >> 16); }
  if constexpr(11 < orderi8 && 2 < ORDER) { *(A += strideA) = int8_t(code[2] >> 24); }
  if constexpr(12 < orderi8 && 3 < ORDER) { *(A += strideA) = int8_t(code[3]); }
  if constexpr(13 < orderi8 && 3 < ORDER) { *(A += strideA) = int8_t(code[3] >> 8); }
  if constexpr(14 < orderi8 && 3 < ORDER) { *(A += strideA) = int8_t(code[3] >> 16); }
  if constexpr(15 < orderi8 && 3 < ORDER) { *(A += strideA) = int8_t(code[3] >> 24); }
  if constexpr(16 < orderi8 && 4 < ORDER) { *(A += strideA) = int8_t(code[4]); }
  if constexpr(17 < orderi8 && 4 < ORDER) { *(A += strideA) = int8_t(code[4] >> 8); }
  if constexpr(18 < orderi8 && 4 < ORDER) { *(A += strideA) = int8_t(code[4] >> 16); }
  if constexpr(19 < orderi8 && 4 < ORDER) { *(A += strideA) = int8_t(code[4] >> 24); }
  if constexpr(20 < orderi8 && 5 < ORDER) { *(A += strideA) = int8_t(code[5]); }
  if constexpr(21 < orderi8 && 5 < ORDER) { *(A += strideA) = int8_t(code[5] >> 8); }
  if constexpr(22 < orderi8 && 5 < ORDER) { *(A += strideA) = int8_t(code[5] >> 16); }
  if constexpr(23 < orderi8 && 5 < ORDER) { *(A += strideA) = int8_t(code[5] >> 24); }
  return (A += strideA);
}

struct u64_add {
  __device__ __forceinline__ ulonglong2 operator()(ulonglong2 a, ulonglong2 b)
  { a.x += b.x; a.y += b.y + (a.x >> 63); a.x &= 0x7fffffffffffffffllu; return a; }
};

template <int32_t beta, int32_t sign>
__device__ __forceinline__ uint64_t* conv_acc(ulonglong2 acc, int64_t M, uint32_t corr, uint64_t* out, int32_t stride) {
  uint64_t a[2]{ uint64_t(acc.x), uint64_t(acc.y) }; device::int8::add_shifted(a, M, corr);
  if constexpr(beta) { device::int8::add_shifted(a, int64_t(out[0]), uint32_t(0)); device::int8::add_shifted(a, int64_t(out[stride]), uint32_t(63)); }
  if constexpr(sign) { out[0] = -a[0]; out[stride] = -a[1]; } else { out[0] = a[0]; out[stride] = a[2]; }
  return &out[int64_t(stride) << 1];
}

template <int32_t orderi8, int32_t beta, int32_t BLOCK_THREADS, class matrix_t>
__global__ void quantize_crt_kernel(int64_t M, const matrix_t* __restrict__ A, int64_t lda, uint32_t corr, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB, uint64_t* __restrict__ vsum) {
  constexpr int32_t ORDER = (orderi8 + 3) / 4, Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  constexpr int64_t BLOCK_THREADS_64 = int64_t(BLOCK_THREADS);
  int32_t expon = vexp[blockIdx.x]; A = &A[int64_t(blockIdx.x) * lda]; B = &B[int64_t(blockIdx.x) * ldb]; vsum = &vsum[blockIdx.x];
  if (expon == int_max) {
    uint32_t code[ORDER]{};
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64) {
      if constexpr(Complex) { write_i8<orderi8>(code, write_i8<orderi8>(code, write_i8<orderi8>(code, &B[i], strideB), strideB), strideB); }
        else { write_i8<orderi8>(code, &B[i], strideB); }
    }

    if (int32_t(threadIdx.x) == 0) {
      if constexpr(Complex && (!beta)) { *vsum = uint64_t(0); *(vsum += int32_t(gridDim.x)) = uint64_t(0); *(vsum += int32_t(gridDim.x)) = uint64_t(0); *(vsum += int32_t(gridDim.x)) = uint64_t(0); }
        else if constexpr(!beta) { *vsum = uint64_t(0); *(vsum += int32_t(gridDim.x)) = uint64_t(0); }
    }
  } else if constexpr(Complex) {
    __shared__ uint64_t rl[BLOCK_THREADS][2], im[BLOCK_THREADS][2]; uint32_t code[ORDER];
    rl[threadIdx.x][0] = rl[threadIdx.x][1] = im[threadIdx.x][0] = im[threadIdx.x][1] = uint64_t(0);
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}, A_im[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i.x, expon, e); device::int8::add_shifted(A_rl, q_rl, e); device::int8::add_shifted(rl[threadIdx.x], q_rl, e);
      int64_t q_im = device::int8::round_i64(A_i.y, expon, e); device::int8::add_shifted(A_im, q_im, e); device::int8::add_shifted(im[threadIdx.x], q_im, e);
      device::int8::add_shifted(A_rl, int64_t(1), corr); device::int8::add_shifted(A_im, int64_t(1), corr);

      quantize_i8<orderi8>(A_rl[0], uint32_t(A_rl[1]), code); int8_t* B_i = write_i8<orderi8>(code, &B[i], strideB);
      device::int8::add_shifted(A_rl, q_im, e);
      device::int8::add_shifted(A_rl, int64_t(1), corr);

      quantize_i8<orderi8>(A_im[0], uint32_t(A_im[1]), code); B_i = write_i8<orderi8>(code, B_i, strideB);
      quantize_i8<orderi8>(A_rl[0], uint32_t(A_rl[1]), code); write_i8<orderi8>(code, B_i, strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage temp_reduce[2];
    ulonglong2 threadA = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(temp_reduce[0]).Reduce(make_ulonglong2(rl[threadIdx.x][0], rl[threadIdx.x][1]), u64_add());
    ulonglong2 threadB = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(temp_reduce[1]).Reduce(make_ulonglong2(im[threadIdx.x][0], im[threadIdx.x][1]), u64_add());
    if (int32_t(threadIdx.x) == 0) { conv_acc<beta, 0>(threadB, M, corr, conv_acc<beta, 0>(threadA, M, corr, vsum, int32_t(gridDim.x)), int32_t(gridDim.x)); }
  } else {
    __shared__ uint64_t rl[BLOCK_THREADS][2]; uint32_t code[ORDER];
    rl[threadIdx.x][0] = rl[threadIdx.x][1] = uint64_t(0);
    for (int64_t i = int64_t(threadIdx.x); i < M; i += BLOCK_THREADS_64) {
      matrix_t A_i = A[i]; uint64_t A_rl[2]{}; uint32_t e;
      int64_t q_rl = device::int8::round_i64(A_i, expon, e); device::int8::add_shifted(A_rl, q_rl, e); device::int8::add_shifted(rl[threadIdx.x], q_rl, e);
      device::int8::add_shifted(A_rl, int64_t(1), corr);
      quantize_i8<orderi8>(A_rl[0], uint32_t(A_rl[1]), code); write_i8<orderi8>(code, &B[i], strideB);
    }

    __shared__ typename cub::BlockReduce<ulonglong2, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong2 threadA = cub::BlockReduce<ulonglong2, BLOCK_THREADS>(temp_reduce).Reduce(make_ulonglong2(rl[threadIdx.x][0], rl[threadIdx.x][1]), u64_add());
    if (int32_t(threadIdx.x) == 0) { conv_acc<beta, 1>(threadA, M, corr, vsum, int32_t(gridDim.x)); }
  }
};

template <class matrix_t>
inline void quantize_crt_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const matrix_t* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum) {
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

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

  void quantize_crt(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, int8_t* B, int32_t ldb, int32_t beta, uint64_t* vsum)
  { quantize_crt_dispatcher(stream, M, N, orderA, A, lda, corr, vexp, B, ldb, beta, vsum); }

}
