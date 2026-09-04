
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>
#include <limits>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
template <int32_t CRT, int32_t orderi8, int32_t ORDER> __device__ __forceinline__ void quantize_i8(uint64_t lo, uint32_t hi, uint32_t (&code)[ORDER]) {
  if constexpr(CRT) {
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
  } else {
    uint32_t c = 0; constexpr uint32_t u31 = uint32_t(0x80000000);
    if constexpr(0 < orderi8 && 0 < ORDER) { code[0] = device::int8::conv_u8i8(uint32_t(lo), c); } else if constexpr(0 < ORDER) { code[0] = uint32_t(0); }
    if constexpr(4 < orderi8 && 1 < ORDER) { code[1] = device::int8::conv_u8i8(uint32_t(lo >> 32) | (hi << 31), c); } else if constexpr(1 < ORDER) { code[1] = uint32_t(0); }
    if constexpr(8 < orderi8 && 2 < ORDER) { code[2] = device::int8::conv_u8i8((hi & u31) | (hi >> 1), c); } else if constexpr(2 < ORDER) { code[2] = uint32_t(0); }
  }
}

template <int32_t orderi8, int32_t ORDER>
__device__ __forceinline__ void write_i8(const uint32_t (&code)[ORDER], int8_t* A, int64_t strideA) {
  if constexpr(0 < orderi8 && 0 < ORDER) { *A = int8_t(code[0]); }
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
}

template <int32_t orderi8, int32_t CRT, class matrix_t>
__global__ void quantize_kernel(int64_t M, const matrix_t* __restrict__ A, int64_t lda, uint32_t corr, const int32_t* __restrict__ vexp, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  constexpr int32_t ORDER = (orderi8 + 3) / 4, Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int64_t y = (int64_t(blockIdx.x) << 8) + int64_t(threadIdx.x), panelB = int64_t(orderi8) * strideB;
  int32_t expon = vexp[blockIdx.y]; B = &B[y + (int64_t(blockIdx.y) * ldb)];
  if (M <= y || expon == int_max) {
    uint32_t code[ORDER]{};
    write_i8<orderi8>(code, B, strideB);
    if constexpr(Complex)
    { write_i8<orderi8>(code, B += panelB, strideB); write_i8<orderi8>(code, B += panelB, strideB); }
  }
  else {
    uint32_t code[ORDER];
    matrix_t A_i = A[y + (int64_t(blockIdx.y) * lda)];
    if constexpr(Complex) {
      uint64_t rl[2]{}, im[2]{}; int32_t e;
      int64_t q_rl = device::int8::round_i64(A_i.x, expon, e); device::int8::add_shifted(rl, q_rl, uint32_t(e));
      int64_t q_im = device::int8::round_i64(A_i.y, expon, e); device::int8::add_shifted(im, q_im, uint32_t(e));
      if constexpr(CRT) { device::int8::add_shifted(rl, int64_t(1), corr); device::int8::add_shifted(im, int64_t(1), corr); }

      quantize_i8<CRT, orderi8>(rl[0], uint32_t(rl[1]), code); write_i8<orderi8>(code, B, strideB);
      device::int8::add_shifted(rl, q_im, uint32_t(e));
      if constexpr(CRT) { device::int8::add_shifted(rl, int64_t(1), corr); }

      quantize_i8<CRT, orderi8>(im[0], uint32_t(im[1]), code); write_i8<orderi8>(code, B += panelB, strideB);
      quantize_i8<CRT, orderi8>(rl[0], uint32_t(rl[1]), code); write_i8<orderi8>(code, B += panelB, strideB);
    }
    else {
      uint64_t rl[2]{}; int32_t e;
      int64_t q_rl = device::int8::round_i64(A_i, expon, e); device::int8::add_shifted(rl, q_rl, uint32_t(e));
      if constexpr(CRT) { device::int8::add_shifted(rl, int64_t(1), corr); }
      quantize_i8<CRT, orderi8>(rl[0], uint32_t(rl[1]), code); write_i8<orderi8>(code, B, strideB);
    }
  }
};

template <class matrix_t>
inline void quantize_dispatcher(cudaStream_t stream, int32_t M, const matrix_t* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
  constexpr int32_t block_threads = 256;
  dim3 grid(uint32_t(dimX) >> 8, uint32_t(dimY));
  int64_t M64 = int64_t(M), ldc64 = int64_t(ldc), dimX64 = int64_t(dimX), strideA = int64_t(dimY) * dimX64;

  if (corr) switch (dimZ) {
    case 2: quantize_kernel<2, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 3: quantize_kernel<3, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 4: quantize_kernel<4, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 5: quantize_kernel<5, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 6: quantize_kernel<6, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 7: quantize_kernel<7, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 8: quantize_kernel<8, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 9: quantize_kernel<9, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 10: quantize_kernel<10, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 11: quantize_kernel<11, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 12: quantize_kernel<12, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 13: quantize_kernel<13, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 14: quantize_kernel<14, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 15: quantize_kernel<15, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 16: quantize_kernel<16, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 17: quantize_kernel<17, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 18: quantize_kernel<18, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 19: quantize_kernel<19, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 20: quantize_kernel<20, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 21: quantize_kernel<21, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 22: quantize_kernel<22, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 23: quantize_kernel<23, 1> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    default: return;
  } switch (dimZ) {
    case 1: quantize_kernel<1, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 2: quantize_kernel<2, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 3: quantize_kernel<3, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 4: quantize_kernel<4, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 5: quantize_kernel<5, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 6: quantize_kernel<6, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 7: quantize_kernel<7, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 8: quantize_kernel<8, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 9: quantize_kernel<9, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 10: quantize_kernel<10, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 11: quantize_kernel<11, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    case 12: quantize_kernel<12, 0> <<< grid, block_threads, 0, stream >>> (M64, C, ldc64, corr, vexp, A, dimX64, strideA); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize(cudaStream_t stream, int32_t M, const double* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

  void quantize(cudaStream_t stream, int32_t M, const float* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

  void quantize(cudaStream_t stream, int32_t M, const __half* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

  void quantize(cudaStream_t stream, int32_t M, const cuDoubleComplex* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

  void quantize(cudaStream_t stream, int32_t M, const cuComplex* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

  void quantize(cudaStream_t stream, int32_t M, const __half2* C, int32_t ldc, uint32_t corr, const int32_t* vexp, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A)
  { quantize_dispatcher(stream, M, C, ldc, corr, vexp, dimZ, dimY, dimX, A); }

}
