
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>

template <int32_t op, int32_t ORDER> __device__ __forceinline__ void quantize_i8(uint64_t lo, uint32_t hi, uint32_t (&code)[3]) {
  constexpr uint32_t i31 = 0x7fffffffu, u31 = ~i31;
  if constexpr(op < 0) {
    uint32_t c = 0;
    if constexpr(0 < ORDER) { code[0] = device::int8::conv_u8i8(uint32_t(lo), c); } else { code[0] = uint32_t(0); }
    if constexpr(4 < ORDER) { code[1] = device::int8::conv_u8i8(uint32_t(lo >> 32) | (hi << 31), c); } else { code[1] = uint32_t(0); }
    if constexpr(8 < ORDER) { code[2] = device::int8::conv_u8i8((hi & u31) | (hi >> 1), c); } else { code[2] = uint32_t(0); }
  }
  else {
    constexpr int32_t op8 = op << 3;
    uint32_t lo_32 = uint32_t(lo), mi = uint32_t(lo >> 32) & i31;
    using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
    if constexpr(0 < ORDER) {
      constexpr uint64_t m = uint64_t(mo[op8]) | (uint64_t(mo[op8 + 1]) << 16) | (uint64_t(mo[op8 + 2]) << 32) | (uint64_t(mo[op8 + 3]) << 48);
      constexpr uint64_t r32 = uint64_t(rem_e32[op8]) | (uint64_t(rem_e32[op8 + 1]) << 16) | (uint64_t(rem_e32[op8 + 2]) << 32) | (uint64_t(rem_e32[op8 + 3]) << 48);
      constexpr uint64_t r63 = uint64_t(rem_e63[op8]) | (uint64_t(rem_e63[op8 + 1]) << 16) | (uint64_t(rem_e63[op8 + 2]) << 32) | (uint64_t(rem_e63[op8 + 3]) << 48);
      code[0] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    } else { code[0] = uint32_t(0); }

    if constexpr(4 < ORDER) {
      constexpr uint64_t m = uint64_t(mo[op8 + 4]) | (uint64_t(mo[op8 + 5]) << 16) | (uint64_t(mo[op8 + 6]) << 32) | (uint64_t(mo[op8 + 7]) << 48);
      constexpr uint64_t r32 = uint64_t(rem_e32[op8 + 4]) | (uint64_t(rem_e32[op8 + 5]) << 16) | (uint64_t(rem_e32[op8 + 6]) << 32) | (uint64_t(rem_e32[op8 + 7]) << 48);
      constexpr uint64_t r63 = uint64_t(rem_e63[op8 + 4]) | (uint64_t(rem_e63[op8 + 5]) << 16) | (uint64_t(rem_e63[op8 + 6]) << 32) | (uint64_t(rem_e63[op8 + 7]) << 48);
      code[1] = device::int8::conv_u32i8_modular<m, r32, r63>(lo_32, mi, hi);
    } else { code[1] = uint32_t(0); }
  }
}

template <int32_t ORDER>
__device__ __forceinline__ void write_i8(const uint32_t (&code)[3], int8_t* A, int64_t strideA) {
  if constexpr(0 < ORDER) { *A = int8_t(code[0]); }
  if constexpr(1 < ORDER) { *(A += strideA) = int8_t(code[0] >> 8); }
  if constexpr(2 < ORDER) { *(A += strideA) = int8_t(code[0] >> 16); }
  if constexpr(3 < ORDER) { *(A += strideA) = int8_t(code[0] >> 24); }
  if constexpr(4 < ORDER) { *(A += strideA) = int8_t(code[1]); }
  if constexpr(5 < ORDER) { *(A += strideA) = int8_t(code[1] >> 8); }
  if constexpr(6 < ORDER) { *(A += strideA) = int8_t(code[1] >> 16); }
  if constexpr(7 < ORDER) { *(A += strideA) = int8_t(code[1] >> 24); }
  if constexpr(8 < ORDER) { *(A += strideA) = int8_t(code[2]); }
  if constexpr(9 < ORDER) { *(A += strideA) = int8_t(code[2] >> 8); }
  if constexpr(10 < ORDER) { *(A += strideA) = int8_t(code[2] >> 16); }
  if constexpr(11 < ORDER) { *(A += strideA) = int8_t(code[2] >> 24); }
}

template <int32_t ORDER, class matrix_t, int32_t op>
__global__ void quantize_kernel(int64_t M, int64_t N, const matrix_t* __restrict__ A, int64_t lda, int32_t umax, const int32_t* __restrict__ vec_expon, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  constexpr int32_t op_complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int64_t y = (int64_t(blockIdx.x) << 8) + int64_t(threadIdx.x), panelB = int64_t(ORDER) * strideB;
  int32_t invalidA = int32_t(M <= y || N <= int64_t(blockIdx.y));
  uint32_t code[3]; B = &B[y + (int64_t(blockIdx.y) * ldb)];
  if (invalidA) {
    code[0] = code[1] = code[2] = uint32_t(0);
    write_i8<ORDER>(code, B, strideB);
    if constexpr(op_complex)
    { write_i8<ORDER>(code, B += panelB, strideB); write_i8<ORDER>(code, B += panelB, strideB); }
  }
  else {
    int32_t expon = umax - vec_expon[blockIdx.y];
    matrix_t A_i = A[y + (int64_t(blockIdx.y) * lda)];
    if constexpr(op_complex) {
      uint64_t rl[2]{}, im[2]{};
      int32_t e_rl; int64_t q_rl = device::int8::round_i64(A_i.x, expon, e_rl); device::int8::add_shifted(rl, q_rl, uint32_t(e_rl));
      int32_t e_im; int64_t q_im = device::int8::round_i64(A_i.y, expon, e_im); device::int8::add_shifted(im, q_im, uint32_t(e_im));
      if constexpr(0 <= op)
      { device::int8::add_shifted(rl, int64_t(1), uint32_t(umax)); device::int8::add_shifted(im, int64_t(1), uint32_t(umax)); }

      quantize_i8<op, ORDER>(rl[0], uint32_t(rl[1]), code); write_i8<ORDER>(code, B, strideB);
      quantize_i8<op, ORDER>(im[0], uint32_t(im[1]), code); write_i8<ORDER>(code, B += panelB, strideB);
      device::int8::add_shifted(im, q_rl, uint32_t(e_rl));
      if constexpr(0 <= op)
      { device::int8::add_shifted(im, int64_t(1), uint32_t(umax)); }
      quantize_i8<op, ORDER>(im[0], uint32_t(im[1]), code); write_i8<ORDER>(code, B += panelB, strideB);
    }
    else {
      uint64_t rl[2]{};
      int32_t e_rl; int64_t q_rl = device::int8::round_i64(A_i, expon, e_rl); device::int8::add_shifted(rl, q_rl, uint32_t(e_rl));
      if constexpr(0 <= op)
      { device::int8::add_shifted(rl, int64_t(1), uint32_t(umax)); }
      quantize_i8<op, ORDER>(rl[0], uint32_t(rl[1]), code); write_i8<ORDER>(code, B, strideB);
    }
  }
};

template <int32_t op, class matrix_t>
inline void quantize_dispatcher(cudaStream_t stream, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
  dim3 grid(uint32_t(dimX) >> 8, uint32_t(dimY)), block_threads(uint32_t(256));
  int64_t dimX64 = int64_t(dimX), strideA = int64_t(dimY) * dimX64;

  switch (dimZ) {
    case 1: quantize_kernel<1, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 2: quantize_kernel<2, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 3: quantize_kernel<3, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 4: quantize_kernel<4, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 5: quantize_kernel<5, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 6: quantize_kernel<6, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 7: quantize_kernel<7, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 8: quantize_kernel<8, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 9: quantize_kernel<9, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 10: quantize_kernel<10, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 11: quantize_kernel<11, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    case 12: quantize_kernel<12, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, umax, vec_expon, A, dimX64, strideA); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<-1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t op, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (op) {
      case 0: quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

}
