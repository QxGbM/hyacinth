
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>

template <int32_t ORDER, class real_t>
__device__ __forceinline__ void quantize_i8limbs(real_t x, int32_t expon, uint32_t (&code)[3]) {
  uint64_t q = uint64_t(device::int8::round_i64(x, expon, expon)), lo = q << expon;
  uint32_t c = 0;

  if constexpr(0 < ORDER) { code[0] = device::int8::conv_u8i8(uint32_t(lo), c); } else { code[0] = uint32_t(0); }
  if constexpr(4 < ORDER) { code[1] = device::int8::conv_u8i8(uint32_t(lo >> 32), c); } else { code[1] = uint32_t(0); }
  if constexpr(8 < ORDER) {
    uint32_t hi = (expon ? uint32_t(q >> (64 - expon)) : uint32_t(0)) | ((-uint32_t(q >> 63)) << expon);
    code[2] = device::int8::conv_u8i8(hi, c);
  } else { code[2] = uint32_t(0); }
}

template <int32_t iter, int32_t ORDER, class real_t>
__device__ __forceinline__ void quantize_i8rems(real_t x, int32_t expon, uint64_t lo, uint32_t hi, uint32_t (&code)[3]) {
  constexpr int32_t iterX8 = iter << 3;
  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  constexpr uint32_t i31 = 0x7fffffffu;

  int64_t q = device::int8::round_i64(x, expon, expon);
  lo += (uint64_t(q) << expon) & i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  uint32_t lo_32 = uint32_t(lo), mi = uint32_t(lo >> 32) & i31;

  if constexpr(0 < ORDER) {
    using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
    constexpr uint64_t m_lo = uint64_t(mo[iterX8]) | (uint64_t(mo[iterX8 + 1]) << 16) | (uint64_t(mo[iterX8 + 2]) << 32) | (uint64_t(mo[iterX8 + 3]) << 48);
    constexpr uint64_t r32_lo = uint64_t(rem_e32[iterX8]) | (uint64_t(rem_e32[iterX8 + 1]) << 16) | (uint64_t(rem_e32[iterX8 + 2]) << 32) | (uint64_t(rem_e32[iterX8 + 3]) << 48);
    constexpr uint64_t r63_lo = uint64_t(rem_e63[iterX8]) | (uint64_t(rem_e63[iterX8 + 1]) << 16) | (uint64_t(rem_e63[iterX8 + 2]) << 32) | (uint64_t(rem_e63[iterX8 + 3]) << 48);
    code[0] = device::int8::conv_u32i8_modular<m_lo, r32_lo, r63_lo>(lo_32, mi, hi);
  } else { code[0] = uint32_t(0); }

  if constexpr(4 < ORDER) {
    using U8CRT::mo, U8CRT::rem_e32, U8CRT::rem_e63;
    constexpr uint64_t m_hi = uint64_t(mo[iterX8 + 4]) | (uint64_t(mo[iterX8 + 5]) << 16) | (uint64_t(mo[iterX8 + 6]) << 32) | (uint64_t(mo[iterX8 + 7]) << 48);
    constexpr uint64_t r32_hi = uint64_t(rem_e32[iterX8 + 4]) | (uint64_t(rem_e32[iterX8 + 5]) << 16) | (uint64_t(rem_e32[iterX8 + 6]) << 32) | (uint64_t(rem_e32[iterX8 + 7]) << 48);
    constexpr uint64_t r63_hi = uint64_t(rem_e63[iterX8 + 4]) | (uint64_t(rem_e63[iterX8 + 5]) << 16) | (uint64_t(rem_e63[iterX8 + 6]) << 32) | (uint64_t(rem_e63[iterX8 + 7]) << 48);
    code[1] = device::int8::conv_u32i8_modular<m_hi, r32_hi, r63_hi>(lo_32, mi, hi);
  } else { code[1] = uint32_t(0); }
}

template <int32_t ORDER>
__device__ __forceinline__ void write_i8(const uint32_t (&code)[3], int8_t* A, int64_t strideA) {
  if constexpr(0 < ORDER) { *A = uint8_t(code[0]); }
  if constexpr(1 < ORDER) { *(A += strideA) = uint8_t(code[0] >> 8); }
  if constexpr(2 < ORDER) { *(A += strideA) = uint8_t(code[0] >> 16); }
  if constexpr(3 < ORDER) { *(A += strideA) = uint8_t(code[0] >> 24); }
  if constexpr(4 < ORDER) { *(A += strideA) = uint8_t(code[1]); }
  if constexpr(5 < ORDER) { *(A += strideA) = uint8_t(code[1] >> 8); }
  if constexpr(6 < ORDER) { *(A += strideA) = uint8_t(code[1] >> 16); }
  if constexpr(7 < ORDER) { *(A += strideA) = uint8_t(code[1] >> 24); }
  if constexpr(8 < ORDER) { *(A += strideA) = uint8_t(code[2]); }
  if constexpr(9 < ORDER) { *(A += strideA) = uint8_t(code[2] >> 8); }
  if constexpr(10 < ORDER) { *(A += strideA) = uint8_t(code[2] >> 16); }
  if constexpr(11 < ORDER) { *(A += strideA) = uint8_t(code[2] >> 24); }
}

template <int32_t ORDER, class matrix_t, int32_t op>
__global__ void quantize_kernel(int64_t M, int64_t N, const matrix_t* __restrict__ A, int64_t lda, uint64_t lo, uint32_t hi, int32_t umax, const int32_t* __restrict__ vec_expon, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  constexpr int32_t op_complex = op & 8, op_iter = (op & 7) - 1;
  int64_t y = (int64_t(blockIdx.x) << 8) + int64_t(threadIdx.x), iter = y + (int64_t(blockIdx.y) * lda);
  int32_t expon = umax - vec_expon[blockIdx.y], invalidA = int32_t(M <= y || N <= int64_t(blockIdx.y));
  matrix_t A_i = invalidA ? matrix_t() : A[iter];
  uint32_t code[3];

  if (invalidA) { code[0] = code[1] = code[2] = uint32_t(0); }
  else if constexpr(op_iter == -1) {
    if constexpr(op_complex) quantize_i8limbs<ORDER>(A_i.x, expon, code);
      else quantize_i8limbs<ORDER>(A_i, expon, code);
  }
  else {
    if constexpr(op_complex) quantize_i8rems<op_iter, ORDER>(A_i.x, expon, lo, hi, code);
      else quantize_i8rems<op_iter, ORDER>(A_i, expon, lo, hi, code);
  }
  write_i8<ORDER>(code, &B[y + (int64_t(blockIdx.y) * ldb)], strideB);

  if constexpr(op_complex) {
    if (invalidA) { code[0] = code[1] = code[2] = uint32_t(0); }
    else {
      if constexpr(op_iter == -1) quantize_i8limbs<ORDER>(A_i.y, expon, code);
        else quantize_i8rems<op_iter, ORDER>(A_i.y, expon, lo, hi, code);
    }
    write_i8<ORDER>(code, &B[y + (int64_t(blockIdx.y) * ldb) + (int64_t(ORDER) * strideB)], strideB);
  }
};

template <int32_t op, class matrix_t>
inline void quantize_dispatcher(cudaStream_t stream, int64_t M, int64_t N, const matrix_t* C, int64_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
  dim3 grid(uint32_t(dimX) >> 8, uint32_t(dimY)), block_threads(uint32_t(256));
  int64_t dimX64 = int64_t(dimX), strideA = int64_t(dimY) * dimX64;
  uint64_t lo = umax < 63 ? (uint64_t(1) << umax) : uint64_t(0);
  uint32_t hi = 63 <= umax ? (uint32_t(1) << (umax - 63)) : uint32_t(0);

  switch (dimZ) {
    case 1: quantize_kernel<1, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 2: quantize_kernel<2, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 3: quantize_kernel<3, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 4: quantize_kernel<4, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 5: quantize_kernel<5, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 6: quantize_kernel<6, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 7: quantize_kernel<7, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 8: quantize_kernel<8, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 9: quantize_kernel<9, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 10: quantize_kernel<10, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 11: quantize_kernel<11, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    case 12: quantize_kernel<12, matrix_t, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, umax, vec_expon, A, dimX64, strideA); return;
    default: return;
  }
}

namespace internal::int8 {

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<0>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<8>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<8>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize(cudaStream_t stream, int32_t M, int32_t N, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    quantize_dispatcher<8>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A);
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const double* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<3>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const float* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<3>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const __half* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<1>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<2>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<3>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const cuDoubleComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<9>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<10>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<11>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const cuComplex* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<9>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<10>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<11>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

  void quantize_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const __half2* C, int32_t ldc, int32_t umax, const int32_t* vec_expon, int32_t dimZ, int32_t dimY, int32_t dimX, int8_t* A) {
    switch (iter) {
      case 0: quantize_dispatcher<9>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 1: quantize_dispatcher<10>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      case 2: quantize_dispatcher<11>(stream, int64_t(M), int64_t(N), C, int64_t(ldc), umax, vec_expon, dimZ, dimY, dimX, A); return;
      default: return;
    }
  }

}
