
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_selector.hpp>

__device__ __forceinline__ void quantize_f64_i8limbs(double x, int32_t expon, uint64_t lo, uint32_t hi, uint32_t (&code)[3]) {
  int64_t q = device::int8::round_f64(x, expon, expon);
  lo += (uint64_t(q) << expon) & device::int8::i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);

  uint32_t c = 0;
  code[0] = device::int8::conv_u8i8(uint32_t(lo), c);
  code[1] = device::int8::conv_u8i8((uint32_t(lo >> 32) & device::int8::i31) | (hi << 31), c);
  code[2] = device::int8::conv_u8i8(hi >> 1, c);
}

template <uint64_t MO, uint64_t R32, uint64_t R63>
__device__ __forceinline__ void quantize_f64_i8rems(double x, int32_t expon, uint64_t lo, uint32_t hi, uint32_t (&code)[3]) {
  constexpr uint32_t m_lo = uint32_t(MO), m_hi = uint32_t(MO >> 32);
  constexpr uint32_t r32_lo = uint32_t(R32), r32_hi = uint32_t(R32 >> 32);
  constexpr uint32_t r63_lo = uint32_t(R63), r63_hi = uint32_t(R63 >> 32);

  int64_t q = device::int8::round_f64(x, expon, expon);
  lo += (uint64_t(q) << expon) & device::int8::i63;
  hi += uint32_t(q >> (63 - expon)) + uint32_t(lo >> 63);
  uint32_t lo_32 = uint32_t(lo), mi = uint32_t(lo >> 32) & device::int8::i31;

  code[0] = device::int8::conv_u32i8_modular<m_lo, r32_lo, r63_lo>(lo_32, mi, hi);
  code[1] = device::int8::conv_u32i8_modular<m_hi, r32_hi, r63_hi>(lo_32, mi, hi);
}

template <uint32_t ORDER, uint64_t MO, uint64_t R32, uint64_t R63, class matrix_t, class matrix_const_ptr, int32_t op>
__global__ void quantize_kernel(int64_t M, int64_t N, matrix_const_ptr A, int64_t lda, uint64_t lo, uint32_t hi, const uint64_t* __restrict__ vec_expon, int8_t* __restrict__ B, int64_t ldb, int64_t strideB) {
  int64_t y = (int64_t(blockIdx.x) << 8) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  int32_t expon = -int32_t(vec_expon[x]);
  matrix_t A_i = y < M ? A[y + x * lda] : matrix_t();
  uint32_t code[3]; int8_t* bytes = (int8_t*)&code[0];

  if constexpr(op == 0)
    quantize_f64_i8limbs(double(A_i), expon, lo, hi, code);
  else if constexpr(op == 1)
    quantize_f64_i8limbs(double(A_i.x), expon, lo, hi, code);
  else if constexpr(op == 2)
    quantize_f64_i8rems<MO, R32, R63>(double(A_i), expon, lo, hi, code);
  else if constexpr(op == 3)
    quantize_f64_i8rems<MO, R32, R63>(double(A_i.x), expon, lo, hi, code);

  if (M <= y)
    code[0] = code[1] = code[2] = uint32_t(0);

  int64_t iter = y + x * ldb;
  #pragma unroll
  for (uint32_t k = 0; k < ORDER; ++k)
  { B[iter] = bytes[k]; iter += strideB; }

  if constexpr(op & 1) {
    if constexpr(op == 1)
      quantize_f64_i8limbs(double(A_i.y), expon, lo, hi, code);
    else if constexpr(op == 3)
      quantize_f64_i8rems<MO, R32, R63>(double(A_i.y), expon, lo, hi, code);

    if (M <= y)
      code[0] = code[1] = code[2] = uint32_t(0);

    #pragma unroll
    for (uint32_t k = 0; k < ORDER; ++k)
    { B[iter] = bytes[k]; iter += strideB; }
  }
};

template <class matrix_t, class matrix_const_ptr, int32_t iter, int32_t op>
inline void quantize_dispatcher(cudaStream_t stream, int64_t M, int64_t N, matrix_const_ptr C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int64_t lda) {
  constexpr uint64_t MO = CRT::modular(iter), R32 = CRT::rem_e32(iter), R63 = CRT::rem_e63(iter);
  dim3 grid(uint32_t(lda >> 8), uint32_t(N)), block_threads(uint32_t(256));
  int64_t strideA = N * lda;
  uint64_t lo = (-uint64_t(umax < 64)) & (uint64_t(1) << umax);
  uint32_t hi = (-uint32_t(63 < umax)) & (uint32_t(1) << (umax - 63));

  switch (orderA) {
    case 1: quantize_kernel<1, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 2: quantize_kernel<2, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 3: quantize_kernel<3, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 4: quantize_kernel<4, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 5: quantize_kernel<5, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 6: quantize_kernel<6, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 7: quantize_kernel<7, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 8: quantize_kernel<8, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 9: quantize_kernel<9, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 10: quantize_kernel<10, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    case 11: quantize_kernel<11, MO, R32, R63, matrix_t, matrix_const_ptr, op> <<< grid, block_threads, 0, stream >>> (M, N, C, ldc, lo, hi, vec_expon, A, lda, strideA); break;
    default: break;
  }
}

template <class matrix_t, class matrix_const_ptr, int32_t op>
inline void quantize_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, matrix_const_ptr C, int64_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int64_t lda) {
  switch (iter) {
    case 0: quantize_dispatcher<matrix_t, matrix_const_ptr, 0, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    case 1: quantize_dispatcher<matrix_t, matrix_const_ptr, 1, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    case 2: quantize_dispatcher<matrix_t, matrix_const_ptr, 2, op> (stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda); break;
    default: break;
  }
}

void internal::int8::quantize_f64(cudaStream_t stream, int32_t M, int32_t N, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double, const double* __restrict__, -1, 0>(stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf64(cudaStream_t stream, int32_t M, int32_t N, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double2, const double2* __restrict__, -1, 1>(stream, M, N, (double2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f32(cudaStream_t stream, int32_t M, int32_t N, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float, const float* __restrict__, -1, 0>(stream, M, N, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf32(cudaStream_t stream, int32_t M, int32_t N, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float2, const float2* __restrict__, -1, 1>(stream, M, N, (float2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const double* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double, const double* __restrict__, 2>(stream, M, N, iter, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf64_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<double>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<double2, const double2* __restrict__, 3>(stream, M, N, iter, (double2*)C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_f32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const float* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float, const float* __restrict__, 2>(stream, M, N, iter, C, ldc, umax, vec_expon, orderA, A, lda);
}

void internal::int8::quantize_cf32_modular(cudaStream_t stream, int32_t M, int32_t N, int32_t iter, const std::complex<float>* C, int32_t ldc, int32_t umax, const uint64_t* vec_expon, int32_t orderA, int8_t* A, int32_t lda) {
  quantize_dispatcher<float2, const float2* __restrict__, 3>(stream, M, N, iter, (float2*)C, ldc, umax, vec_expon, orderA, A, lda);
}
