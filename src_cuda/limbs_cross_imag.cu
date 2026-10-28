
#include <internal.hpp>
#include <ext_arith.hpp>

template<int32_t sign, int32_t sft, int32_t orderA>
static __device__ __forceinline__ void accumulate_single(uint64_t (&r)[orderA], uint64_t (&i)[orderA], int32_t X) {
  if constexpr(sign) { device::add_shifted<sft>(r, int64_t(-X)); device::add_shifted<sft>(i, int64_t(-X)); } else
  { device::add_shifted<sft>(r, int64_t(-X)); device::add_shifted<sft>(i, int64_t(X)); }
}

template<int32_t sign, int32_t sft, int32_t orderX, int32_t orderA>
static __device__ __forceinline__ void accumulate_complex(uint64_t (&r)[orderA], uint64_t (&i)[orderA], const int32_t* X, int64_t strideX) {
  if constexpr(0 < orderX) { accumulate_single<sign, sft>(r, i, *X); }
  if constexpr(1 < orderX) { accumulate_single<sign, sft + 8>(r, i, *(X += strideX)); }
  if constexpr(2 < orderX) { accumulate_single<sign, sft + 16>(r, i, *(X += strideX)); }
  if constexpr(3 < orderX) { accumulate_single<sign, sft + 24>(r, i, *(X += strideX)); }
  if constexpr(4 < orderX) { accumulate_single<sign, sft + 32>(r, i, *(X += strideX)); }
  if constexpr(5 < orderX) { accumulate_single<sign, sft + 40>(r, i, *(X += strideX)); }
  if constexpr(6 < orderX) { accumulate_single<sign, sft + 48>(r, i, *(X += strideX)); }
  if constexpr(7 < orderX) { accumulate_single<sign, sft + 56>(r, i, *(X += strideX)); }
  if constexpr(8 < orderX) { accumulate_single<sign, sft + 64>(r, i, *(X += strideX)); }
  if constexpr(9 < orderX) { accumulate_single<sign, sft + 72>(r, i, *(X += strideX)); }
  if constexpr(10 < orderX) { accumulate_single<sign, sft + 80>(r, i, *(X += strideX)); }
  if constexpr(11 < orderX) { accumulate_single<sign, sft + 88>(r, i, *(X += strideX)); }
}

template<int32_t sft, int32_t orderA, int32_t orderX, int32_t beta>
__global__ void limbs_cross_imaginary_kernel(int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    uint64_t r[orderA], i[orderA]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)];

    if constexpr(beta) {
      const uint64_t* A_in = A;
      if constexpr(0 < orderA) { r[0] = *A_in; } if constexpr(1 < orderA) { r[1] = *(A_in += strideA); } if constexpr(2 < orderA) { r[2] = *(A_in += strideA); }
      if constexpr(0 < orderA) { i[0] = *(A_in += strideA); } if constexpr(1 < orderA) { i[1] = *(A_in += strideA); } if constexpr(2 < orderA) { i[2] = *(A_in += strideA); }
    } else {
      const uint64_t* A_in = A;
      if constexpr(0 < orderA) { r[0] = *A_in; } if constexpr(1 < orderA) { r[1] = *(A_in += strideA); } if constexpr(2 < orderA) { r[2] = *(A_in += strideA); }
      if constexpr(0 < orderA) { i[0] = uint64_t(0); } if constexpr(1 < orderA) { i[1] = uint64_t(0); } if constexpr(2 < orderA) { i[2] = uint64_t(0); }
    }
    accumulate_complex<0, sft, orderX>(r, i, &X[y + (x * ldx)], strideX);
    accumulate_complex<1, sft, orderX>(r, i, &X[x + (y * ldx)], strideX);
    if constexpr(0 < orderA) { *A = r[0]; } if constexpr(1 < orderA) { *(A += strideA) = r[1]; } if constexpr(2 < orderA) { *(A += strideA) = r[2]; }
    if constexpr(0 < orderA) { *(A += strideA) = i[0]; } if constexpr(1 < orderA) { *(A += strideA) = i[1]; } if constexpr(2 < orderA) { *(A += strideA) = i[2]; }
  }
}

template <int32_t orderA, int32_t orderX>
static inline void limbs_cross_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t sft, const int32_t* X, int32_t ldx, uint64_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t(N + 127) >> 7, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta == 0 && sft == 0) { limbs_cross_imaginary_kernel<0, orderA, orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  switch (sft) {
    case 0: limbs_cross_imaginary_kernel<0, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 8: limbs_cross_imaginary_kernel<8, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 16: limbs_cross_imaginary_kernel<16, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 24: limbs_cross_imaginary_kernel<24, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 32: limbs_cross_imaginary_kernel<32, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 40: limbs_cross_imaginary_kernel<40, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 48: limbs_cross_imaginary_kernel<48, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 56: limbs_cross_imaginary_kernel<56, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 64: limbs_cross_imaginary_kernel<64, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 72: limbs_cross_imaginary_kernel<72, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 80: limbs_cross_imaginary_kernel<80, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 88: limbs_cross_imaginary_kernel<88, orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    default: return;
  }
}

void internal::int8::accumulate_limbs_cross_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t sft, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A) {
  if (orderA == 1) switch (orderX) {
    case 1: limbs_cross_imag<1, 1>(stream, beta, N, sft, X, ldx, A); return;
    case 2: limbs_cross_imag<1, 2>(stream, beta, N, sft, X, ldx, A); return;
    case 3: limbs_cross_imag<1, 3>(stream, beta, N, sft, X, ldx, A); return;
    case 4: limbs_cross_imag<1, 4>(stream, beta, N, sft, X, ldx, A); return;
    case 5: limbs_cross_imag<1, 5>(stream, beta, N, sft, X, ldx, A); return;
    case 6: limbs_cross_imag<1, 6>(stream, beta, N, sft, X, ldx, A); return;
    case 7: limbs_cross_imag<1, 7>(stream, beta, N, sft, X, ldx, A); return;
    case 8: limbs_cross_imag<1, 8>(stream, beta, N, sft, X, ldx, A); return;
    case 9: limbs_cross_imag<1, 9>(stream, beta, N, sft, X, ldx, A); return;
    case 10: limbs_cross_imag<1, 10>(stream, beta, N, sft, X, ldx, A); return;
    case 11: limbs_cross_imag<1, 11>(stream, beta, N, sft, X, ldx, A); return;
    case 12: limbs_cross_imag<1, 12>(stream, beta, N, sft, X, ldx, A); return;
    default: return;
  } else if (orderA == 2) switch (orderX) {
    case 1: limbs_cross_imag<2, 1>(stream, beta, N, sft, X, ldx, A); return;
    case 2: limbs_cross_imag<2, 2>(stream, beta, N, sft, X, ldx, A); return;
    case 3: limbs_cross_imag<2, 3>(stream, beta, N, sft, X, ldx, A); return;
    case 4: limbs_cross_imag<2, 4>(stream, beta, N, sft, X, ldx, A); return;
    case 5: limbs_cross_imag<2, 5>(stream, beta, N, sft, X, ldx, A); return;
    case 6: limbs_cross_imag<2, 6>(stream, beta, N, sft, X, ldx, A); return;
    case 7: limbs_cross_imag<2, 7>(stream, beta, N, sft, X, ldx, A); return;
    case 8: limbs_cross_imag<2, 8>(stream, beta, N, sft, X, ldx, A); return;
    case 9: limbs_cross_imag<2, 9>(stream, beta, N, sft, X, ldx, A); return;
    case 10: limbs_cross_imag<2, 10>(stream, beta, N, sft, X, ldx, A); return;
    case 11: limbs_cross_imag<2, 11>(stream, beta, N, sft, X, ldx, A); return;
    case 12: limbs_cross_imag<2, 12>(stream, beta, N, sft, X, ldx, A); return;
    default: return;
  } else if (orderA == 3) switch (orderX) {
    case 1: limbs_cross_imag<3, 1>(stream, beta, N, sft, X, ldx, A); return;
    case 2: limbs_cross_imag<3, 2>(stream, beta, N, sft, X, ldx, A); return;
    case 3: limbs_cross_imag<3, 3>(stream, beta, N, sft, X, ldx, A); return;
    case 4: limbs_cross_imag<3, 4>(stream, beta, N, sft, X, ldx, A); return;
    case 5: limbs_cross_imag<3, 5>(stream, beta, N, sft, X, ldx, A); return;
    case 6: limbs_cross_imag<3, 6>(stream, beta, N, sft, X, ldx, A); return;
    case 7: limbs_cross_imag<3, 7>(stream, beta, N, sft, X, ldx, A); return;
    case 8: limbs_cross_imag<3, 8>(stream, beta, N, sft, X, ldx, A); return;
    case 9: limbs_cross_imag<3, 9>(stream, beta, N, sft, X, ldx, A); return;
    case 10: limbs_cross_imag<3, 10>(stream, beta, N, sft, X, ldx, A); return;
    case 11: limbs_cross_imag<3, 11>(stream, beta, N, sft, X, ldx, A); return;
    case 12: limbs_cross_imag<3, 12>(stream, beta, N, sft, X, ldx, A); return;
    default: return;
  }
}
