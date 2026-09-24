
#include <internal.hpp>
#include <ext_arith.hpp>

template<int32_t sft, int32_t sft_iter, int32_t orderX, int32_t orderA>
__device__ __forceinline__ void accumulate(uint64_t (&a)[orderA], const int32_t* X, int64_t strideX) {
  if constexpr(0 < orderX) { device::add_shifted<sft>(a, int64_t(*X)); }
  if constexpr(1 < orderX) { device::add_shifted<sft + sft_iter>(a, int64_t(*(X += strideX))); }
  if constexpr(2 < orderX) { device::add_shifted<sft + (sft_iter * 2)>(a, int64_t(*(X += strideX))); }
  if constexpr(3 < orderX) { device::add_shifted<sft + (sft_iter * 3)>(a, int64_t(*(X += strideX))); }
  if constexpr(4 < orderX) { device::add_shifted<sft + (sft_iter * 4)>(a, int64_t(*(X += strideX))); }
  if constexpr(5 < orderX) { device::add_shifted<sft + (sft_iter * 5)>(a, int64_t(*(X += strideX))); }
  if constexpr(6 < orderX) { device::add_shifted<sft + (sft_iter * 6)>(a, int64_t(*(X += strideX))); }
  if constexpr(7 < orderX) { device::add_shifted<sft + (sft_iter * 7)>(a, int64_t(*(X += strideX))); }
  if constexpr(8 < orderX) { device::add_shifted<sft + (sft_iter * 8)>(a, int64_t(*(X += strideX))); }
  if constexpr(9 < orderX) { device::add_shifted<sft + (sft_iter * 9)>(a, int64_t(*(X += strideX))); }
  if constexpr(10 < orderX) { device::add_shifted<sft + (sft_iter * 10)>(a, int64_t(*(X += strideX))); }
  if constexpr(11 < orderX) { device::add_shifted<sft + (sft_iter * 11)>(a, int64_t(*(X += strideX))); }
}

template<int32_t sft, int32_t orderA, int32_t orderX, int32_t beta, char mode>
__global__ void i32_accum_kernel(int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U' || mode == 'T') { pred = y <= x; } else { pred = y < N; }
  if (pred) {
    int64_t x = int64_t(blockIdx.y);
    uint64_t acc[orderA]; A = &A[y + x * N]; X = &X[y + x * ldx];

    if constexpr(beta) {
      if constexpr(0 < orderA) { acc[0] = A[0]; }
      if constexpr(1 < orderA) { acc[1] = A[strideA]; }
      if constexpr(2 < orderA) { acc[2] = A[strideA + strideA]; }
      if constexpr(mode == 'T') {
        uint64_t* AT = &A[(x - y) * (int64_t(1) - N)];
        if constexpr(0 < orderA) { device::add_shifted<0>(acc, int64_t(*AT)); }
        if constexpr(1 < orderA) { device::add_shifted<63>(acc, int64_t(*(AT += strideA))); }
        if constexpr(2 < orderA) { device::add_shifted<126>(acc, int64_t(*(AT += strideA))); }
      }
    }
    else {
      if constexpr(0 < orderA) { acc[0] = uint64_t(0); }
      if constexpr(1 < orderA) { acc[1] = uint64_t(0); }
      if constexpr(2 < orderA) { acc[2] = uint64_t(0); }
    }
    accumulate<sft, (mode == 'U' || mode == 'T') ? 16 : 8, orderX>(acc, X, strideX);
    if constexpr(0 < orderA) { *A = acc[0]; }
    if constexpr(1 < orderA) { *(A += strideA) = acc[1]; }
    if constexpr(2 < orderA) { *(A += strideA) = acc[2]; }
  }
}

template <int32_t orderA, int32_t orderX>
inline void acc_dispatcher(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t sft, const int32_t* X, int32_t ldx, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = N64 * N64;
  if ((mode == 'U' || mode == 'T') && beta == 0) { i32_accum_kernel<0, orderA, orderX, 0, 'U'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  if (mode == 'U' && beta == 1) { i32_accum_kernel<0, orderA, orderX, 1, 'U'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  if (mode == 'T' && beta == 1) { i32_accum_kernel<0, orderA, orderX, 1, 'T'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  if (mode == 'A' && beta == 0 && sft == 0) { i32_accum_kernel<0, orderA, orderX, 0, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  if (mode == 'A' && beta == 0 && sft == 8) { i32_accum_kernel<8, orderA, orderX, 0, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  if (mode == 'A' && beta == 1) switch (sft) {
    case 0: i32_accum_kernel<0, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 8: i32_accum_kernel<8, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 16: i32_accum_kernel<16, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 24: i32_accum_kernel<24, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 32: i32_accum_kernel<32, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 40: i32_accum_kernel<40, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 48: i32_accum_kernel<48, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 56: i32_accum_kernel<56, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 64: i32_accum_kernel<64, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 72: i32_accum_kernel<72, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 80: i32_accum_kernel<80, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 88: i32_accum_kernel<88, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    case 96: i32_accum_kernel<96, orderA, orderX, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); return;
    default: return;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t sft, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A) {
  if (orderA == 1) switch (orderX) {
    case 1: acc_dispatcher<1, 1>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 2: acc_dispatcher<1, 2>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 3: acc_dispatcher<1, 3>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 4: acc_dispatcher<1, 4>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 5: acc_dispatcher<1, 5>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 6: acc_dispatcher<1, 6>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 7: acc_dispatcher<1, 7>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 8: acc_dispatcher<1, 8>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 9: acc_dispatcher<1, 9>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 10: acc_dispatcher<1, 10>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 11: acc_dispatcher<1, 11>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 12: acc_dispatcher<1, 12>(stream, mode, beta, N, sft, X, ldx, A); return;
    default: return;
  } else if (orderA == 2) switch (orderX) {
    case 1: acc_dispatcher<2, 1>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 2: acc_dispatcher<2, 2>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 3: acc_dispatcher<2, 3>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 4: acc_dispatcher<2, 4>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 5: acc_dispatcher<2, 5>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 6: acc_dispatcher<2, 6>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 7: acc_dispatcher<2, 7>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 8: acc_dispatcher<2, 8>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 9: acc_dispatcher<2, 9>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 10: acc_dispatcher<2, 10>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 11: acc_dispatcher<2, 11>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 12: acc_dispatcher<2, 12>(stream, mode, beta, N, sft, X, ldx, A); return;
    default: return;
  } else if (orderA == 3) switch (orderX) {
    case 1: acc_dispatcher<3, 1>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 2: acc_dispatcher<3, 2>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 3: acc_dispatcher<3, 3>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 4: acc_dispatcher<3, 4>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 5: acc_dispatcher<3, 5>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 6: acc_dispatcher<3, 6>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 7: acc_dispatcher<3, 7>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 8: acc_dispatcher<3, 8>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 9: acc_dispatcher<3, 9>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 10: acc_dispatcher<3, 10>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 11: acc_dispatcher<3, 11>(stream, mode, beta, N, sft, X, ldx, A); return;
    case 12: acc_dispatcher<3, 12>(stream, mode, beta, N, sft, X, ldx, A); return;
    default: return;
  }
}
