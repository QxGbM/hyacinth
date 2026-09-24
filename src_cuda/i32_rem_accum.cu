
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>

template <int32_t x>
__device__ __forceinline__ int64_t i32_i64_prod(int32_t y) {
  uint32_t lo = uint32_t(x * y), hi = uint32_t(__mulhi(x, y));
  return int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
}

template <int32_t orderX, int32_t x, int32_t orderA>
__device__ __forceinline__ void add_pd(uint64_t (&a)[orderA], int32_t i) {
  using U8CRT::Constants;
  constexpr int32_t x6 = x * 6, pd[6]{ Constants<orderX>::pd[x6], Constants<orderX>::pd[x6 + 1], Constants<orderX>::pd[x6 + 2],
    Constants<orderX>::pd[x6 + 3], Constants<orderX>::pd[x6 + 4], Constants<orderX>::pd[x6 + 5] };
  constexpr uint32_t MO = U8CRT::mo[x], MINV = Constants<orderX>::minv[x], R32 = Constants<orderX>::rem_e32[x];

  uint32_t u = uint32_t(i), mask_sign = uint32_t(a[orderA - 1] >> 63) + 0x7fffffffu;
  i = int32_t(device::barrett_reduc<MO>(device::mulx_reduc<MINV, MO>(u) + ((-(u >> 31)) & R32)) - (MO & mask_sign));
  if constexpr(pd[0]) { device::add_shifted<0>(a, i32_i64_prod<pd[0]>(i)); }
  if constexpr(pd[1]) { device::add_shifted<31>(a, i32_i64_prod<pd[1]>(i)); }
  if constexpr(pd[2]) { device::add_shifted<62>(a, i32_i64_prod<pd[2]>(i)); }
  if constexpr(pd[3]) { device::add_shifted<93>(a, i32_i64_prod<pd[3]>(i)); }
  if constexpr(pd[4]) { device::add_shifted<124>(a, i32_i64_prod<pd[4]>(i)); }
  if constexpr(pd[5]) { device::add_shifted<155>(a, i32_i64_prod<pd[5]>(i)); }
}

template<int32_t orderX, int32_t orderA, int32_t beta, char mode>
__global__ void i32_crt_accum_kernel(int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U') { pred = y <= x; } else { pred = y < N; }
  if (pred) {
    uint64_t acc[orderA];
    A = &A[y + x * N]; X = &X[y + x * ldx];

    if constexpr(beta) {
      if constexpr(0 < orderA) { acc[0] = A[0]; }
      if constexpr(1 < orderA) { acc[1] = A[strideA]; }
      if constexpr(2 < orderA) { acc[2] = A[strideA + strideA]; }
    }
    else {
      if constexpr(0 < orderA) { acc[0] = uint64_t(0); }
      if constexpr(1 < orderA) { acc[1] = uint64_t(0); }
      if constexpr(2 < orderA) { acc[2] = uint64_t(0); }
    }

    if constexpr(0 < orderX) { add_pd<orderX, 0>(acc, *X); }
    if constexpr(1 < orderX) { add_pd<orderX, 1>(acc, *(X += strideX)); }
    if constexpr(2 < orderX) { add_pd<orderX, 2>(acc, *(X += strideX)); }
    if constexpr(3 < orderX) { add_pd<orderX, 3>(acc, *(X += strideX)); }
    if constexpr(4 < orderX) { add_pd<orderX, 4>(acc, *(X += strideX)); }
    if constexpr(5 < orderX) { add_pd<orderX, 5>(acc, *(X += strideX)); }
    if constexpr(6 < orderX) { add_pd<orderX, 6>(acc, *(X += strideX)); }
    if constexpr(7 < orderX) { add_pd<orderX, 7>(acc, *(X += strideX)); }
    if constexpr(8 < orderX) { add_pd<orderX, 8>(acc, *(X += strideX)); }
    if constexpr(9 < orderX) { add_pd<orderX, 9>(acc, *(X += strideX)); }
    if constexpr(10 < orderX) { add_pd<orderX, 10>(acc, *(X += strideX)); }
    if constexpr(11 < orderX) { add_pd<orderX, 11>(acc, *(X += strideX)); }
    if constexpr(12 < orderX) { add_pd<orderX, 12>(acc, *(X += strideX)); }
    if constexpr(13 < orderX) { add_pd<orderX, 13>(acc, *(X += strideX)); }
    if constexpr(14 < orderX) { add_pd<orderX, 14>(acc, *(X += strideX)); }
    if constexpr(15 < orderX) { add_pd<orderX, 15>(acc, *(X += strideX)); }
    if constexpr(16 < orderX) { add_pd<orderX, 16>(acc, *(X += strideX)); }
    if constexpr(17 < orderX) { add_pd<orderX, 17>(acc, *(X += strideX)); }
    if constexpr(18 < orderX) { add_pd<orderX, 18>(acc, *(X += strideX)); }
    if constexpr(19 < orderX) { add_pd<orderX, 19>(acc, *(X += strideX)); }
    if constexpr(20 < orderX) { add_pd<orderX, 20>(acc, *(X += strideX)); }
    if constexpr(21 < orderX) { add_pd<orderX, 21>(acc, *(X += strideX)); }
    if constexpr(22 < orderX) { add_pd<orderX, 22>(acc, *(X += strideX)); }

    if (acc[orderA - 1] >> 63) {
      constexpr int64_t p0 = U8CRT::Constants<orderX>::p[0], p1 = U8CRT::Constants<orderX>::p[1], p2 = U8CRT::Constants<orderX>::p[2];
      if constexpr(p0) { device::add_shifted<0>(acc, p0); }
      if constexpr(p1) { device::add_shifted<63>(acc, p1); }
      if constexpr(p2) { device::add_shifted<126>(acc, p2); }
    }

    if constexpr(0 < orderA) { *A = acc[0]; }
    if constexpr(1 < orderA) { *(A += strideA) = acc[1]; }
    if constexpr(2 < orderA) { *(A += strideA) = acc[2]; }
  }
}

template <int32_t orderX, int32_t orderA>
inline void crt_acc_dispatcher(cudaStream_t stream, char mode, int32_t beta, int64_t N, const int32_t* X, int64_t ldx, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideX = ldx * N, strideA = N * N;
  if (mode == 'U' && beta == 0) { i32_crt_accum_kernel<orderX, orderA, 0, 'U'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'U' && beta == 1) { i32_crt_accum_kernel<orderX, orderA, 1, 'U'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'A' && beta == 0) { i32_crt_accum_kernel<orderX, orderA, 0, 'A'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'A' && beta == 1) { i32_crt_accum_kernel<orderX, orderA, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); }
}

template <int32_t orderA>
inline void crt_acc_dispatcher(cudaStream_t stream, char mode, int32_t beta, int64_t N, int32_t orderX, const int32_t* X, int64_t ldx, uint64_t* A) {
  switch (orderX) {
    case 2: crt_acc_dispatcher<2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 3: crt_acc_dispatcher<3, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 4: crt_acc_dispatcher<4, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 5: crt_acc_dispatcher<5, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 6: crt_acc_dispatcher<6, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 7: crt_acc_dispatcher<7, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 8: crt_acc_dispatcher<8, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 9: crt_acc_dispatcher<9, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 10: crt_acc_dispatcher<10, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 11: crt_acc_dispatcher<11, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 12: crt_acc_dispatcher<12, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 13: crt_acc_dispatcher<13, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 14: crt_acc_dispatcher<14, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 15: crt_acc_dispatcher<15, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 16: crt_acc_dispatcher<16, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 17: crt_acc_dispatcher<17, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 18: crt_acc_dispatcher<18, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 19: crt_acc_dispatcher<19, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 20: crt_acc_dispatcher<20, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 21: crt_acc_dispatcher<21, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 22: crt_acc_dispatcher<22, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 23: crt_acc_dispatcher<23, orderA>(stream, mode, beta, N, X, ldx, A); return;
    default: return;
  }
}

void internal::int8::accumulate_remainder_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A) {
  switch (orderA) {
    case 1: crt_acc_dispatcher<1>(stream, mode, beta, int64_t(N), orderX, X, int64_t(ldx), A); return;
    case 2: crt_acc_dispatcher<2>(stream, mode, beta, int64_t(N), orderX, X, int64_t(ldx), A); return;
    case 3: crt_acc_dispatcher<3>(stream, mode, beta, int64_t(N), orderX, X, int64_t(ldx), A); return;
    default: return;
  }
}
