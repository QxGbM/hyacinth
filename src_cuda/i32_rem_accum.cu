
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_constants.hpp>

template <int32_t orderX>
__device__ __forceinline__ void loadX(int32_t (&x)[4], const int32_t* X, int64_t strideX) {
  if constexpr(0 < orderX) { x[0] = *X; } else { x[0] = 0; }
  if constexpr(1 < orderX) { x[1] = *(X += strideX); } else { x[1] = 0; }
  if constexpr(2 < orderX) { x[2] = *(X += strideX); } else { x[2] = 0; }
  if constexpr(3 < orderX) { x[3] = *(X += strideX); } else { x[3] = 0; }
}

template <int32_t x>
__device__ __forceinline__ int64_t i32_i64_prod(int32_t y) {
  uint32_t lo = uint32_t(x * y), hi = uint32_t(__mulhi(x, y));
  return int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
}

template <int32_t orderX, int32_t x, int32_t m, uint32_t ORDER>
__device__ __forceinline__ void add_pd(uint64_t (&a)[ORDER], int32_t i) {
  if constexpr(x < orderX) {
    constexpr int32_t x6 = x * 6, pd[6]{ U8CRT::Constants<orderX>::pd[x6], U8CRT::Constants<orderX>::pd[x6 + 1], U8CRT::Constants<orderX>::pd[x6 + 2],
      U8CRT::Constants<orderX>::pd[x6 + 3], U8CRT::Constants<orderX>::pd[x6 + 4], U8CRT::Constants<orderX>::pd[x6 + 5] };

    i = (a[ORDER - 1] >> 63) ? i : (i - m);
    if constexpr(pd[0]) { device::int8::add_shifted(a, i32_i64_prod<pd[0]>(i), uint32_t(0)); }
    if constexpr(pd[1]) { device::int8::add_shifted(a, i32_i64_prod<pd[1]>(i), uint32_t(31)); }
    if constexpr(pd[2]) { device::int8::add_shifted(a, i32_i64_prod<pd[2]>(i), uint32_t(62)); }
    if constexpr(pd[3]) { device::int8::add_shifted(a, i32_i64_prod<pd[3]>(i), uint32_t(93)); }
    if constexpr(pd[4]) { device::int8::add_shifted(a, i32_i64_prod<pd[4]>(i), uint32_t(124)); }
    if constexpr(pd[5]) { device::int8::add_shifted(a, i32_i64_prod<pd[5]>(i), uint32_t(155)); }
  }
}

template<int32_t orderX, int32_t iterX, int32_t orderA, int32_t beta, char mode>
__global__ void i32_crt_accum_kernel(int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  bool pred; if constexpr(mode == 'U') pred = y <= x; else pred = y < N;
  if (pred) {
    constexpr int32_t iterX8 = iterX << 3;
    constexpr int32_t m[8]{ U8CRT::mo[iterX8], U8CRT::mo[iterX8 + 1], U8CRT::mo[iterX8 + 2], U8CRT::mo[iterX8 + 3],
      U8CRT::mo[iterX8 + 4], U8CRT::mo[iterX8 + 5], U8CRT::mo[iterX8 + 6], U8CRT::mo[iterX8 + 7] };

    uint64_t acc[orderA]; int32_t rem[4];
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

    if constexpr(iterX8 < orderX) {
      using U8CRT::mo, U8CRT::Constants;
      constexpr uint64_t m_lo = uint64_t(mo[iterX8]) | (uint64_t(mo[iterX8 + 1]) << 16) | (uint64_t(mo[iterX8 + 2]) << 32) | (uint64_t(mo[iterX8 + 3]) << 48);
      constexpr uint64_t i_lo = uint64_t(Constants<orderX>::minv[iterX8]) | (uint64_t(Constants<orderX>::minv[iterX8 + 1]) << 16) | (uint64_t(Constants<orderX>::minv[iterX8 + 2]) << 32) | (uint64_t(Constants<orderX>::minv[iterX8 + 3]) << 48);
      constexpr uint64_t r_lo = uint64_t(Constants<orderX>::rem_e32[iterX8]) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 1]) << 16) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 2]) << 32) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 3]) << 48);
      loadX<orderX - iterX8>(rem, X, strideX);
      device::int8::crt_recover<m_lo, i_lo, r_lo>(rem);
      add_pd<orderX, iterX8, m[0]>(acc, rem[0]);
      add_pd<orderX, iterX8 + 1, m[1]>(acc, rem[1]);
      add_pd<orderX, iterX8 + 2, m[2]>(acc, rem[2]);
      add_pd<orderX, iterX8 + 3, m[3]>(acc, rem[3]);
    }
  
    if constexpr(iterX8 + 4 < orderX) {
      using U8CRT::mo, U8CRT::Constants;
      constexpr uint64_t m_hi = uint64_t(mo[iterX8 + 4]) | (uint64_t(mo[iterX8 + 5]) << 16) | (uint64_t(mo[iterX8 + 6]) << 32) | (uint64_t(mo[iterX8 + 7]) << 48);
      constexpr uint64_t i_hi = uint64_t(Constants<orderX>::minv[iterX8 + 4]) | (uint64_t(Constants<orderX>::minv[iterX8 + 5]) << 16) | (uint64_t(Constants<orderX>::minv[iterX8 + 6]) << 32) | (uint64_t(Constants<orderX>::minv[iterX8 + 7]) << 48);
      constexpr uint64_t r_hi = uint64_t(Constants<orderX>::rem_e32[iterX8 + 4]) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 5]) << 16) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 6]) << 32) | (uint64_t(Constants<orderX>::rem_e32[iterX8 + 7]) << 48);
      loadX<orderX - (iterX8 + 4)>(rem, &X[strideX << 2], strideX);
      device::int8::crt_recover<m_hi, i_hi, r_hi>(rem);
      add_pd<orderX, iterX8 + 4, m[4]>(acc, rem[0]);
      add_pd<orderX, iterX8 + 5, m[5]>(acc, rem[1]);
      add_pd<orderX, iterX8 + 6, m[6]>(acc, rem[2]);
      add_pd<orderX, iterX8 + 7, m[7]>(acc, rem[3]);
    }

    if (acc[orderA - 1] >> 63) {
      constexpr int64_t p0 = U8CRT::Constants<orderX>::p[0], p1 = U8CRT::Constants<orderX>::p[1], p2 = U8CRT::Constants<orderX>::p[2];
      if constexpr(p0) device::int8::add_shifted(acc, p0, uint32_t(0));
      if constexpr(p1) device::int8::add_shifted(acc, p1, uint32_t(63));
      if constexpr(p2) device::int8::add_shifted(acc, p2, uint32_t(126));
    }

    if constexpr(0 < orderA) { *A = acc[0]; }
    if constexpr(1 < orderA) { *(A += strideA) = acc[1]; }
    if constexpr(2 < orderA) { *(A += strideA) = acc[2]; }
  }
}

template <int32_t orderX, int32_t iter, int32_t orderA>
inline void crt_acc_dispatcher(cudaStream_t stream, char mode, int32_t beta, int64_t N, const int32_t* X, int64_t ldx, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideX = ldx * N, strideA = N * N + N;
  if (mode == 'U' && beta == 0) { i32_crt_accum_kernel<orderX, iter, orderA, 0, 'U'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'U' && beta == 1) { i32_crt_accum_kernel<orderX, iter, orderA, 1, 'U'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'A' && beta == 0) { i32_crt_accum_kernel<orderX, iter, orderA, 0, 'A'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); } else
  if (mode == 'A' && beta == 1) { i32_crt_accum_kernel<orderX, iter, orderA, 1, 'A'> <<< grid, block_threads, 0, stream >>> (N, X, ldx, strideX, A, strideA); }
}

template <int32_t orderA>
inline void crt_acc_dispatcher(cudaStream_t stream, char mode, int32_t beta, int64_t N, int32_t orderX, int32_t iter, const int32_t* X, int64_t ldx, uint64_t* A) {
  if (iter == 0) switch (orderX) {
    case 2: crt_acc_dispatcher<2, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 3: crt_acc_dispatcher<3, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 4: crt_acc_dispatcher<4, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 5: crt_acc_dispatcher<5, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 6: crt_acc_dispatcher<6, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 7: crt_acc_dispatcher<7, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 8: crt_acc_dispatcher<8, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 9: crt_acc_dispatcher<9, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 10: crt_acc_dispatcher<10, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 11: crt_acc_dispatcher<11, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 12: crt_acc_dispatcher<12, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 13: crt_acc_dispatcher<13, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 14: crt_acc_dispatcher<14, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 15: crt_acc_dispatcher<15, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 16: crt_acc_dispatcher<16, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 17: crt_acc_dispatcher<17, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 18: crt_acc_dispatcher<18, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 19: crt_acc_dispatcher<19, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 20: crt_acc_dispatcher<20, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 21: crt_acc_dispatcher<21, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 22: crt_acc_dispatcher<22, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 23: crt_acc_dispatcher<23, 0, orderA>(stream, mode, beta, N, X, ldx, A); return;
    default: return;
  }
  else if (iter == 1) switch (orderX) {
    case 9: crt_acc_dispatcher<9, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 10: crt_acc_dispatcher<10, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 11: crt_acc_dispatcher<11, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 12: crt_acc_dispatcher<12, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 13: crt_acc_dispatcher<13, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 14: crt_acc_dispatcher<14, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 15: crt_acc_dispatcher<15, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 16: crt_acc_dispatcher<16, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 17: crt_acc_dispatcher<17, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 18: crt_acc_dispatcher<18, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 19: crt_acc_dispatcher<19, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 20: crt_acc_dispatcher<20, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 21: crt_acc_dispatcher<21, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 22: crt_acc_dispatcher<22, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 23: crt_acc_dispatcher<23, 1, orderA>(stream, mode, beta, N, X, ldx, A); return;
    default: return;
  }
  else if (iter == 2) switch (orderX) {
    case 17: crt_acc_dispatcher<17, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 18: crt_acc_dispatcher<18, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 19: crt_acc_dispatcher<19, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 20: crt_acc_dispatcher<20, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 21: crt_acc_dispatcher<21, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 22: crt_acc_dispatcher<22, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    case 23: crt_acc_dispatcher<23, 2, orderA>(stream, mode, beta, N, X, ldx, A); return;
    default: return;
  }
}

void internal::int8::accumulate_remainder_i32tensor(cudaStream_t stream, char mode, int32_t beta, int32_t N, int32_t orderX, int32_t iter, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A) {
  switch (orderA) {
    case 1: crt_acc_dispatcher<1>(stream, mode, beta, int64_t(N), orderX, iter, X, int64_t(ldx), A); return;
    case 2: crt_acc_dispatcher<2>(stream, mode, beta, int64_t(N), orderX, iter, X, int64_t(ldx), A); return;
    case 3: crt_acc_dispatcher<3>(stream, mode, beta, int64_t(N), orderX, iter, X, int64_t(ldx), A); return;
    default: return;
  }
}
