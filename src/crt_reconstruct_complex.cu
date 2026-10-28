
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>

template <int64_t x> static __device__ __forceinline__ int64_t i32_i64_prod(int32_t y) { return x * int64_t(y); }
template <int32_t orderX, int32_t x, int32_t orderA> static __device__ __forceinline__ void add_pd(uint64_t (&a)[orderA], uint32_t i) {
  using U8CRT2M::mo, U8CRT2M::Constants;
  constexpr int32_t x4 = x * 4; constexpr uint32_t mask_8bits = 0xffu, mask_32bits = 0xffffffffu;
  if constexpr(x < orderX) {
    uint32_t mask_sign = uint32_t(a[orderA - 1] >> 63) + mask_32bits;
    int32_t s = int32_t(device::barrett_reduc<mo[x]>(device::mulx_reduc<0, Constants<orderX>::minv[x], mo[x]>(i & mask_8bits)) - (mo[x] & mask_sign));
    if constexpr(Constants<orderX>::pd[x4]) { device::add_shifted<0>(a, i32_i64_prod<Constants<orderX>::pd[x4]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 1]) { device::add_shifted<55>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 1]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 2]) { device::add_shifted<110>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 2]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 3]) { device::add_shifted<165>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 3]>(s)); }
  } if constexpr(x + 1 < orderX) {
    uint32_t mask_sign = uint32_t(a[orderA - 1] >> 63) + mask_32bits;
    int32_t s = int32_t(device::barrett_reduc<mo[x + 1]>(device::mulx_reduc<0, Constants<orderX>::minv[x + 1], mo[x + 1]>((i >> 8) & mask_8bits)) - (mo[x + 1] & mask_sign));
    if constexpr(Constants<orderX>::pd[x4 + 4]) { device::add_shifted<0>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 4]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 5]) { device::add_shifted<55>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 5]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 6]) { device::add_shifted<110>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 6]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 7]) { device::add_shifted<165>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 7]>(s)); }
  } if constexpr(x + 2 < orderX) {
    uint32_t mask_sign = uint32_t(a[orderA - 1] >> 63) + mask_32bits;
    int32_t s = int32_t(device::barrett_reduc<mo[x + 2]>(device::mulx_reduc<0, Constants<orderX>::minv[x + 2], mo[x + 2]>((i >> 16) & mask_8bits)) - (mo[x + 2] & mask_sign));
    if constexpr(Constants<orderX>::pd[x4 + 8]) { device::add_shifted<0>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 8]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 9]) { device::add_shifted<55>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 9]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 10]) { device::add_shifted<110>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 10]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 11]) { device::add_shifted<165>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 11]>(s)); }
  } if constexpr(x + 3 < orderX) {
    uint32_t mask_sign = uint32_t(a[orderA - 1] >> 63) + mask_32bits;
    int32_t s = int32_t(device::barrett_reduc<mo[x + 3]>(device::mulx_reduc<0, Constants<orderX>::minv[x + 3], mo[x + 3]>(i >> 24)) - (mo[x + 3] & mask_sign));
    if constexpr(Constants<orderX>::pd[x4 + 12]) { device::add_shifted<0>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 12]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 13]) { device::add_shifted<55>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 13]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 14]) { device::add_shifted<110>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 14]>(s)); }
    if constexpr(Constants<orderX>::pd[x4 + 15]) { device::add_shifted<165>(a, i32_i64_prod<Constants<orderX>::pd[x4 + 15]>(s)); }
  }
}

template <int32_t beta, int32_t orderOut, int32_t orderA> static __device__ __forceinline__ void write_a(uint64_t (&a)[orderA], uint64_t* out, int64_t stride) {
  if constexpr(beta) {
    const uint64_t* A_in = out;
    if constexpr(0 < orderOut) { device::add_shifted<0>(a, int64_t(*A_in)); }
    if constexpr(1 < orderOut) { device::add_shifted<63>(a, int64_t(*(A_in += stride))); }
    if constexpr(2 < orderOut) { device::add_shifted<126>(a, int64_t(*(A_in += stride))); }
  }
  if constexpr(0 < orderOut && orderOut < orderA) { a[orderOut - 1] |= a[orderA - 1] & uint64_t(0x8000000000000000llu); }
  if constexpr(0 < orderOut) { *out = a[0]; } if constexpr(1 < orderOut) { *(out += stride) = a[1]; } if constexpr(2 < orderOut) { *(out += stride) = a[2]; }
}

template<int32_t orderX, int32_t orderA, int32_t beta>
__global__ void reconstruct_complex_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x);
  if (y < N) {
    constexpr int64_t p0 = U8CRT2M::Constants<orderX>::p[0], p1 = U8CRT2M::Constants<orderX>::p[1], p2 = U8CRT2M::Constants<orderX>::p[2];
    constexpr int32_t orderP = int32_t(int64_t(0) < p0) + int32_t(int64_t(0) < p1) + int32_t(int64_t(0) < p2), orderAcc = orderA < orderP ? orderP : orderA;
    uint64_t a[orderAcc]{}; int64_t z = int64_t(blockIdx.y); X = &X[y + (z * strideX)]; A = &A[y + (z * strideA)];

    if constexpr(0 < orderX) { add_pd<orderX, 0>(a, *X); }
    if constexpr(4 < orderX) { add_pd<orderX, 4>(a, *(X += N)); }
    if constexpr(8 < orderX) { add_pd<orderX, 8>(a, *(X += N)); }
    if constexpr(12 < orderX) { add_pd<orderX, 12>(a, *(X += N)); }
    if constexpr(16 < orderX) { add_pd<orderX, 16>(a, *(X += N)); }
    if constexpr(20 < orderX) { add_pd<orderX, 20>(a, *(X += N)); }
    if constexpr(24 < orderX) { add_pd<orderX, 24>(a, *(X += N)); }

    uint64_t neg[orderAcc];
    constexpr int64_t neg_p0 = -p0, neg_p1 = -p1, neg_p2 = -p2;
    if constexpr(0 < orderAcc) { neg[0] = a[0]; }
    if constexpr(1 < orderAcc) { neg[1] = a[1]; }
    if constexpr(2 < orderAcc) { neg[2] = a[2]; }
    if (a[orderAcc - 1] >> 63) {
      if constexpr(p0) { device::add_shifted<0>(a, p0); }
      if constexpr(p1) { device::add_shifted<63>(a, p1); }
      if constexpr(p2) { device::add_shifted<126>(a, p2); }
    } else {
      if constexpr(p0) { device::add_shifted<0>(neg, neg_p0); }
      if constexpr(p1) { device::add_shifted<63>(neg, neg_p1); }
      if constexpr(p2) { device::add_shifted<126>(neg, neg_p2); }
    }

    uint64_t sum;
    if constexpr(0 < orderAcc) { sum = a[0] + neg[0]; } else { sum = uint64_t(0); }
    if constexpr(1 < orderAcc) { sum = (sum >> 63) + a[1] + neg[1]; }
    if constexpr(2 < orderAcc) { sum = (sum >> 63) + a[2] + neg[2]; }
    if (!bool(sum >> 63)) {
      if constexpr(0 < orderAcc) { a[0] = neg[0]; }
      if constexpr(1 < orderAcc) { a[1] = neg[1]; }
      if constexpr(2 < orderAcc) { a[2] = neg[2]; }
    }
    write_a<beta, orderA>(a, A, N);
  }
}

template <int32_t orderA, int32_t orderX>
static inline void reconstruct_crt_complex(cudaStream_t stream, int32_t beta, int64_t N, const uint32_t* X, uint64_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t((uint64_t(N) + uint64_t(127)) >> 7), uint32_t(2));
  int64_t strideX = int64_t((orderX + 3) / 4) * N, strideA = int64_t(orderA) * N;
  if (beta) { reconstruct_complex_kernel<orderX, orderA, 1> <<< grid, block_threads, 0, stream >>> (N, X, strideX, A, strideA); } else
  { reconstruct_complex_kernel<orderX, orderA, 0> <<< grid, block_threads, 0, stream >>> (N, X, strideX, A, strideA); }
}

void internal::int8::reconstruct_complex(cudaStream_t stream, int32_t beta, int64_t N, int32_t orderX, const uint32_t* X, int32_t orderA, uint64_t* A) {
  if (orderA == 1) switch (orderX) {
    case 1: reconstruct_crt_complex<1, 1>(stream, beta, N, X, A); return;
    case 2: reconstruct_crt_complex<1, 2>(stream, beta, N, X, A); return;
    case 3: reconstruct_crt_complex<1, 3>(stream, beta, N, X, A); return;
    case 4: reconstruct_crt_complex<1, 4>(stream, beta, N, X, A); return;
    case 5: reconstruct_crt_complex<1, 5>(stream, beta, N, X, A); return;
    case 6: reconstruct_crt_complex<1, 6>(stream, beta, N, X, A); return;
    case 7: reconstruct_crt_complex<1, 7>(stream, beta, N, X, A); return;
    case 8: reconstruct_crt_complex<1, 8>(stream, beta, N, X, A); return;
    case 9: reconstruct_crt_complex<1, 9>(stream, beta, N, X, A); return;
    case 10: reconstruct_crt_complex<1, 10>(stream, beta, N, X, A); return;
    case 11: reconstruct_crt_complex<1, 11>(stream, beta, N, X, A); return;
    case 12: reconstruct_crt_complex<1, 12>(stream, beta, N, X, A); return;
    case 13: reconstruct_crt_complex<1, 13>(stream, beta, N, X, A); return;
    case 14: reconstruct_crt_complex<1, 14>(stream, beta, N, X, A); return;
    case 15: reconstruct_crt_complex<1, 15>(stream, beta, N, X, A); return;
    case 16: reconstruct_crt_complex<1, 16>(stream, beta, N, X, A); return;
    case 17: reconstruct_crt_complex<1, 17>(stream, beta, N, X, A); return;
    case 18: reconstruct_crt_complex<1, 18>(stream, beta, N, X, A); return;
    case 19: reconstruct_crt_complex<1, 19>(stream, beta, N, X, A); return;
    case 20: reconstruct_crt_complex<1, 20>(stream, beta, N, X, A); return;
    case 21: reconstruct_crt_complex<1, 21>(stream, beta, N, X, A); return;
    case 22: reconstruct_crt_complex<1, 22>(stream, beta, N, X, A); return;
    case 23: reconstruct_crt_complex<1, 23>(stream, beta, N, X, A); return;
    case 24: reconstruct_crt_complex<1, 24>(stream, beta, N, X, A); return;
    case 25: reconstruct_crt_complex<1, 25>(stream, beta, N, X, A); return;
    default: return;
  } else if (orderA == 2) switch (orderX) {
    case 1: reconstruct_crt_complex<2, 1>(stream, beta, N, X, A); return;
    case 2: reconstruct_crt_complex<2, 2>(stream, beta, N, X, A); return;
    case 3: reconstruct_crt_complex<2, 3>(stream, beta, N, X, A); return;
    case 4: reconstruct_crt_complex<2, 4>(stream, beta, N, X, A); return;
    case 5: reconstruct_crt_complex<2, 5>(stream, beta, N, X, A); return;
    case 6: reconstruct_crt_complex<2, 6>(stream, beta, N, X, A); return;
    case 7: reconstruct_crt_complex<2, 7>(stream, beta, N, X, A); return;
    case 8: reconstruct_crt_complex<2, 8>(stream, beta, N, X, A); return;
    case 9: reconstruct_crt_complex<2, 9>(stream, beta, N, X, A); return;
    case 10: reconstruct_crt_complex<2, 10>(stream, beta, N, X, A); return;
    case 11: reconstruct_crt_complex<2, 11>(stream, beta, N, X, A); return;
    case 12: reconstruct_crt_complex<2, 12>(stream, beta, N, X, A); return;
    case 13: reconstruct_crt_complex<2, 13>(stream, beta, N, X, A); return;
    case 14: reconstruct_crt_complex<2, 14>(stream, beta, N, X, A); return;
    case 15: reconstruct_crt_complex<2, 15>(stream, beta, N, X, A); return;
    case 16: reconstruct_crt_complex<2, 16>(stream, beta, N, X, A); return;
    case 17: reconstruct_crt_complex<2, 17>(stream, beta, N, X, A); return;
    case 18: reconstruct_crt_complex<2, 18>(stream, beta, N, X, A); return;
    case 19: reconstruct_crt_complex<2, 19>(stream, beta, N, X, A); return;
    case 20: reconstruct_crt_complex<2, 20>(stream, beta, N, X, A); return;
    case 21: reconstruct_crt_complex<2, 21>(stream, beta, N, X, A); return;
    case 22: reconstruct_crt_complex<2, 22>(stream, beta, N, X, A); return;
    case 23: reconstruct_crt_complex<2, 23>(stream, beta, N, X, A); return;
    case 24: reconstruct_crt_complex<2, 24>(stream, beta, N, X, A); return;
    case 25: reconstruct_crt_complex<2, 25>(stream, beta, N, X, A); return;
    default: return;
  } else if (orderA == 3) switch (orderX) {
    case 1: reconstruct_crt_complex<3, 1>(stream, beta, N, X, A); return;
    case 2: reconstruct_crt_complex<3, 2>(stream, beta, N, X, A); return;
    case 3: reconstruct_crt_complex<3, 3>(stream, beta, N, X, A); return;
    case 4: reconstruct_crt_complex<3, 4>(stream, beta, N, X, A); return;
    case 5: reconstruct_crt_complex<3, 5>(stream, beta, N, X, A); return;
    case 6: reconstruct_crt_complex<3, 6>(stream, beta, N, X, A); return;
    case 7: reconstruct_crt_complex<3, 7>(stream, beta, N, X, A); return;
    case 8: reconstruct_crt_complex<3, 8>(stream, beta, N, X, A); return;
    case 9: reconstruct_crt_complex<3, 9>(stream, beta, N, X, A); return;
    case 10: reconstruct_crt_complex<3, 10>(stream, beta, N, X, A); return;
    case 11: reconstruct_crt_complex<3, 11>(stream, beta, N, X, A); return;
    case 12: reconstruct_crt_complex<3, 12>(stream, beta, N, X, A); return;
    case 13: reconstruct_crt_complex<3, 13>(stream, beta, N, X, A); return;
    case 14: reconstruct_crt_complex<3, 14>(stream, beta, N, X, A); return;
    case 15: reconstruct_crt_complex<3, 15>(stream, beta, N, X, A); return;
    case 16: reconstruct_crt_complex<3, 16>(stream, beta, N, X, A); return;
    case 17: reconstruct_crt_complex<3, 17>(stream, beta, N, X, A); return;
    case 18: reconstruct_crt_complex<3, 18>(stream, beta, N, X, A); return;
    case 19: reconstruct_crt_complex<3, 19>(stream, beta, N, X, A); return;
    case 20: reconstruct_crt_complex<3, 20>(stream, beta, N, X, A); return;
    case 21: reconstruct_crt_complex<3, 21>(stream, beta, N, X, A); return;
    case 22: reconstruct_crt_complex<3, 22>(stream, beta, N, X, A); return;
    case 23: reconstruct_crt_complex<3, 23>(stream, beta, N, X, A); return;
    case 24: reconstruct_crt_complex<3, 24>(stream, beta, N, X, A); return;
    case 25: reconstruct_crt_complex<3, 25>(stream, beta, N, X, A); return;
    default: return;
  }
}
