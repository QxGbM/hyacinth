
#include <internal.hpp>
#include <ext_arith.hpp>

template<int32_t orderX, int32_t orderA>
static __device__ __forceinline__ void accumulate_diag(uint64_t (&a)[orderA], const int32_t* X, int64_t strideX) {
  if constexpr(0 < orderX) { device::add_shifted<0>(a, int64_t(*X)); }
  if constexpr(1 < orderX) { device::add_shifted<16>(a, int64_t(*(X += strideX))); }
  if constexpr(2 < orderX) { device::add_shifted<32>(a, int64_t(*(X += strideX))); }
  if constexpr(3 < orderX) { device::add_shifted<48>(a, int64_t(*(X += strideX))); }
  if constexpr(4 < orderX) { device::add_shifted<64>(a, int64_t(*(X += strideX))); }
  if constexpr(5 < orderX) { device::add_shifted<80>(a, int64_t(*(X += strideX))); }
  if constexpr(6 < orderX) { device::add_shifted<96>(a, int64_t(*(X += strideX))); }
  if constexpr(7 < orderX) { device::add_shifted<112>(a, int64_t(*(X += strideX))); }
  if constexpr(8 < orderX) { device::add_shifted<128>(a, int64_t(*(X += strideX))); }
  if constexpr(9 < orderX) { device::add_shifted<144>(a, int64_t(*(X += strideX))); }
  if constexpr(10 < orderX) { device::add_shifted<160>(a, int64_t(*(X += strideX))); }
  if constexpr(11 < orderX) { device::add_shifted<176>(a, int64_t(*(X += strideX))); }
}

template<int32_t orderA, int32_t orderX, int32_t beta>
__global__ void limbs_diag_real_kernel(int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    uint64_t a[orderA]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)]; 

    if constexpr(beta) {
      const uint64_t* A_in = A;
      if constexpr(0 < orderA) { a[0] = *A_in; }
      if constexpr(1 < orderA) { a[1] = *(A_in += strideA); }
      if constexpr(2 < orderA) { a[2] = *(A_in += strideA); }
    } else {
      if constexpr(0 < orderA) { a[0] = uint64_t(0); }
      if constexpr(1 < orderA) { a[1] = uint64_t(0); }
      if constexpr(2 < orderA) { a[2] = uint64_t(0); }
    }
    accumulate_diag<orderX>(a, &X[y + (x * ldx)], strideX);
    if constexpr(0 < orderA) { *A = a[0]; }
    if constexpr(1 < orderA) { *(A += strideA) = a[1]; }
    if constexpr(2 < orderA) { *(A += strideA) = a[2]; }
  }
}

template <int32_t orderA, int32_t orderX>
static inline void limbs_diag(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, int64_t strideX, uint64_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t(N + 127) >> 7, uint32_t(N), uint32_t(1));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { limbs_diag_real_kernel<orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); } else
  { limbs_diag_real_kernel<orderA, orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, X, ldx64, strideX, A, strideA); }
}

void internal::int8::accumulate_limbs_diag(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, int32_t orderA, uint64_t* A) {
  if (orderA == 1) switch (orderX) {
    case 1: limbs_diag<1, 1>(stream, beta, N, X, ldx, strideX, A); return;
    case 2: limbs_diag<1, 2>(stream, beta, N, X, ldx, strideX, A); return;
    case 3: limbs_diag<1, 3>(stream, beta, N, X, ldx, strideX, A); return;
    case 4: limbs_diag<1, 4>(stream, beta, N, X, ldx, strideX, A); return;
    case 5: limbs_diag<1, 5>(stream, beta, N, X, ldx, strideX, A); return;
    case 6: limbs_diag<1, 6>(stream, beta, N, X, ldx, strideX, A); return;
    case 7: limbs_diag<1, 7>(stream, beta, N, X, ldx, strideX, A); return;
    case 8: limbs_diag<1, 8>(stream, beta, N, X, ldx, strideX, A); return;
    case 9: limbs_diag<1, 9>(stream, beta, N, X, ldx, strideX, A); return;
    case 10: limbs_diag<1, 10>(stream, beta, N, X, ldx, strideX, A); return;
    case 11: limbs_diag<1, 11>(stream, beta, N, X, ldx, strideX, A); return;
    case 12: limbs_diag<1, 12>(stream, beta, N, X, ldx, strideX, A); return;
    default: return;
  } else if (orderA == 2) switch (orderX) {
    case 1: limbs_diag<2, 1>(stream, beta, N, X, ldx, strideX, A); return;
    case 2: limbs_diag<2, 2>(stream, beta, N, X, ldx, strideX, A); return;
    case 3: limbs_diag<2, 3>(stream, beta, N, X, ldx, strideX, A); return;
    case 4: limbs_diag<2, 4>(stream, beta, N, X, ldx, strideX, A); return;
    case 5: limbs_diag<2, 5>(stream, beta, N, X, ldx, strideX, A); return;
    case 6: limbs_diag<2, 6>(stream, beta, N, X, ldx, strideX, A); return;
    case 7: limbs_diag<2, 7>(stream, beta, N, X, ldx, strideX, A); return;
    case 8: limbs_diag<2, 8>(stream, beta, N, X, ldx, strideX, A); return;
    case 9: limbs_diag<2, 9>(stream, beta, N, X, ldx, strideX, A); return;
    case 10: limbs_diag<2, 10>(stream, beta, N, X, ldx, strideX, A); return;
    case 11: limbs_diag<2, 11>(stream, beta, N, X, ldx, strideX, A); return;
    case 12: limbs_diag<2, 12>(stream, beta, N, X, ldx, strideX, A); return;
    default: return;
  } else if (orderA == 3) switch (orderX) {
    case 1: limbs_diag<3, 1>(stream, beta, N, X, ldx, strideX, A); return;
    case 2: limbs_diag<3, 2>(stream, beta, N, X, ldx, strideX, A); return;
    case 3: limbs_diag<3, 3>(stream, beta, N, X, ldx, strideX, A); return;
    case 4: limbs_diag<3, 4>(stream, beta, N, X, ldx, strideX, A); return;
    case 5: limbs_diag<3, 5>(stream, beta, N, X, ldx, strideX, A); return;
    case 6: limbs_diag<3, 6>(stream, beta, N, X, ldx, strideX, A); return;
    case 7: limbs_diag<3, 7>(stream, beta, N, X, ldx, strideX, A); return;
    case 8: limbs_diag<3, 8>(stream, beta, N, X, ldx, strideX, A); return;
    case 9: limbs_diag<3, 9>(stream, beta, N, X, ldx, strideX, A); return;
    case 10: limbs_diag<3, 10>(stream, beta, N, X, ldx, strideX, A); return;
    case 11: limbs_diag<3, 11>(stream, beta, N, X, ldx, strideX, A); return;
    case 12: limbs_diag<3, 12>(stream, beta, N, X, ldx, strideX, A); return;
    default: return;
  }
}
