
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>

template<int32_t orderX, int32_t x>
static __device__ __forceinline__ void accumulate_modular_real(uchar4& a, const uint32_t* X, int64_t strideX) {
  if constexpr(x < orderX) { constexpr uint32_t mo = U8CRT2M::mo[x]; a.x = uint8_t(device::barrett_reduc<mo>(uint32_t(a.x) + device::mulx_reduc<1, 1, mo>(*X))); }
  if constexpr(x + 1 < orderX) { constexpr uint32_t mo = U8CRT2M::mo[x + 1]; a.y = uint8_t(device::barrett_reduc<mo>(uint32_t(a.y) + device::mulx_reduc<1, 1, mo>(*(X += strideX)))); }
  if constexpr(x + 2 < orderX) { constexpr uint32_t mo = U8CRT2M::mo[x + 2]; a.z = uint8_t(device::barrett_reduc<mo>(uint32_t(a.z) + device::mulx_reduc<1, 1, mo>(*(X += strideX)))); }
  if constexpr(x + 3 < orderX) { constexpr uint32_t mo = U8CRT2M::mo[x + 3]; a.w = uint8_t(device::barrett_reduc<mo>(uint32_t(a.w) + device::mulx_reduc<1, 1, mo>(*(X += strideX)))); }
}

template<int32_t orderX, int32_t beta>
__global__ void crt_diag_complex_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t ldx, int64_t strideX, uchar4* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int64_t z = int64_t(blockIdx.z); X = &X[y + (x * N) + ((z << 2) * strideX)]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1) + (z * strideA)];

    uchar4 a; if constexpr(beta) { a = *A; } else { a = uchar4(); }
    switch (int32_t(blockIdx.z)) {
      case 0: accumulate_modular_real<orderX, 0>(a, X, strideX); break;
      case 1: accumulate_modular_real<orderX, 4>(a, X, strideX); break;
      case 2: accumulate_modular_real<orderX, 8>(a, X, strideX); break;
      case 3: accumulate_modular_real<orderX, 12>(a, X, strideX); break;
      case 4: accumulate_modular_real<orderX, 16>(a, X, strideX); break;
      case 5: accumulate_modular_real<orderX, 20>(a, X, strideX); break;
    }
    *A = a;
  }
}

template <int32_t orderX> static inline void crt_diag_complex(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, uint32_t* A) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t((orderX + 3) / 4));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { crt_diag_complex_kernel<orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); } else
  { crt_diag_complex_kernel<orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); }
}

void internal::int8::accumulate_crt_diag_complex(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, uint32_t* A) {
  switch (orderX) {
    case 1: crt_diag_complex<1>(stream, beta, N, X, ldx, A); return;
    case 2: crt_diag_complex<2>(stream, beta, N, X, ldx, A); return;
    case 3: crt_diag_complex<3>(stream, beta, N, X, ldx, A); return;
    case 4: crt_diag_complex<4>(stream, beta, N, X, ldx, A); return;
    case 5: crt_diag_complex<5>(stream, beta, N, X, ldx, A); return;
    case 6: crt_diag_complex<6>(stream, beta, N, X, ldx, A); return;
    case 7: crt_diag_complex<7>(stream, beta, N, X, ldx, A); return;
    case 8: crt_diag_complex<8>(stream, beta, N, X, ldx, A); return;
    case 9: crt_diag_complex<9>(stream, beta, N, X, ldx, A); return;
    case 10: crt_diag_complex<10>(stream, beta, N, X, ldx, A); return;
    case 11: crt_diag_complex<11>(stream, beta, N, X, ldx, A); return;
    case 12: crt_diag_complex<12>(stream, beta, N, X, ldx, A); return;
    case 13: crt_diag_complex<13>(stream, beta, N, X, ldx, A); return;
    case 14: crt_diag_complex<14>(stream, beta, N, X, ldx, A); return;
    case 15: crt_diag_complex<15>(stream, beta, N, X, ldx, A); return;
    case 16: crt_diag_complex<16>(stream, beta, N, X, ldx, A); return;
    case 17: crt_diag_complex<17>(stream, beta, N, X, ldx, A); return;
    case 18: crt_diag_complex<18>(stream, beta, N, X, ldx, A); return;
    case 19: crt_diag_complex<19>(stream, beta, N, X, ldx, A); return;
    case 20: crt_diag_complex<20>(stream, beta, N, X, ldx, A); return;
    case 21: crt_diag_complex<21>(stream, beta, N, X, ldx, A); return;
    case 22: crt_diag_complex<22>(stream, beta, N, X, ldx, A); return;
    case 23: crt_diag_complex<23>(stream, beta, N, X, ldx, A); return;
    case 24: crt_diag_complex<24>(stream, beta, N, X, ldx, A); return;
    case 25: crt_diag_complex<25>(stream, beta, N, X, ldx, A); return;
    default: return;
  }
}
