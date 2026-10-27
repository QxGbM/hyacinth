
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>

template<int32_t sign, int32_t orderX, int32_t x>
static __device__ __forceinline__ void accumulate_modular_imaginary(uchar4& r, uchar4& i, const uint32_t* X, int64_t strideX) {
  if constexpr(x < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x], mul = mo - uint32_t(1), s = sign ? mul : 1;
    uint32_t a = *X;
    r.x = uint8_t(device::barrett_reduc<mo>(uint32_t(r.x) + device::mulx_reduc<1, mul, mo>(a)));
    i.x = uint8_t(device::barrett_reduc<mo>(uint32_t(i.x) + device::mulx_reduc<1, s, mo>(a)));
  } if constexpr(x + 1 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 1], mul = mo - uint32_t(1), s = sign ? mul : 1;
    uint32_t a = *(X += strideX);
    r.y = uint8_t(device::barrett_reduc<mo>(uint32_t(r.y) + device::mulx_reduc<1, mul, mo>(a)));
    i.y = uint8_t(device::barrett_reduc<mo>(uint32_t(i.y) + device::mulx_reduc<1, s, mo>(a)));
  } if constexpr(x + 2 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 2], mul = mo - uint32_t(1), s = sign ? mul : 1;
    uint32_t a = *(X += strideX);
    r.z = uint8_t(device::barrett_reduc<mo>(uint32_t(r.y) + device::mulx_reduc<1, mul, mo>(a)));
    i.z = uint8_t(device::barrett_reduc<mo>(uint32_t(i.y) + device::mulx_reduc<1, s, mo>(a)));
  } if constexpr(x + 3 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 3], mul = mo - uint32_t(1), s = sign ? mul : 1;
    uint32_t a = *(X += strideX);
    r.w = uint8_t(device::barrett_reduc<mo>(uint32_t(r.y) + device::mulx_reduc<1, mul, mo>(a)));
    i.w = uint8_t(device::barrett_reduc<mo>(uint32_t(i.y) + device::mulx_reduc<1, s, mo>(a)));
  }
}

template<int32_t orderX, int32_t beta>
__global__ void crt_cross_imag_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t ldx, int64_t strideX, uchar4* __restrict__ R, uchar4* __restrict__ I, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int64_t z = int64_t(blockIdx.z), offsetA = y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1) + (z * strideA);
    X = &X[(z << 2) * strideX]; R = &R[offsetA]; I = &I[offsetA];

    uchar4 r = *R, i; if constexpr(beta) { i = *I; } else { i = uchar4(); }
    switch (int32_t(blockIdx.z)) {
      case 0: accumulate_modular_imaginary<0, orderX, 0>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 0>(r, i, &X[x + (y * N)], strideX); break;
      case 1: accumulate_modular_imaginary<0, orderX, 4>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 4>(r, i, &X[x + (y * N)], strideX); break;
      case 2: accumulate_modular_imaginary<0, orderX, 8>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 8>(r, i, &X[x + (y * N)], strideX); break;
      case 3: accumulate_modular_imaginary<0, orderX, 12>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 12>(r, i, &X[x + (y * N)], strideX); break;
      case 4: accumulate_modular_imaginary<0, orderX, 16>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 16>(r, i, &X[x + (y * N)], strideX); break;
      case 5: accumulate_modular_imaginary<0, orderX, 20>(r, i, &X[y + (x * N)], strideX); accumulate_modular_imaginary<1, orderX, 20>(r, i, &X[x + (y * N)], strideX); break;
    }
    *R = r; *I = i;
  }
}

template <int32_t orderX> static inline void crt_cross_imag(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, uint32_t* R, uint32_t* I) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t((orderX + 3) / 4));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { crt_cross_imag_kernel<orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)R, (uchar4*)I, strideA); } else
  { crt_cross_imag_kernel<orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)R, (uchar4*)I, strideA); }
}

void internal::int8::accumulate_crt_cross_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, uint32_t* R, uint32_t* I) {
  switch (orderX) {
    case 1: crt_cross_imag<1>(stream, beta, N, X, ldx, R, I); return;
    case 2: crt_cross_imag<2>(stream, beta, N, X, ldx, R, I); return;
    case 3: crt_cross_imag<3>(stream, beta, N, X, ldx, R, I); return;
    case 4: crt_cross_imag<4>(stream, beta, N, X, ldx, R, I); return;
    case 5: crt_cross_imag<5>(stream, beta, N, X, ldx, R, I); return;
    case 6: crt_cross_imag<6>(stream, beta, N, X, ldx, R, I); return;
    case 7: crt_cross_imag<7>(stream, beta, N, X, ldx, R, I); return;
    case 8: crt_cross_imag<8>(stream, beta, N, X, ldx, R, I); return;
    case 9: crt_cross_imag<9>(stream, beta, N, X, ldx, R, I); return;
    case 10: crt_cross_imag<10>(stream, beta, N, X, ldx, R, I); return;
    case 11: crt_cross_imag<11>(stream, beta, N, X, ldx, R, I); return;
    case 12: crt_cross_imag<12>(stream, beta, N, X, ldx, R, I); return;
    case 13: crt_cross_imag<13>(stream, beta, N, X, ldx, R, I); return;
    case 14: crt_cross_imag<14>(stream, beta, N, X, ldx, R, I); return;
    case 15: crt_cross_imag<15>(stream, beta, N, X, ldx, R, I); return;
    case 16: crt_cross_imag<16>(stream, beta, N, X, ldx, R, I); return;
    case 17: crt_cross_imag<17>(stream, beta, N, X, ldx, R, I); return;
    case 18: crt_cross_imag<18>(stream, beta, N, X, ldx, R, I); return;
    case 19: crt_cross_imag<19>(stream, beta, N, X, ldx, R, I); return;
    case 20: crt_cross_imag<20>(stream, beta, N, X, ldx, R, I); return;
    case 21: crt_cross_imag<21>(stream, beta, N, X, ldx, R, I); return;
    case 22: crt_cross_imag<22>(stream, beta, N, X, ldx, R, I); return;
    case 23: crt_cross_imag<23>(stream, beta, N, X, ldx, R, I); return;
    case 24: crt_cross_imag<24>(stream, beta, N, X, ldx, R, I); return;
    default: return;
  }
}
