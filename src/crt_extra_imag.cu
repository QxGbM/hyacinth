
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
    r.z = uint8_t(device::barrett_reduc<mo>(uint32_t(r.z) + device::mulx_reduc<1, mul, mo>(a)));
    i.z = uint8_t(device::barrett_reduc<mo>(uint32_t(i.z) + device::mulx_reduc<1, s, mo>(a)));
  } if constexpr(x + 3 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 3], mul = mo - uint32_t(1), s = sign ? mul : 1;
    uint32_t a = *(X += strideX);
    r.w = uint8_t(device::barrett_reduc<mo>(uint32_t(r.w) + device::mulx_reduc<1, mul, mo>(a)));
    i.w = uint8_t(device::barrett_reduc<mo>(uint32_t(i.w) + device::mulx_reduc<1, s, mo>(a)));
  }
}

template<int32_t orderX, int32_t beta>
__global__ void crt_extra_imag_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t ldx, int64_t strideX, uchar4* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int64_t z = int64_t(blockIdx.z); X = &X[(z << 2) * strideX]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1) + ((z + int64_t(5)) * strideA)];
    constexpr int64_t panelsA = int64_t((orderX + 3) / 4);
    uchar4* AImag = &A[panelsA * strideA], r = *A, i; if constexpr(beta) { i = *AImag; } else { i = uchar4(); }
    if (blockIdx.z) {
      accumulate_modular_imaginary<0, orderX, 24>(r, i, &X[y + (x * ldx)], strideX);
      accumulate_modular_imaginary<1, orderX, 24>(r, i, &X[x + (y * ldx)], strideX);
    } else {
      accumulate_modular_imaginary<0, orderX, 20>(r, i, &X[y + (x * ldx)], strideX);
      accumulate_modular_imaginary<1, orderX, 20>(r, i, &X[x + (y * ldx)], strideX);
    }
    *A = r; *AImag = i;
  }
}

template <int32_t orderX> static inline void crt_extra_imag(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, uint32_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t(N + 127) >> 7, uint32_t(N), uint32_t((orderX - 17) / 4));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { crt_extra_imag_kernel<orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); } else
  { crt_extra_imag_kernel<orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); }
}

void internal::int8::accumulate_crt_extra_imag(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, uint32_t* A) {
  switch (orderX) {
    case 21: crt_extra_imag<21>(stream, beta, N, X, ldx, A); return;
    case 22: crt_extra_imag<22>(stream, beta, N, X, ldx, A); return;
    case 23: crt_extra_imag<23>(stream, beta, N, X, ldx, A); return;
    case 24: crt_extra_imag<24>(stream, beta, N, X, ldx, A); return;
    case 25: crt_extra_imag<25>(stream, beta, N, X, ldx, A); return;
    default: return;
  }
}
