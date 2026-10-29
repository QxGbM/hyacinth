
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
__global__ void crt_extra_real_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t ldx, int64_t strideX, uchar4* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int64_t z = int64_t(blockIdx.z); X = &X[y + (x * ldx) + ((z << 2) * strideX)]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1) + ((z + int64_t(5)) * strideA)];
    uchar4 a; if constexpr(beta) { a = *A; } else { a = uchar4(); }
    if (blockIdx.z) { accumulate_modular_real<orderX, 24>(a, X, strideX); } else { accumulate_modular_real<orderX, 20>(a, X, strideX); }
    *A = a;
  }
}

template <int32_t orderX> static inline void crt_extra_real(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, uint32_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t(N + 127) >> 7, uint32_t(N), uint32_t((orderX - U8CRT2M::len_2m_moduli + 3) / 4));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideX = ldx64 * N64, strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { crt_extra_real_kernel<orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); } else
  { crt_extra_real_kernel<orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); }
}

void internal::int8::accumulate_crt_extra_real(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, uint32_t* A) {
  switch (orderX) {
    case 21: crt_extra_real<21>(stream, beta, N, X, ldx, A); return;
    case 22: crt_extra_real<22>(stream, beta, N, X, ldx, A); return;
    case 23: crt_extra_real<23>(stream, beta, N, X, ldx, A); return;
    case 24: crt_extra_real<24>(stream, beta, N, X, ldx, A); return;
    case 25: crt_extra_real<25>(stream, beta, N, X, ldx, A); return;
    default: return;
  }
}
