
#include <internal.hpp>
#include <ext_arith.hpp>
#include <crt_constants.hpp>

template<int32_t orderX, int32_t x>
static __device__ __forceinline__ void accumulate_modular_rotate(uchar4& r, uchar4& i, const uint32_t* P, const uint32_t* Q, int64_t strideX) {
  if constexpr(x < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x], minv = U8CRT2M::im_inv[x], inv2 = (mo + 1) / 2;
    uint32_t p = *P, q = *Q;
    r.x = uint8_t(device::barrett_reduc<mo>(uint32_t(r.x) + device::mulx_reduc<1, inv2, mo>(p) + device::mulx_reduc<1, inv2, mo>(q)));
    i.x = uint8_t(device::barrett_reduc<mo>(uint32_t(i.x) + device::mulx_reduc<1, minv, mo>(p) + device::mulx_reduc<1, mo - minv, mo>(q)));
  } if constexpr(x + 1 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 1], minv = U8CRT2M::im_inv[x + 1], inv2 = (mo + 1) / 2;
    uint32_t p = *(P += strideX), q = *(Q += strideX);
    r.y = uint8_t(device::barrett_reduc<mo>(uint32_t(r.y) + device::mulx_reduc<1, inv2, mo>(p) + device::mulx_reduc<1, inv2, mo>(q)));
    i.y = uint8_t(device::barrett_reduc<mo>(uint32_t(i.y) + device::mulx_reduc<1, minv, mo>(p) + device::mulx_reduc<1, mo - minv, mo>(q)));
  } if constexpr(x + 2 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 2], minv = U8CRT2M::im_inv[x + 2], inv2 = (mo + 1) / 2;
    uint32_t p = *(P += strideX), q = *(Q += strideX);
    r.z = uint8_t(device::barrett_reduc<mo>(uint32_t(r.z) + device::mulx_reduc<1, inv2, mo>(p) + device::mulx_reduc<1, inv2, mo>(q)));
    i.z = uint8_t(device::barrett_reduc<mo>(uint32_t(i.z) + device::mulx_reduc<1, minv, mo>(p) + device::mulx_reduc<1, mo - minv, mo>(q)));
  } if constexpr(x + 3 < orderX) {
    constexpr uint32_t mo = U8CRT2M::mo[x + 3], minv = U8CRT2M::im_inv[x + 3], inv2 = (mo + 1) / 2;
    uint32_t p = *(P += strideX), q = *(Q += strideX);
    r.w = uint8_t(device::barrett_reduc<mo>(uint32_t(r.w) + device::mulx_reduc<1, inv2, mo>(p) + device::mulx_reduc<1, inv2, mo>(q)));
    i.w = uint8_t(device::barrett_reduc<mo>(uint32_t(i.w) + device::mulx_reduc<1, minv, mo>(p) + device::mulx_reduc<1, mo - minv, mo>(q)));
  }
}

template<int32_t orderX, int32_t beta>
__global__ void crt_rotate_kernel(int64_t N, const uint32_t* __restrict__ X, int64_t ldx, int64_t strideX, uchar4* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 7) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    int64_t z = int64_t(blockIdx.z); X = &X[(z << 2) * strideX]; A = &A[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1) + (z * strideA)];
    constexpr int64_t panelsA = int64_t((orderX + 3) / 4);
    uchar4* AImag = &A[panelsA * strideA], r, i; if constexpr(beta) { r = *A; i = *AImag; } else { r = i = uchar4(); }
    switch (int32_t(blockIdx.z)) {
      case 0: accumulate_modular_rotate<orderX, 0>(r, i, &X[y + (x * ldx)], &X[x + (y * ldx)], strideX); break;
      case 1: accumulate_modular_rotate<orderX, 4>(r, i, &X[y + (x * ldx)], &X[x + (y * ldx)], strideX); break;
      case 2: accumulate_modular_rotate<orderX, 8>(r, i, &X[y + (x * ldx)], &X[x + (y * ldx)], strideX); break;
      case 3: accumulate_modular_rotate<orderX, 12>(r, i, &X[y + (x * ldx)], &X[x + (y * ldx)], strideX); break;
      case 4: accumulate_modular_rotate<orderX, 16>(r, i, &X[y + (x * ldx)], &X[x + (y * ldx)], strideX); break;
    }
    *A = r; *AImag = i;
  }
}

template <int32_t orderX> static inline void crt_rotate(cudaStream_t stream, int32_t beta, int32_t N, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A) {
  constexpr int32_t block_threads = 128;
  dim3 grid(uint32_t(N + 127) >> 7, uint32_t(N), uint32_t((std::min(U8CRT2M::len_2m_moduli, orderX) + 3) / 4));
  int64_t N64 = int64_t(N), ldx64 = int64_t(ldx), strideA = (N64 + (N64 * N64)) / int64_t(2);
  if (beta) { crt_rotate_kernel<orderX, 1> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); } else
  { crt_rotate_kernel<orderX, 0> <<< grid, block_threads, 0, stream >>> (N64, (uint32_t*)X, ldx64, strideX, (uchar4*)A, strideA); }
}

void internal::int8::accumulate_crt_rotate(cudaStream_t stream, int32_t beta, int32_t N, int32_t orderX, const int32_t* X, int32_t ldx, int64_t strideX, uint32_t* A) {
  switch (orderX) {
    case 1: crt_rotate<1>(stream, beta, N, X, ldx, strideX, A); return;
    case 2: crt_rotate<2>(stream, beta, N, X, ldx, strideX, A); return;
    case 3: crt_rotate<3>(stream, beta, N, X, ldx, strideX, A); return;
    case 4: crt_rotate<4>(stream, beta, N, X, ldx, strideX, A); return;
    case 5: crt_rotate<5>(stream, beta, N, X, ldx, strideX, A); return;
    case 6: crt_rotate<6>(stream, beta, N, X, ldx, strideX, A); return;
    case 7: crt_rotate<7>(stream, beta, N, X, ldx, strideX, A); return;
    case 8: crt_rotate<8>(stream, beta, N, X, ldx, strideX, A); return;
    case 9: crt_rotate<9>(stream, beta, N, X, ldx, strideX, A); return;
    case 10: crt_rotate<10>(stream, beta, N, X, ldx, strideX, A); return;
    case 11: crt_rotate<11>(stream, beta, N, X, ldx, strideX, A); return;
    case 12: crt_rotate<12>(stream, beta, N, X, ldx, strideX, A); return;
    case 13: crt_rotate<13>(stream, beta, N, X, ldx, strideX, A); return;
    case 14: crt_rotate<14>(stream, beta, N, X, ldx, strideX, A); return;
    case 15: crt_rotate<15>(stream, beta, N, X, ldx, strideX, A); return;
    case 16: crt_rotate<16>(stream, beta, N, X, ldx, strideX, A); return;
    case 17: crt_rotate<17>(stream, beta, N, X, ldx, strideX, A); return;
    case 18: crt_rotate<18>(stream, beta, N, X, ldx, strideX, A); return;
    case 19: crt_rotate<19>(stream, beta, N, X, ldx, strideX, A); return;
    case 20: crt_rotate<20>(stream, beta, N, X, ldx, strideX, A); return;
    case 21: crt_rotate<21>(stream, beta, N, X, ldx, strideX, A); return;
    case 22: crt_rotate<22>(stream, beta, N, X, ldx, strideX, A); return;
    case 23: crt_rotate<23>(stream, beta, N, X, ldx, strideX, A); return;
    case 24: crt_rotate<24>(stream, beta, N, X, ldx, strideX, A); return;
    case 25: crt_rotate<25>(stream, beta, N, X, ldx, strideX, A); return;
    default: return;
  }
}
