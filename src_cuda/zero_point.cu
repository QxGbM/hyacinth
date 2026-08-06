
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template <int32_t beta, int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER) { if constexpr(beta) { a[0] = *in; } else { a[0] = uint64_t(0); }}
  if constexpr(1 < ORDER) { if constexpr(beta) { a[1] = *(in += stride); } else { a[1] = uint64_t(0); }}
  if constexpr(2 < ORDER) { if constexpr(beta) { a[2] = *(in += stride); } else { a[2] = uint64_t(0); }}
}

template <int32_t alpha, int32_t ORDER> __device__ __forceinline__ void add_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride, int32_t sft = 0) {
  if constexpr(0 < ORDER) { if constexpr(alpha == 1) { device::int8::add_shifted(a, int64_t(*in), uint32_t(sft)); }
    else if constexpr(alpha == -1) { device::int8::add_shifted(a, -int64_t(*in), uint32_t(sft)); }}
  if constexpr(1 < ORDER) { if constexpr(alpha == 1) { device::int8::add_shifted(a, int64_t(*(in += stride)), uint32_t(sft + 63)); }
    else if constexpr(alpha == -1) { device::int8::add_shifted(a, -int64_t(*(in += stride)), uint32_t(sft + 63)); }}
  if constexpr(2 < ORDER) { if constexpr(alpha == 1) { device::int8::add_shifted(a, int64_t(*(in += stride)), uint32_t(sft + 126)); }
    else if constexpr(alpha == -1) { device::int8::add_shifted(a, -int64_t(*(in += stride)), uint32_t(sft + 126)); }}
}

template <int32_t ORDER> __device__ __forceinline__ void store_i(const uint64_t (&a)[ORDER], uint64_t* out, int64_t stride) {
  if constexpr(0 < ORDER) { *out = a[0]; }
  if constexpr(1 < ORDER) { *(out += stride) = a[1]; }
  if constexpr(2 < ORDER) { *(out += stride) = a[2]; }
}

template<int32_t orderA, int32_t beta>
__global__ void triangle_pack_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int64_t K, const uint64_t* __restrict__ sum, int32_t umax, uint64_t* __restrict__ P, int64_t strideP) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    P = &P[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)];
    uint64_t acc[orderA]; load_i<beta>(acc, P, strideP);
    add_i<1>(acc, &A[y + x * N], strideA);

    if (K) {
      device::int8::add_shifted(acc, K, uint32_t(umax) << 1);
      add_i<-1>(acc, &sum[y], N, umax);
      add_i<-1>(acc, &sum[x], N, umax);
    }
    store_i(acc, P, strideP);
  }
}

namespace internal::int8 {


}
