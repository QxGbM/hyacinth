
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <stdexcept>

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void add_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride, uint32_t sft) {
  if constexpr(0 < orderIn) { device::int8::add_shifted(a, int64_t(*in), sft); }
  if constexpr(1 < orderIn) { device::int8::add_shifted(a, int64_t(*(in += stride)), sft + uint32_t(63)); }
  if constexpr(2 < orderIn) { device::int8::add_shifted(a, int64_t(*(in += stride)), sft + uint32_t(126)); }
}

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void add_i(uint64_t (&r)[ORDER], uint64_t (&i)[ORDER], const uint64_t* in, int64_t stride, uint32_t sft) {
  if constexpr(0 < orderIn) { device::int8::add_shifted(r, int64_t(*in), sft); }
  if constexpr(1 < orderIn) { device::int8::add_shifted(r, int64_t(*(in += stride)), sft + uint32_t(63)); }
  if constexpr(2 < orderIn) { device::int8::add_shifted(r, int64_t(*(in += stride)), sft + uint32_t(126)); }
  if constexpr(0 < orderIn) { device::int8::add_shifted(i, int64_t(*(in += stride)), sft); }
  if constexpr(1 < orderIn) { device::int8::add_shifted(i, int64_t(*(in += stride)), sft + uint32_t(63)); }
  if constexpr(2 < orderIn) { device::int8::add_shifted(i, int64_t(*(in += stride)), sft + uint32_t(126)); }
}

template <int32_t ORDER>
__device__ __forceinline__ void cross_sum(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER]) {
  uint64_t t[ORDER]{};
  if constexpr(0 < ORDER) { device::int8::add_shifted(t, -int64_t(im[0]), uint32_t(0)); }
  if constexpr(1 < ORDER) { device::int8::add_shifted(t, -int64_t(im[1]), uint32_t(63)); }
  if constexpr(2 < ORDER) { device::int8::add_shifted(t, -int64_t(im[2]), uint32_t(126)); }
  if constexpr(0 < ORDER) { int64_t r = -int64_t(rl[0]); device::int8::add_shifted(t, r, uint32_t(0)); device::int8::add_shifted(im, r, uint32_t(0)); }
  if constexpr(1 < ORDER) { int64_t r = -int64_t(rl[1]); device::int8::add_shifted(t, r, uint32_t(63)); device::int8::add_shifted(im, r, uint32_t(63)); }
  if constexpr(2 < ORDER) { int64_t r = -int64_t(rl[2]); device::int8::add_shifted(t, r, uint32_t(126)); device::int8::add_shifted(im, r, uint32_t(126)); }
  if constexpr(0 < ORDER) { rl[0] = t[0]; }
  if constexpr(1 < ORDER) { rl[1] = t[1]; }
  if constexpr(2 < ORDER) { rl[2] = t[2]; }
}

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < orderIn) { a[0] = *in; } else if constexpr(0 < ORDER) { a[0] = uint64_t(0); }
  if constexpr(1 < orderIn) { a[1] = *(in += stride); } else if constexpr(1 < ORDER) { a[1] = -(a[0] >> 63); a[0] &= uint64_t(0x7fffffffffffffffllu); }
  if constexpr(2 < orderIn) { a[2] = *(in += stride); } else if constexpr(2 < ORDER) { a[2] = -(a[1] >> 63); a[1] &= uint64_t(0x7fffffffffffffffllu); }
}

template <int32_t orderOut, int32_t ORDER> __device__ __forceinline__ void store_i(uint64_t (&a)[ORDER], uint64_t* out, int64_t stride) {
  if constexpr(0 < orderOut && orderOut < ORDER) { a[orderOut - 1] |= a[ORDER - 1] & uint64_t(0x8000000000000000llu); }
  if constexpr(0 < orderOut) { *out = a[0]; } if constexpr(1 < orderOut) { *(out += stride) = a[1]; } if constexpr(2 < orderOut) { *(out += stride) = a[2]; }
}

template <int32_t orderOut, int32_t ORDER> __device__ __forceinline__ void store_i(uint64_t (&r)[ORDER], uint64_t (&i)[ORDER], uint64_t* out, int64_t stride) {
  if constexpr(0 < orderOut && orderOut < ORDER) { r[orderOut - 1] |= r[ORDER - 1] & uint64_t(0x8000000000000000llu); i[orderOut - 1] |= i[ORDER - 1] & uint64_t(0x8000000000000000llu); }
  if constexpr(0 < orderOut) { *out = r[0]; } if constexpr(1 < orderOut) { *(out += stride) = r[1]; } if constexpr(2 < orderOut) { *(out += stride) = r[2]; }
  if constexpr(0 < orderOut) { *(out += stride) = i[0]; } if constexpr(1 < orderOut) { *(out += stride) = i[1]; } if constexpr(2 < orderOut) { *(out += stride) = i[2]; }
}

template<int32_t orderA, int32_t orderB, int32_t Complex, int32_t beta>
__global__ void triangle_pack_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int64_t K, const uint64_t* __restrict__ vsum, uint32_t corr, uint64_t* __restrict__ B, int64_t strideB) {
  constexpr int32_t ORDER = orderA < orderB ? orderB : orderA;
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    A = &A[y + (x * N)]; B = &B[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)];
    if constexpr(Complex) {
      int64_t strideIm = int64_t(orderA) * strideA, strideImT = strideIm + ((y - x) * (N - int64_t(1)));
      uint64_t acc_rl[ORDER], acc_im[ORDER];
      load_i<orderA>(acc_rl, &A[strideImT], strideA);
      load_i<orderA>(acc_im, &A[strideIm], strideA);

      if (K) {
        add_i<2>(acc_rl, acc_im, &vsum[y], N, corr);
        add_i<2>(acc_im, acc_rl, &vsum[x], N, corr);
        cross_sum(acc_rl, acc_im);
        device::int8::add_shifted(acc_rl, K, corr + corr);
      }
      else { cross_sum(acc_rl, acc_im); }
      add_i<orderA>(acc_rl, A, strideA, 0);

      if constexpr(beta) { add_i<orderB>(acc_rl, acc_im, B, strideB, uint32_t(0)); }
      store_i<orderB>(acc_rl, acc_im, B, strideB);
    }
    else {
      uint64_t acc[ORDER]{};
      load_i<orderA>(acc, A, strideA);

      if (K) {
        add_i<2>(acc, &vsum[y], N, corr);
        add_i<2>(acc, &vsum[x], N, corr);
        device::int8::add_shifted(acc, K, corr + corr);
      }
      if constexpr(beta) { add_i<orderB>(acc, B, strideB, uint32_t(0)); }
      store_i<orderB>(acc, B, strideB);
    }
  }
}

template <int32_t Complex, int32_t beta>
inline void triangle_pack_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const uint64_t* vsum, uint32_t corr, int32_t orderB, uint64_t* B) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t K64 = int64_t(M) << Complex, N64 = int64_t(N);
  int64_t strideA = N64 * N64, strideB = (strideA + N64) / int64_t(2);
  int32_t mode = (1 <= orderA && orderA <= 3 && 1 <= orderB && orderB <= 3) ? ((orderA - 1) + ((orderB - 1) * 3)) : -1;
  vsum = 0 < M ? vsum : nullptr;

  switch(mode) {
    case 0: triangle_pack_kernel<1, 1, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 1: triangle_pack_kernel<2, 1, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 2: triangle_pack_kernel<3, 1, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 3: triangle_pack_kernel<1, 2, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 4: triangle_pack_kernel<2, 2, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 5: triangle_pack_kernel<3, 2, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 6: triangle_pack_kernel<1, 3, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 7: triangle_pack_kernel<2, 3, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 8: triangle_pack_kernel<3, 3, Complex, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    default: return;
  }
}

template <> void internal::int8::triangle_pack<0>(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const uint64_t* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B) {
  if (beta) triangle_pack_dispatcher<0, 1>(stream, M, N, orderA, A, vsum, corr, orderB, B);
    else triangle_pack_dispatcher<0, 0>(stream, M, N, orderA, A, vsum, corr, orderB, B);
}

template <> void internal::int8::triangle_pack<1>(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const uint64_t* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B) {
  if (beta) triangle_pack_dispatcher<1, 1>(stream, M, N, orderA, A, vsum, corr, orderB, B);
    else triangle_pack_dispatcher<1, 0>(stream, M, N, orderA, A, vsum, corr, orderB, B);
}
