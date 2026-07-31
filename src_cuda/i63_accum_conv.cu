
#include <internal.hpp>

template <int32_t hi_bits> __device__ __forceinline__ uint64_t sign_bits(uint64_t a) {
  static_assert(0 <= hi_bits && hi_bits <= 64);
  constexpr uint64_t mask = hi_bits ? (uint64_t(0xffffffffffffffffllu) << (64 - hi_bits)) : uint64_t(0);
  return mask & (-(a >> 63));
}

template <int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER) { a[0] = *in; }
  if constexpr(1 < ORDER) { a[1] = *(in += stride); }
  if constexpr(2 < ORDER) { a[2] = *(in += stride); }
}

template <int32_t ORDER> __device__ __forceinline__ void conv_u32(const uint64_t (&a)[ORDER], uint64_t* out, int64_t stride) {
  constexpr uint64_t i32 = uint64_t(0xffffffffllu);
  *out = a[0] & i32;
  if constexpr(1 < ORDER) {
    *(out += stride) = ((a[0] >> 32) | (a[1] << 31)) & i32;
    *(out += stride) = (a[1] >> 1) & i32;
    if constexpr(2 < ORDER) {
      *(out += stride) = ((a[1] >> 33) | (a[2] << 30)) & i32;
      *(out += stride) = (a[2] >> 2) & i32;
      *(out += stride) = (a[2] >> 34) | sign_bits<34>(a[2]);
    } else { *(out += stride) = (a[1] >> 33) | sign_bits<33>(a[1]); }
  } else { *(out += stride) = (a[0] >> 32) | sign_bits<32>(a[0]); }
}

__device__ __forceinline__ void conv_u43(const uint64_t (&a)[2], uint64_t* out, int64_t stride) {
  constexpr uint64_t i43 = uint64_t(0x7ffffffffffllu);
  *out = a[0] & i43;
  *(out += stride) = ((a[0] >> 43) | (a[1] << 20)) & i43;
  *(out += stride) = (a[1] >> 23) | sign_bits<23>(a[1]);
}

__device__ __forceinline__ void conv_u48(const uint64_t (&a)[3], uint64_t* out, int64_t stride) {
  constexpr uint64_t i48 = uint64_t(0xffffffffffffllu);
  *out = a[0] & i48;
  *(out += stride) = ((a[0] >> 48) | (a[1] << 15)) & i48;
  *(out += stride) = ((a[1] >> 33) | (a[2] << 30)) & i48;
  *(out += stride) = (a[2] >> 18) | sign_bits<18>(a[2]);
}

__device__ __forceinline__ void conv_u38(const uint64_t (&a)[3], uint64_t* out, int64_t stride) {
  constexpr uint64_t i38 = uint64_t(0x3fffffffffllu);
  *out = a[0] & i38;
  *(out += stride) = ((a[0] >> 38) | (a[1] << 25)) & i38;
  *(out += stride) = (a[1] >> 13) & i38;
  *(out += stride) = ((a[1] >> 51) | (a[2] << 12)) & i38;
  *(out += stride) = (a[2] >> 26) | sign_bits<26>(a[2]);
}

template <int32_t mode, int32_t ORDER> __global__ void limbs_convert_kernel(int64_t N, uint64_t* __restrict__ A) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    uint64_t a[ORDER]; A = &A[i];
    if constexpr(mode == 0) { load_i(a, A, N); conv_u32(a, A, N); } else
    if constexpr(mode == 1) { load_i(a, A, N); conv_u43(a, A, N); } else
    if constexpr(mode == 2) { load_i(a, A, N); conv_u48(a, A, N); } else
    if constexpr(mode == 3) { load_i(a, A, N); conv_u38(a, A, N); } else
    if constexpr(mode == 4) { load_i(a, &A[N * int64_t(ORDER)], N); conv_u32(a, &A[N * int64_t(2 * ORDER)], N); load_i(a, A, N); conv_u32(a, A, N); } else
    if constexpr(mode == 5) { load_i(a, &A[N * int64_t(2)], N); conv_u43(a, &A[N * int64_t(3)], N); load_i(a, A, N); conv_u43(a, A, N); } else
    if constexpr(mode == 6) { load_i(a, &A[N * int64_t(3)], N); conv_u48(a, &A[N * int64_t(4)], N); load_i(a, A, N); conv_u48(a, A, N); } else
    if constexpr(mode == 7) { load_i(a, &A[N * int64_t(3)], N); conv_u38(a, &A[N * int64_t(5)], N); load_i(a, A, N); conv_u38(a, A, N); }
  }
}

void internal::int8::accumulate_conv_i63(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t Complex, int64_t N, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);
  if (!Complex && orderA == 1 && LimbCount == 2) { limbs_convert_kernel<0, 1> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (!Complex && orderA == 2 && LimbCount == 3) { limbs_convert_kernel<1, 2> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (!Complex && orderA == 2 && LimbCount == 4) { limbs_convert_kernel<0, 2> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (!Complex && orderA == 3 && LimbCount == 4) { limbs_convert_kernel<2, 3> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (!Complex && orderA == 3 && LimbCount == 5) { limbs_convert_kernel<3, 3> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (!Complex && orderA == 3 && LimbCount == 6) { limbs_convert_kernel<0, 3> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 1 && LimbCount == 2) { limbs_convert_kernel<4, 1> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 2 && LimbCount == 3) { limbs_convert_kernel<5, 2> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 2 && LimbCount == 4) { limbs_convert_kernel<4, 2> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 3 && LimbCount == 4) { limbs_convert_kernel<6, 3> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 3 && LimbCount == 5) { limbs_convert_kernel<7, 3> <<< grid, block_threads, 0, stream >>> (N, A); } else
  if (Complex && orderA == 3 && LimbCount == 6) { limbs_convert_kernel<4, 3> <<< grid, block_threads, 0, stream >>> (N, A); }
}
