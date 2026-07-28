
#include <internal.hpp>

template <int32_t hi_bits> __device__ __forceinline__ uint64_t sign_bits(uint64_t i) {
  static_assert(0 <= hi_bits && hi_bits <= 64);
  constexpr uint64_t mask = hi_bits ? (uint64_t(0xffffffffffffffffllu) << (64 - hi_bits)) : uint64_t(0);
  return mask & (-(i >> 63));
}

template <int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&i)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER) { i[0] = *in; }
  if constexpr(1 < ORDER) { i[1] = *(in += stride); }
  if constexpr(2 < ORDER) { i[2] = *(in += stride); }
}

__device__ __forceinline__ void conv_u32_x1(uint64_t i0, uint64_t* out, int64_t stride) {
  constexpr uint64_t i32 = uint64_t(0xffffffffllu);
  *out = i0 & i32;
  *(out += stride) = (i0 >> 32) | sign_bits<32>(i0);
}

__device__ __forceinline__ void conv_u43_x2(uint64_t i0, uint64_t i1, uint64_t* out, int64_t stride) {
  constexpr uint64_t i43 = uint64_t(0x7ffffffffffllu);
  *out = i0 & i43;
  *(out += stride) = ((i0 >> 43) | (i1 << 20)) & i43;
  *(out += stride) = (i1 >> 23) | sign_bits<23>(i1);
}

__device__ __forceinline__ void conv_u32_x2(uint64_t i0, uint64_t i1, uint64_t* out, int64_t stride) {
  constexpr uint64_t i32 = uint64_t(0xffffffffllu);
  *out = i0 & i32;
  *(out += stride) = ((i0 >> 32) | (i1 << 31)) & i32;
  *(out += stride) = (i1 >> 1) & i32;
  *(out += stride) = (i1 >> 33) | sign_bits<33>(i1);
}

__device__ __forceinline__ void conv_u48_x3(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t* out, int64_t stride) {
  constexpr uint64_t i48 = uint64_t(0xffffffffffffllu);
  *out = i0 & i48;
  *(out += stride) = ((i0 >> 48) | (i1 << 15)) & i48;
  *(out += stride) = ((i1 >> 33) | (i2 << 30)) & i48;
  *(out += stride) = (i1 >> 18) | sign_bits<18>(i1);
}

__device__ __forceinline__ void conv_u38_x3(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t* out, int64_t stride) {
  constexpr uint64_t i38 = uint64_t(0x3fffffffffllu);
  *out = i0 & i38;
  *(out += stride) = ((i0 >> 38) | (i1 << 25)) & i38;
  *(out += stride) = (i1 >> 13) & i38;
  *(out += stride) = ((i1 >> 51) | (i2 << 12)) & i38;
  *(out += stride) = (i2 >> 26) | sign_bits<26>(i2);
}

__device__ __forceinline__ void conv_u32_x3(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t* out, int64_t stride) {
  constexpr uint64_t i32 = uint64_t(0xffffffffllu);
  *out = i0 & i32;
  *(out += stride) = ((i0 >> 32) | (i1 << 31)) & i32;
  *(out += stride) = (i1 >> 1) & i32;
  *(out += stride) = ((i1 >> 33) | (i2 << 30)) & i32;
  *(out += stride) = (i2 >> 2) & i32;
  *(out += stride) = (i2 >> 34) | sign_bits<34>(i2);
}

template <int32_t mode> __global__ void limbs_convert_kernel(int64_t N, uint64_t* __restrict__ A) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x); A = &A[i];
  if (i < N) {
    if constexpr(mode == 0) { uint64_t a[1]; load_i(a, A, N); conv_u32_x1(a[0], A, N); }
    else if constexpr(mode == 1) { uint64_t a[2]; load_i(a, A, N); conv_u43_x2(a[0], a[1], A, N); }
    else if constexpr(mode == 2) { uint64_t a[2]; load_i(a, A, N); conv_u32_x2(a[0], a[1], A, N); }
    else if constexpr(mode == 3) { uint64_t a[3]; load_i(a, A, N); conv_u48_x3(a[0], a[1], a[2], A, N); }
    else if constexpr(mode == 4) { uint64_t a[3]; load_i(a, A, N); conv_u38_x3(a[0], a[1], a[2], A, N); }
    else if constexpr(mode == 5) { uint64_t a[3]; load_i(a, A, N); conv_u32_x3(a[0], a[1], a[2], A, N); }
    else if constexpr(mode == 6) { uint64_t a[1]; load_i(a, &A[N], N); conv_u32_x1(a[0], &A[N * int64_t(2)], N); load_i(a, A, N); conv_u32_x1(a[0], A, N); }
    else if constexpr(mode == 7) { uint64_t a[2]; load_i(a, &A[N * int64_t(2)], N); conv_u43_x2(a[0], a[1], &A[N * int64_t(3)], N); load_i(a, A, N); conv_u43_x2(a[0], a[1], A, N); }
    else if constexpr(mode == 8) { uint64_t a[2]; load_i(a, &A[N * int64_t(2)], N); conv_u32_x2(a[0], a[1], &A[N * int64_t(4)], N); load_i(a, A, N); conv_u32_x2(a[0], a[1], A, N); }
    else if constexpr(mode == 9) { uint64_t a[3]; load_i(a, &A[N * int64_t(3)], N); conv_u48_x3(a[0], a[1], a[2], &A[N * int64_t(4)], N); load_i(a, A, N); conv_u48_x3(a[0], a[1], a[2], A, N); }
    else if constexpr(mode == 10) { uint64_t a[3]; load_i(a, &A[N * int64_t(3)], N); conv_u38_x3(a[0], a[1], a[2], &A[N * int64_t(5)], N); load_i(a, A, N); conv_u38_x3(a[0], a[1], a[2], A, N); }
    else if constexpr(mode == 11) { uint64_t a[3]; load_i(a, &A[N * int64_t(3)], N); conv_u32_x3(a[0], a[1], a[2], &A[N * int64_t(6)], N); load_i(a, A, N); conv_u32_x3(a[0], a[1], a[2], A, N); }
  }
}

void internal::int8::accumulate_conv_i63(cudaStream_t stream, int32_t orderA, int32_t LimbCount, int32_t Complex, int64_t N, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);
  if (!Complex && orderA == 1 && LimbCount == 2) {
    limbs_convert_kernel<0> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (!Complex && orderA == 2 && LimbCount == 3) {
    limbs_convert_kernel<1> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (!Complex && orderA == 2 && LimbCount == 4) {
    limbs_convert_kernel<2> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (!Complex && orderA == 3 && LimbCount == 4) {
    limbs_convert_kernel<3> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (!Complex && orderA == 3 && LimbCount == 5) {
    limbs_convert_kernel<4> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (!Complex && orderA == 3 && LimbCount == 6) {
    limbs_convert_kernel<5> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 1 && LimbCount == 2) {
    limbs_convert_kernel<6> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 2 && LimbCount == 3) {
    limbs_convert_kernel<7> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 2 && LimbCount == 4) {
    limbs_convert_kernel<8> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 3 && LimbCount == 4) {
    limbs_convert_kernel<9> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 3 && LimbCount == 5) {
    limbs_convert_kernel<10> <<< grid, block_threads, 0, stream >>> (N, A);
  } else if (Complex && orderA == 3 && LimbCount == 6) {
    limbs_convert_kernel<11> <<< grid, block_threads, 0, stream >>> (N, A);
  }
}
