
#include <internal.hpp>
#include <ext_arith.hpp>
#include <stdexcept>

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  constexpr uint64_t i63 = uint64_t(0x7fffffffffffffffllu);
  if constexpr(0 < orderIn) { a[0] = *in; } else if constexpr(0 < ORDER) { a[0] = uint64_t(0); }
  if constexpr(1 < orderIn) { a[1] = *(in += stride); } else if constexpr(1 < ORDER) { a[1] = -(a[0] >> 63); a[0] &= i63; }
  if constexpr(2 < orderIn) { a[2] = *(in += stride); } else if constexpr(2 < ORDER) { a[2] = -(a[1] >> 63); a[1] &= i63; }
}

template <int32_t ORDER> __device__ __forceinline__ void add_shifted_dyn(uint64_t (&a)[ORDER], int64_t i, uint32_t expon) {
  constexpr uint64_t i63 = 0x7fffffffffffffffllu;
  if constexpr(ORDER == 1) {
    uint64_t m0 = -uint64_t(expon <= uint32_t(63));
    uint32_t rem = expon & uint32_t(m0);
    uint64_t q0 = uint64_t(i) << rem;
    a[0] += q0 & m0;
  } else if constexpr(ORDER == 2) {
    uint64_t m0 = -uint64_t(expon < uint32_t(63));
    uint64_t m1 = -uint64_t((expon - uint32_t(63)) <= uint32_t(63));
    uint32_t rem = (expon - (uint32_t(63) & uint32_t(m1))) & (uint32_t(m0) | uint32_t(m1));
    uint64_t q0 = uint64_t(i) << rem;
    a[0] += q0 & m0 & i63;
    uint64_t sign = -(uint64_t(i) >> 63), q1 = (sign << (uint32_t(1) + rem)) | (uint64_t(i) >> (uint32_t(63) - rem));
    a[1] += ((q1 & m0) | (q0 & m1)) + (a[0] >> 63); a[0] &= i63;
  } else if constexpr(ORDER == 3) {
    uint64_t m0 = -uint64_t(expon < uint32_t(63));
    uint64_t m1 = -uint64_t((expon - uint32_t(63)) < uint32_t(63));
    uint64_t m2 = -uint64_t((expon - uint32_t(126)) <= uint32_t(63));
    uint32_t rem = (expon - (uint32_t(63) & uint32_t(m1)) - (uint32_t(126) & uint32_t(m2))) & (uint32_t(m0) | uint32_t(m1) | uint32_t(m2));
    uint64_t q0 = uint64_t(i) << rem;
    a[0] += q0 & m0 & i63;
    uint64_t sign = -(uint64_t(i) >> 63), q1 = (sign << (uint32_t(1) + rem)) | (uint64_t(i) >> (uint32_t(63) - rem));
    a[1] += (((q1 & m0) | (q0 & m1)) & i63) + (a[0] >> 63); a[0] &= i63;
    a[2] += ((sign & m0) | (q1 & m1) | (q0 & m2)) + (a[1] >> 63); a[1] &= i63;
  }
}

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void add_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < orderIn) { device::add_shifted<0>(a, int64_t(*in)); }
  if constexpr(1 < orderIn) { device::add_shifted<63>(a, int64_t(*(in += stride))); }
  if constexpr(2 < orderIn) { device::add_shifted<126>(a, int64_t(*(in += stride))); }
}

template <int32_t orderIn, int32_t ORDER> __device__ __forceinline__ void add_i(uint64_t (&r)[ORDER], uint64_t (&i)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < orderIn) { device::add_shifted<0>(r, int64_t(*in)); }
  if constexpr(1 < orderIn) { device::add_shifted<63>(r, int64_t(*(in += stride))); }
  if constexpr(2 < orderIn) { device::add_shifted<126>(r, int64_t(*(in += stride))); }
  if constexpr(0 < orderIn) { device::add_shifted<0>(i, int64_t(*(in += stride))); }
  if constexpr(1 < orderIn) { device::add_shifted<63>(i, int64_t(*(in += stride))); }
  if constexpr(2 < orderIn) { device::add_shifted<126>(i, int64_t(*(in += stride))); }
}

template <int32_t ORDER>
__device__ __forceinline__ void cross_sum(uint64_t (&rl)[ORDER], uint64_t (&im)[ORDER]) {
  uint64_t t[ORDER]{};
  if constexpr(0 < ORDER) { device::add_shifted<0>(t, -int64_t(im[0])); }
  if constexpr(1 < ORDER) { device::add_shifted<63>(t, -int64_t(im[1])); }
  if constexpr(2 < ORDER) { device::add_shifted<126>(t, -int64_t(im[2])); }
  if constexpr(0 < ORDER) { int64_t r = -int64_t(rl[0]); device::add_shifted<0>(t, r); device::add_shifted<0>(im, r); rl[0] = t[0]; }
  if constexpr(1 < ORDER) { int64_t r = -int64_t(rl[1]); device::add_shifted<63>(t, r); device::add_shifted<63>(im, r); rl[1] = t[1]; }
  if constexpr(2 < ORDER) { int64_t r = -int64_t(rl[2]); device::add_shifted<126>(t, r); device::add_shifted<126>(im, r); rl[2] = t[2]; }
}

template <int32_t orderOut, int32_t ORDER> __device__ __forceinline__ void store_i(uint64_t (&a)[ORDER], uint64_t* out, int64_t stride) {
  constexpr uint64_t b63 = uint64_t(0x8000000000000000llu);
  if constexpr(0 < orderOut && orderOut < ORDER) { a[orderOut - 1] |= a[ORDER - 1] & b63; }
  if constexpr(0 < orderOut) { *out = a[0]; } if constexpr(1 < orderOut) { *(out += stride) = a[1]; } if constexpr(2 < orderOut) { *(out += stride) = a[2]; }
}

template <int32_t orderOut, int32_t ORDER> __device__ __forceinline__ void store_i(uint64_t (&r)[ORDER], uint64_t (&i)[ORDER], uint64_t* out, int64_t stride) {
  constexpr uint64_t b63 = uint64_t(0x8000000000000000llu);
  if constexpr(0 < orderOut && orderOut < ORDER) { r[orderOut - 1] |= r[ORDER - 1] & b63; i[orderOut - 1] |= i[ORDER - 1] & b63; }
  if constexpr(0 < orderOut) { *out = r[0]; } if constexpr(1 < orderOut) { *(out += stride) = r[1]; } if constexpr(2 < orderOut) { *(out += stride) = r[2]; }
  if constexpr(0 < orderOut) { *(out += stride) = i[0]; } if constexpr(1 < orderOut) { *(out += stride) = i[1]; } if constexpr(2 < orderOut) { *(out += stride) = i[2]; }
}

template<int32_t orderA, int32_t orderB, int32_t beta, class sum_t>
__global__ void triangle_pack_kernel(int64_t N, const uint64_t* __restrict__ A, int64_t strideA, int64_t K, const sum_t* __restrict__ vsum, uint32_t corr, uint64_t* __restrict__ B, int64_t strideB) {
  constexpr int32_t ORDER = orderA < orderB ? orderB : orderA, Complex = std::is_same_v<sum_t, ulonglong4_32a>;
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x), x = int64_t(blockIdx.y);
  if (y <= x) {
    A = &A[y + (x * N)]; B = &B[y + int64_t(uint64_t((x + int64_t(1)) * x) >> 1)];
    if constexpr(Complex) {
      int64_t strideIm = int64_t(orderA) * strideA, strideImT = strideIm + ((y - x) * (N - int64_t(1)));
      uint64_t acc_rl[ORDER], acc_im[ORDER];
      load_i<orderA>(acc_rl, &A[strideImT], strideA);
      load_i<orderA>(acc_im, &A[strideIm], strideA);

      if (K) {
        ulonglong4_32a sy = vsum[y];
        add_shifted_dyn(acc_rl, int64_t(sy.x), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc_rl, int64_t(sy.y), corr + uint32_t(63)); }
        add_shifted_dyn(acc_im, int64_t(sy.z), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc_im, int64_t(sy.w), corr + uint32_t(63)); }
        ulonglong4_32a sx = vsum[x];
        add_shifted_dyn(acc_im, int64_t(sx.x), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc_im, int64_t(sx.y), corr + uint32_t(63)); }
        add_shifted_dyn(acc_rl, int64_t(sx.z), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc_rl, int64_t(sx.w), corr + uint32_t(63)); }
        cross_sum(acc_rl, acc_im);
        add_shifted_dyn(acc_rl, K << 1, corr << 1);
      }
      else { cross_sum(acc_rl, acc_im); }
      add_i<orderA>(acc_rl, A, strideA);

      if constexpr(beta) { add_i<orderB>(acc_rl, acc_im, B, strideB); }
      store_i<orderB>(acc_rl, acc_im, B, strideB);
    }
    else {
      uint64_t acc[ORDER];
      load_i<orderA>(acc, A, strideA);

      if (K) {
        ulonglong2 sy = vsum[y];
        add_shifted_dyn(acc, int64_t(-sy.x), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc, int64_t(-sy.y), corr + uint32_t(63)); }
        ulonglong2 sx = vsum[x];
        add_shifted_dyn(acc, int64_t(-sx.x), corr); if constexpr(1 < ORDER) { add_shifted_dyn(acc, int64_t(-sx.y), corr + uint32_t(63)); }
        add_shifted_dyn(acc, K, corr << 1);
      }
      if constexpr(beta) { add_i<orderB>(acc, B, strideB); }
      store_i<orderB>(acc, B, strideB);
    }
  }
}

template <int32_t beta, class sum_t>
inline void triangle_pack_dispatcher(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const sum_t* vsum, uint32_t corr, int32_t orderB, uint64_t* B) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t K64 = int64_t(M), N64 = int64_t(N);
  int64_t strideA = N64 * N64, strideB = (strideA + N64) / int64_t(2);
  int32_t mode = (1 <= orderA && orderA <= 3 && 1 <= orderB && orderB <= 3) ? ((orderA - 1) + ((orderB - 1) * 3)) : -1;
  vsum = 0 < M ? vsum : nullptr;

  switch(mode) {
    case 0: triangle_pack_kernel<1, 1, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 1: triangle_pack_kernel<2, 1, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 2: triangle_pack_kernel<3, 1, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 3: triangle_pack_kernel<1, 2, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 4: triangle_pack_kernel<2, 2, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 5: triangle_pack_kernel<3, 2, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 6: triangle_pack_kernel<1, 3, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 7: triangle_pack_kernel<2, 3, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    case 8: triangle_pack_kernel<3, 3, beta> <<< grid, block_threads, 0, stream >>> (N64, A, strideA, K64, vsum, corr, B, strideB); return;
    default: return;
  }
}

void internal::int8::triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong2* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B) {
  if (beta) triangle_pack_dispatcher<1>(stream, M, N, orderA, A, vsum, corr, orderB, B);
    else triangle_pack_dispatcher<0>(stream, M, N, orderA, A, vsum, corr, orderB, B);
}

void internal::int8::triangle_pack(cudaStream_t stream, int32_t M, int32_t N, int32_t orderA, const uint64_t* A, const ulonglong4_32a* vsum, uint32_t corr, int32_t beta, int32_t orderB, uint64_t* B) {
  corr += uint32_t(corr ? -1 : 0);
  if (beta) triangle_pack_dispatcher<1>(stream, M, N, orderA, A, vsum, corr, orderB, B);
    else triangle_pack_dispatcher<0>(stream, M, N, orderA, A, vsum, corr, orderB, B);
}
