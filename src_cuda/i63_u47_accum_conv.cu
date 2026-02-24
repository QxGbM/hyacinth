
#include <internal.hpp>

__device__ __forceinline__ void conv_u47_x1(uint64_t i0, uint64_t& o0, uint64_t& o1) {
  constexpr uint64_t i47 = (uint64_t(1) << 47) - uint64_t(1);
  o0 = i0 & i47; o1 = i0 >> 47;
}

__device__ __forceinline__ void conv_u47_x2(uint64_t i0, uint64_t i1, uint64_t& o0, uint64_t& o1, uint64_t& o2) {
  constexpr uint64_t i47 = (uint64_t(1) << 47) - uint64_t(1);
  o0 = i0 & i47; o1 = ((i0 >> 47) | (i1 << 16)) & i47; o2 = i1 >> 31;
}

__device__ __forceinline__ void conv_u47_x3(uint64_t i0, uint64_t i1, uint64_t i2, uint64_t& o0, uint64_t& o1, uint64_t& o2, uint64_t& o3) {
  constexpr uint64_t i47 = (uint64_t(1) << 47) - uint64_t(1);
  o0 = i0 & i47; o1 = ((i0 >> 47) | (i1 << 16)) & i47; o2 = ((i1 >> 31) | (i2 << 32)) & i47; o3 = i2 >> 15;
}

template <int32_t mode>
__global__ void limbs_convert_kernel(int64_t N, uint64_t* __restrict__ A) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x); A = &A[i];
  if (i < N) {
    int64_t N2 = N << 1, N4 = N << 2;

    if constexpr(mode == 0)
      conv_u47_x1(A[0], A[0], A[N]);
    else if constexpr(mode == 1)
      conv_u47_x2(A[0], A[N], A[0], A[N], A[N2]);
    else if constexpr(mode == 2)
      conv_u47_x3(A[0], A[N], A[N2], A[0], A[N], A[N2], A[N + N2]);
    else if constexpr(mode == 3) {
      conv_u47_x1(A[N], A[N2], A[N + N2]);
      conv_u47_x1(A[0], A[0], A[N]);
    }
    else if constexpr(mode == 4) {
      conv_u47_x2(A[N2], A[N + N2], A[N + N2], A[N4], A[N + N4]);
      conv_u47_x2(A[0], A[N], A[0], A[N], A[N2]);
    }
    else if constexpr(mode == 5) {
      conv_u47_x3(A[N + N2], A[N4], A[N + N4], A[N4], A[N + N4], A[N2 + N4], A[N + N2 + N4]);
      conv_u47_x3(A[0], A[N], A[N2], A[0], A[N], A[N2], A[N + N2]);
    }
  }
}

void internal::int8::accumulate_conv_i63_u47(cudaStream_t stream, int32_t orderA, int32_t Complex, int64_t N, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);
  if (Complex) switch (orderA) {
    case 1: limbs_convert_kernel<3> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 2: limbs_convert_kernel<4> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 3: limbs_convert_kernel<5> <<< grid, block_threads, 0, stream >>> (N, A); break;
    default: break;
  }
  else switch (orderA) {
    case 1: limbs_convert_kernel<0> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 2: limbs_convert_kernel<1> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 3: limbs_convert_kernel<2> <<< grid, block_threads, 0, stream >>> (N, A); break;
    default: break;
  }
}
