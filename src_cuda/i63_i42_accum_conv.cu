
#include <internal.hpp>

__device__ __forceinline__ void conv1_2(uint64_t i_lo, uint64_t& o_lo, uint64_t& o_hi) {
  constexpr uint64_t i42 = 0x3ffffffffffllu;
  o_hi = i_lo >> 21; o_lo = i_lo & i42;
}

__device__ __forceinline__ void conv2_3(uint64_t i_lo, uint64_t i_hi, uint64_t& o_lo, uint64_t& o_mi, uint64_t& o_hi) {
  constexpr uint64_t i42 = 0x3ffffffffffllu;
  o_hi = i_hi >> 21; o_mi = ((i_hi << 21) & i42) | (i_lo >> 42); o_lo = i_lo & i42;
}

template <int32_t mode>
__global__ void limbs_convert_kernel(int64_t N, uint64_t* __restrict__ A) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x); A = &A[i];
  if (i < N) {
    int64_t N2 = N << 1, N4 = N << 2, N8 = N << 3;

    if constexpr(mode == 2)
      conv1_2(A[0], A[0], A[N]);
    else if constexpr(mode == 3)
      conv2_3(A[0], A[N], A[0], A[N], A[N2]);
    else if constexpr(mode == 4) {
      conv1_2(A[N], A[N2], A[N + N2]);
      conv1_2(A[0], A[0], A[N]);
    }
    else if constexpr(mode == 5) {
      conv1_2(A[N2], A[N + N2], A[N4]);
      conv2_3(A[0], A[N], A[0], A[N], A[N2]);
    }
    else if constexpr(mode == 6) {
      conv2_3(A[N2], A[N + N2], A[N + N2], A[N4], A[N + N4]);
      conv2_3(A[0], A[N], A[0], A[N], A[N2]);
    }
    else if constexpr(mode == 10) {
      conv1_2(A[N + N4], A[N8], A[N + N8]);
      conv2_3(A[N + N2], A[N4], A[N + N4], A[N2 + N4], A[N + N2 + N4]);
      conv1_2(A[N2], A[N + N2], A[N4]);
      conv2_3(A[0], A[N], A[0], A[N], A[N2]);
    }
    else if constexpr(mode == 12) {
      conv2_3(A[N2 + N4], A[N + N2 + N4], A[N + N8], A[N2 + N8], A[N + N2 + N8]);
      conv2_3(A[N4], A[N + N4], A[N2 + N4], A[N + N2 + N4], A[N8]);
      conv2_3(A[N2], A[N + N2], A[N + N2], A[N4], A[N + N4]);
      conv2_3(A[0], A[N], A[0], A[N], A[N2]);
    }
  }
}

inline void limbs_convert_dispatcher(cudaStream_t stream, int32_t orderA, int64_t N, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);

  switch (orderA) {
    case 2: limbs_convert_kernel<2> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 3: limbs_convert_kernel<3> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 4: limbs_convert_kernel<4> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 5: limbs_convert_kernel<5> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 6: limbs_convert_kernel<6> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 10: limbs_convert_kernel<10> <<< grid, block_threads, 0, stream >>> (N, A); break;
    case 12: limbs_convert_kernel<12> <<< grid, block_threads, 0, stream >>> (N, A); break;
    default: break;
  }
}

void internal::int8::accumulate_conv_i63_i42(cudaStream_t stream, int32_t orderA, int64_t N, uint64_t* A) {
  limbs_convert_dispatcher(stream, orderA, N, A);
}
