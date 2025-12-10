
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template<uint32_t depth, uint32_t orderA, int32_t BLOCK_THREADS>
__global__ void i32_accum_kernel(uint32_t sft, uint32_t sft_iter, int64_t N, const int32_t* __restrict__ X, uint64_t* __restrict__ A) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  if (i < N) {
    uint64_t acc[orderA];
    int64_t iter = i;
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { acc[r] = A[iter]; iter += N; }

    iter = i;
    #pragma unroll
    for (uint32_t r = 0; r < depth; ++r) {
      device::int8::add_shifted(acc, int64_t(X[iter]), sft);
      sft += sft_iter; iter += N;
    }

    iter = i;
    #pragma unroll
    for (uint32_t r = 0; r < orderA; ++r)
    { A[iter] = acc[r]; iter += N; }
  }
}

constexpr int32_t block_threads = 512;

template <uint32_t orderA>
inline void acc_dispatcher(cudaStream_t stream, int64_t N, uint32_t depth_lo, uint32_t depth_hi, uint32_t sft_iter, const int32_t* X, uint64_t* A) {
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  uint32_t depth = depth_hi - depth_lo, sft = sft_iter * depth_lo;

  switch (depth) {
    case 1: i32_accum_kernel<1, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 2: i32_accum_kernel<2, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 3: i32_accum_kernel<3, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 4: i32_accum_kernel<4, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 5: i32_accum_kernel<5, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 6: i32_accum_kernel<6, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 7: i32_accum_kernel<7, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 8: i32_accum_kernel<8, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 9: i32_accum_kernel<9, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 10: i32_accum_kernel<10, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    case 11: i32_accum_kernel<11, orderA, block_threads> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, A); break;
    default: break;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, int64_t N, int32_t depth_lo, int32_t depth_hi, int32_t sft, const int32_t* X, int32_t orderA, uint64_t* A) {
  switch (orderA) {
    case 1: acc_dispatcher<1>(stream, N, uint32_t(depth_lo), uint32_t(depth_hi), uint32_t(sft), X, A); break;
    case 2: acc_dispatcher<2>(stream, N, uint32_t(depth_lo), uint32_t(depth_hi), uint32_t(sft), X, A); break;
    case 3: acc_dispatcher<3>(stream, N, uint32_t(depth_lo), uint32_t(depth_hi), uint32_t(sft), X, A); break;
    case 4: acc_dispatcher<4>(stream, N, uint32_t(depth_lo), uint32_t(depth_hi), uint32_t(sft), X, A); break;
    default: break;
  }
}
