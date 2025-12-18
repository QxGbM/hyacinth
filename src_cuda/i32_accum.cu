
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template<uint32_t orderX, uint32_t orderA, uint32_t sft_iter, uint32_t beta, int32_t BLOCK_THREADS>
__global__ void i32_accum_kernel(uint32_t sft, int64_t N, const int32_t* __restrict__ X, uint64_t* __restrict__ A) {
  int64_t i = int64_t(blockIdx.x) * int64_t(BLOCK_THREADS) + int64_t(threadIdx.x);
  if (i < N) {
    uint64_t acc[orderA];
    int64_t iter = i;

    if constexpr(beta) {
      #pragma unroll
      for (uint32_t r = 0; r < orderA; ++r)
      { acc[r] = A[iter]; iter += N; }
      iter = i;
    }
    else {
      #pragma unroll
      for (uint32_t r = 0; r < orderA; ++r)
      { acc[r] = uint64_t(0); }
    }
    
    #pragma unroll
    for (uint32_t r = 0; r < orderX; ++r) {
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

template <uint32_t orderA, uint32_t sft_iter, uint32_t beta>
inline void acc_dispatcher(cudaStream_t stream, int64_t N, uint32_t sft_lo, uint32_t orderX, int32_t alpha, const int32_t* X, uint64_t* A) {
  int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));
  uint32_t sft = (sft_lo << 3) + uint32_t(alpha == 2);

  switch (orderX) {
    case 1: i32_accum_kernel<1, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 2: i32_accum_kernel<2, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 3: i32_accum_kernel<3, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 4: i32_accum_kernel<4, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 5: i32_accum_kernel<5, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 6: i32_accum_kernel<6, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 7: i32_accum_kernel<7, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 8: i32_accum_kernel<8, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 9: i32_accum_kernel<9, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 10: i32_accum_kernel<10, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    case 11: i32_accum_kernel<11, orderA, sft_iter, beta, block_threads> <<< grid, block_threads, 0, stream >>> (sft, N, X, A); break;
    default: break;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, int64_t N, int32_t sft_lo, int32_t orderX, int32_t alpha, const int32_t* X, int32_t orderA, int32_t beta, uint64_t* A) {
  if (beta == 0)
    switch (orderA) {
      case 1: acc_dispatcher<1, 8, 0>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 2: acc_dispatcher<2, 8, 0>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 3: acc_dispatcher<3, 8, 0>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 4: acc_dispatcher<4, 8, 0>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      default: break;
    }
  else
    switch (orderA) {
      case 1: acc_dispatcher<1, 8, 1>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 2: acc_dispatcher<2, 8, 1>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 3: acc_dispatcher<3, 8, 1>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      case 4: acc_dispatcher<4, 8, 1>(stream, N, uint32_t(sft_lo), uint32_t(orderX), alpha, X, A); break;
      default: break;
    }
}

void internal::int8::accumulate_i32tensor_sft2x(cudaStream_t stream, int64_t N, int32_t orderX, const int32_t* X, int32_t orderA, int32_t beta, uint64_t* A) {
  if (beta == 0)
    switch (orderA) {
      case 1: acc_dispatcher<1, 16, 0>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 2: acc_dispatcher<2, 16, 0>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 3: acc_dispatcher<3, 16, 0>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 4: acc_dispatcher<4, 16, 0>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      default: break;
    }
  else
    switch (orderA) {
      case 1: acc_dispatcher<1, 16, 1>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 2: acc_dispatcher<2, 16, 1>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 3: acc_dispatcher<3, 16, 1>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      case 4: acc_dispatcher<4, 16, 1>(stream, N, uint32_t(0), uint32_t(orderX), 1, X, A); break;
      default: break;
    }
}
