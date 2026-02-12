
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template<int32_t orderX, int32_t orderA, uint32_t sft_iter, int32_t beta>
__global__ void i32_accum_kernel(uint32_t sft, int64_t N, const int32_t* __restrict__ X, int64_t incx, uint64_t* __restrict__ A, int64_t inca) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    uint64_t acc[orderA];
    int64_t iter = i;

    if constexpr(beta) {
      #pragma unroll
      for (int32_t r = 0; r < orderA; ++r)
      { acc[r] = A[iter]; iter += inca; } iter = i;
    }
    else {
      #pragma unroll
      for (int32_t r = 0; r < orderA; ++r)
      { acc[r] = uint64_t(0); }
    }
    
    #pragma unroll
    for (int32_t r = 0; r < orderX; ++r) {
      device::int8::add_shifted(acc, int64_t(X[iter]), sft);
      sft += sft_iter; iter += incx;
    }

    iter = i;
    #pragma unroll
    for (int32_t r = 0; r < orderA; ++r)
    { A[iter] = acc[r]; iter += inca; }
  }
}

template <int32_t orderA, int32_t option>
inline void acc_dispatcher(cudaStream_t stream, int64_t N, uint32_t sft_lo, uint32_t orderX, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  constexpr uint32_t sft_iter = uint32_t(option & 4 ? 16 : 8), alpha = uint32_t(option & 2) >> 1;
  constexpr int32_t beta = option & 1, block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);
  uint32_t sft = alpha + (sft_lo << 3);

  switch (orderX) {
    case 1: i32_accum_kernel<1, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 2: i32_accum_kernel<2, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 3: i32_accum_kernel<3, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 4: i32_accum_kernel<4, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 5: i32_accum_kernel<5, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 6: i32_accum_kernel<6, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 7: i32_accum_kernel<7, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 8: i32_accum_kernel<8, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 9: i32_accum_kernel<9, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 10: i32_accum_kernel<10, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 11: i32_accum_kernel<11, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    case 12: i32_accum_kernel<12, orderA, sft_iter, beta> <<< grid, block_threads, 0, stream >>> (sft, N, X, incx, A, inca); break;
    default: break;
  }
}

template <uint32_t orderA>
inline void acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, uint32_t sft_lo, uint32_t orderX, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  switch (option) {
    case 0: acc_dispatcher<orderA, 0>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    case 1: acc_dispatcher<orderA, 1>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    case 2: acc_dispatcher<orderA, 2>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    case 3: acc_dispatcher<orderA, 3>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    case 4: acc_dispatcher<orderA, 4>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    case 5: acc_dispatcher<orderA, 5>(stream, N, sft_lo, orderX, X, incx, A, inca); break;
    default: break;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, int32_t option, int64_t N, int32_t sft_lo, int32_t orderX, const int32_t* X, int64_t incx, int32_t orderA, uint64_t* A, int64_t inca) {
  switch (orderA) {
    case 1: acc_dispatcher<1>(stream, option, N, uint32_t(sft_lo), uint32_t(orderX), X, incx, A, inca); break;
    case 2: acc_dispatcher<2>(stream, option, N, uint32_t(sft_lo), uint32_t(orderX), X, incx, A, inca); break;
    case 3: acc_dispatcher<3>(stream, option, N, uint32_t(sft_lo), uint32_t(orderX), X, incx, A, inca); break;
    case 4: acc_dispatcher<4>(stream, option, N, uint32_t(sft_lo), uint32_t(orderX), X, incx, A, inca); break;
    default: break;
  }
}
