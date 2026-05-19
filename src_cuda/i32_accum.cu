
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template<int32_t orderX, int32_t orderA, int32_t option>
__global__ void i32_accum_kernel(uint32_t sft, uint32_t sft_iter, int64_t N, const int32_t* __restrict__ X, int64_t incx, uint64_t* __restrict__ A, int64_t inca) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    constexpr int32_t alpha = option & 2, beta = option & 1;
    uint64_t acc[orderA]; A = &A[i]; X = &X[i];

    if constexpr(alpha) ++sft;
    if constexpr(beta) {
      if constexpr(uint32_t(0) < orderA) { acc[0] = A[0]; }
      if constexpr(uint32_t(1) < orderA) { acc[1] = A[inca]; }
      if constexpr(uint32_t(2) < orderA) { acc[2] = A[inca * int64_t(2)]; }
    }
    else {
      if constexpr(uint32_t(0) < orderA) { acc[0] = uint64_t(0); }
      if constexpr(uint32_t(1) < orderA) { acc[1] = uint64_t(0); }
      if constexpr(uint32_t(2) < orderA) { acc[2] = uint64_t(0); }
    }

    if constexpr(uint32_t(0) < orderX) { device::int8::add_shifted(acc, int64_t(X[0]), sft); }
    if constexpr(uint32_t(1) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx]), sft + sft_iter); }
    if constexpr(uint32_t(2) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(2)]), sft + sft_iter * uint32_t(2)); }
    if constexpr(uint32_t(3) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(3)]), sft + sft_iter * uint32_t(3)); }
    if constexpr(uint32_t(4) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(4)]), sft + sft_iter * uint32_t(4)); }
    if constexpr(uint32_t(5) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(5)]), sft + sft_iter * uint32_t(5)); }
    if constexpr(uint32_t(6) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(6)]), sft + sft_iter * uint32_t(6)); }
    if constexpr(uint32_t(7) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(7)]), sft + sft_iter * uint32_t(7)); }
    if constexpr(uint32_t(8) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(8)]), sft + sft_iter * uint32_t(8)); }
    if constexpr(uint32_t(9) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(9)]), sft + sft_iter * uint32_t(9)); }
    if constexpr(uint32_t(10) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(10)]), sft + sft_iter * uint32_t(10)); }
    if constexpr(uint32_t(11) < orderX) { device::int8::add_shifted(acc, int64_t(X[incx * int64_t(11)]), sft + sft_iter * uint32_t(11)); }

    if constexpr(uint32_t(0) < orderA) { A[0] = acc[0]; }
    if constexpr(uint32_t(1) < orderA) { A[inca] = acc[1]; }
    if constexpr(uint32_t(2) < orderA) { A[inca * int64_t(2)] = acc[2]; }
  }
}

template <int32_t orderA, int32_t option>
inline void acc_dispatcher(cudaStream_t stream, int64_t N, uint32_t sft, uint32_t sft_iter, uint32_t orderX, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);

  switch (orderX) {
    case 1: i32_accum_kernel<1, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 2: i32_accum_kernel<2, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 3: i32_accum_kernel<3, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 4: i32_accum_kernel<4, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 5: i32_accum_kernel<5, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 6: i32_accum_kernel<6, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 7: i32_accum_kernel<7, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 8: i32_accum_kernel<8, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 9: i32_accum_kernel<9, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 10: i32_accum_kernel<10, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 11: i32_accum_kernel<11, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    case 12: i32_accum_kernel<12, orderA, option> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, incx, A, inca); return;
    default: return;
  }
}

template <uint32_t orderA>
inline void acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, uint32_t sft, uint32_t sft_iter, uint32_t orderX, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  switch (option) {
    case 0: acc_dispatcher<orderA, 0>(stream, N, sft, sft_iter, orderX, X, incx, A, inca); return;
    case 1: acc_dispatcher<orderA, 1>(stream, N, sft, sft_iter, orderX, X, incx, A, inca); return;
    case 2: acc_dispatcher<orderA, 2>(stream, N, sft, sft_iter, orderX, X, incx, A, inca); return;
    case 3: acc_dispatcher<orderA, 3>(stream, N, sft, sft_iter, orderX, X, incx, A, inca); return;
    default: return;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, int32_t option, int64_t N, int32_t sft, uint32_t sft_iter, int32_t orderX, const int32_t* X, int64_t incx, int32_t orderA, uint64_t* A, int64_t inca) {
  switch (orderA) {
    case 1: acc_dispatcher<1>(stream, option, N, uint32_t(sft), uint32_t(sft_iter), uint32_t(orderX), X, incx, A, inca); return;
    case 2: acc_dispatcher<2>(stream, option, N, uint32_t(sft), uint32_t(sft_iter), uint32_t(orderX), X, incx, A, inca); return;
    case 3: acc_dispatcher<3>(stream, option, N, uint32_t(sft), uint32_t(sft_iter), uint32_t(orderX), X, incx, A, inca); return;
    default: return;
  }
}
