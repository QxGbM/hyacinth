
#include <internal.hpp>
#include <int_fp_quantize.hpp>

template<int32_t orderA, int32_t orderX, int32_t beta>
__global__ void i32_accum_kernel(uint32_t sft, uint32_t sft_iter, int64_t N, const int32_t* __restrict__ X, int64_t ldx, int64_t strideX, uint64_t* __restrict__ A, int64_t strideA) {
  int64_t y = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (y < N) {
    int64_t x = int64_t(blockIdx.y);
    uint64_t acc[orderA]; A = &A[y + x * N]; X = &X[y + x * ldx];

    if constexpr(beta) {
      if constexpr(uint32_t(0) < orderA) { acc[0] = A[0]; }
      if constexpr(uint32_t(1) < orderA) { acc[1] = A[strideA]; }
      if constexpr(uint32_t(2) < orderA) { acc[2] = A[strideA + strideA]; }
    }
    else {
      if constexpr(uint32_t(0) < orderA) { acc[0] = uint64_t(0); }
      if constexpr(uint32_t(1) < orderA) { acc[1] = uint64_t(0); }
      if constexpr(uint32_t(2) < orderA) { acc[2] = uint64_t(0); }
    }

    if constexpr(uint32_t(0) < orderX) { device::int8::add_shifted(acc, int64_t(*X), sft); }
    if constexpr(uint32_t(1) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(2) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(3) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(4) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(5) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(6) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(7) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(8) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(9) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(10) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }
    if constexpr(uint32_t(11) < orderX) { device::int8::add_shifted(acc, int64_t(*(X += strideX)), sft += sft_iter); }

    if constexpr(uint32_t(0) < orderA) { A[0] = acc[0]; }
    if constexpr(uint32_t(1) < orderA) { A[strideA] = acc[1]; }
    if constexpr(uint32_t(2) < orderA) { A[strideA + strideA] = acc[2]; }
  }
}

template <int32_t orderA, int32_t orderX>
inline void acc_dispatcher(cudaStream_t stream, int32_t beta, int64_t N, uint32_t sft, uint32_t sft_iter, const int32_t* X, int64_t ldx, uint64_t* A) {
  constexpr int32_t block_threads = 512;
  dim3 grid(uint32_t(N + 511) >> 9, uint32_t(N), uint32_t(1));
  int64_t strideX = ldx * N, strideA = N * N + N;
  switch (beta) {
    case 0: i32_accum_kernel<orderA, orderX, 0> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, ldx, strideX, A, strideA); return;
    case 1: i32_accum_kernel<orderA, orderX, 1> <<< grid, block_threads, 0, stream >>> (sft, sft_iter, N, X, ldx, strideX, A, strideA); return;
    default: return;
  }
}

template <uint32_t orderA>
inline void acc_dispatcher(cudaStream_t stream, int32_t beta, int64_t N, uint32_t sft, uint32_t sft_iter, int32_t orderX, const int32_t* X, int64_t ldx, uint64_t* A) {
  switch (orderX) {
    case 1: acc_dispatcher<orderA, 1>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 2: acc_dispatcher<orderA, 2>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 3: acc_dispatcher<orderA, 3>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 4: acc_dispatcher<orderA, 4>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 5: acc_dispatcher<orderA, 5>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 6: acc_dispatcher<orderA, 6>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 7: acc_dispatcher<orderA, 7>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 8: acc_dispatcher<orderA, 8>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 9: acc_dispatcher<orderA, 9>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 10: acc_dispatcher<orderA, 10>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 11: acc_dispatcher<orderA, 11>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    case 12: acc_dispatcher<orderA, 12>(stream, beta, N, sft, sft_iter, X, ldx, A); return;
    default: return;
  }
}

void internal::int8::accumulate_i32tensor(cudaStream_t stream, int32_t beta, int32_t N, int32_t sft, uint32_t sft_iter, int32_t orderX, const int32_t* X, int32_t ldx, int32_t orderA, uint64_t* A) {
  switch (orderA) {
    case 1: acc_dispatcher<1>(stream, beta, int64_t(N), uint32_t(sft), uint32_t(sft_iter), orderX, X, int64_t(ldx), A); return;
    case 2: acc_dispatcher<2>(stream, beta, int64_t(N), uint32_t(sft), uint32_t(sft_iter), orderX, X, int64_t(ldx), A); return;
    case 3: acc_dispatcher<3>(stream, beta, int64_t(N), uint32_t(sft), uint32_t(sft_iter), orderX, X, int64_t(ldx), A); return;
    default: return;
  }
}
