
#include <internal.hpp>
#include <int_fp_quantize.hpp>

__global__ void i32_remainder_normalize_kernel(int64_t N, int32_t* __restrict__ X) {
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < N) {
    i += N * int64_t(blockIdx.y);
    if (int32_t(blockIdx.y) & 1) 
    { int32_t r255 = X[i]; X[i] = int32_t(device::int8::fast_rem_u32<255>(uint32_t(r255))) - int32_t(r255 < 0); }
    else
      X[i] = int32_t(uint8_t(X[i]));
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::normalize_remainder_i32tensor(cudaStream_t stream, int64_t N, int32_t* X, int32_t nbatch) {
  dim3 grid((uint32_t(N) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(nbatch) << 1);
  i32_remainder_normalize_kernel <<< grid, block_threads, 0, stream >>> (N, X);
}
