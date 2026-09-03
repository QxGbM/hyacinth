
#include <hyacin.h>
#ifndef NO_NCCL

#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <stdexcept>

template <int32_t hi_bits> __device__ __forceinline__ uint64_t sign_bits(uint64_t a) {
  static_assert(0 <= hi_bits && hi_bits <= 64);
  constexpr uint64_t mask = hi_bits ? (uint64_t(0xffffffffffffffffllu) << (64 - hi_bits)) : uint64_t(0);
  return mask & (-(a >> 63));
}

template <int32_t LEN, int32_t ORDER> __device__ __forceinline__ void load_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 < ORDER && 0 < LEN) { a[0] = *in; }
  if constexpr(1 < ORDER && 1 < LEN) { a[1] = *(in += stride); }
  if constexpr(2 < ORDER && 2 < LEN) { a[2] = *(in += stride); }
}

template <int32_t OFF, int32_t LEN, int32_t ORDER> __device__ __forceinline__ void store_i(const uint64_t (&a)[ORDER], uint64_t* out, int64_t stride) {
  if constexpr(OFF < ORDER && 0 < LEN) { *out = a[OFF]; }
  if constexpr(OFF + 1 < ORDER && 1 < LEN) { *(out += stride) = a[OFF + 1]; }
  if constexpr(OFF + 2 < ORDER && 2 < LEN) { *(out += stride) = a[OFF + 2]; }
  if constexpr(OFF + 3 < ORDER && 3 < LEN) { *(out += stride) = a[OFF + 3]; }
  if constexpr(OFF + 4 < ORDER && 4 < LEN) { *(out += stride) = a[OFF + 4]; }
  if constexpr(OFF + 5 < ORDER && 5 < LEN) { *(out += stride) = a[OFF + 5]; }
}

template <int32_t ORDER> __device__ __forceinline__ void conv_u32(uint64_t (&a)[ORDER]) {
  constexpr uint64_t i32 = uint64_t(0xffffffffllu);
  uint64_t b0 = a[0] & i32;
  if constexpr(2 < ORDER) {
    uint64_t b1 = ((a[0] >> 32) | (a[1] << 31)) & i32;
    uint64_t b2 = (a[1] >> 1) & i32;
    if constexpr(4 < ORDER) {
      uint64_t b3 = ((a[1] >> 33) | (a[2] << 30)) & i32;
      uint64_t b4 = (a[2] >> 2) & i32;
      uint64_t b5 = (a[2] >> 34) | sign_bits<34>(a[2]);
      a[3] = b3; a[4] = b4; a[5] = b5;
    } else { a[3] = (a[1] >> 33) | sign_bits<33>(a[1]); }
    a[1] = b1; a[2] = b2; 
  } else { a[1] = (a[0] >> 32) | sign_bits<32>(a[0]); }
  a[0] = b0;
}

__device__ __forceinline__ void conv_u43(uint64_t (&a)[3]) {
  constexpr uint64_t i43 = uint64_t(0x7ffffffffffllu);
  uint64_t b0 = a[0] & i43;
  uint64_t b1 = ((a[0] >> 43) | (a[1] << 20)) & i43;
  uint64_t b2 = (a[1] >> 23) | sign_bits<23>(a[1]);
  a[0] = b0; a[1] = b1; a[2] = b2;
}

__device__ __forceinline__ void conv_u48(uint64_t (&a)[4]) {
  constexpr uint64_t i48 = uint64_t(0xffffffffffffllu);
  uint64_t b0 = a[0] & i48;
  uint64_t b1 = ((a[0] >> 48) | (a[1] << 15)) & i48;
  uint64_t b2 = ((a[1] >> 33) | (a[2] << 30)) & i48;
  uint64_t b3 = (a[2] >> 18) | sign_bits<18>(a[2]);
  a[0] = b0; a[1] = b1; a[2] = b2; a[3] = b3;
}

__device__ __forceinline__ void conv_u38(uint64_t (&a)[5]) {
  constexpr uint64_t i38 = uint64_t(0x3fffffffffllu);
  uint64_t b0 = a[0] & i38;
  uint64_t b1 = ((a[0] >> 38) | (a[1] << 25)) & i38;
  uint64_t b2 = (a[1] >> 13) & i38;
  uint64_t b3 = ((a[1] >> 51) | (a[2] << 12)) & i38;
  uint64_t b4 = (a[2] >> 26) | sign_bits<26>(a[2]);
  a[0] = b0; a[1] = b1; a[2] = b2; a[3] = b3; a[4] = b4;
}

template <int32_t op> __global__ void limbs_convert_kernel(int64_t N, uint64_t* __restrict__ A, uint64_t* __restrict__ E) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    constexpr int32_t LimbCount_lis[] = { 2, 3, 4, 4, 5, 6, 2, 3, 4, 4, 5, 6 };
    constexpr int32_t LimbCount = LimbCount_lis[op];
    uint64_t a[LimbCount]; A = &A[i]; E = &E[i];
    if constexpr(op == 0) { load_i<1>(a, A, N); conv_u32(a); store_i<0, 1>(a, A, N); store_i<1, 1>(a, E, N); } else
    if constexpr(op == 1) { load_i<2>(a, A, N); conv_u43(a); store_i<0, 2>(a, A, N); store_i<2, 1>(a, E, N); } else
    if constexpr(op == 2) { load_i<2>(a, A, N); conv_u32(a); store_i<0, 2>(a, A, N); store_i<2, 2>(a, E, N); } else
    if constexpr(op == 3) { load_i<3>(a, A, N); conv_u48(a); store_i<0, 3>(a, A, N); store_i<3, 1>(a, E, N); } else
    if constexpr(op == 4) { load_i<3>(a, A, N); conv_u38(a); store_i<0, 3>(a, A, N); store_i<3, 2>(a, E, N); } else
    if constexpr(op == 5) { load_i<3>(a, A, N); conv_u32(a); store_i<0, 3>(a, A, N); store_i<3, 2>(a, E, N); } else
    if constexpr(op == 6) { load_i<1>(a, &A[N], N); conv_u32(a); store_i<0, 2>(a, E, N); load_i<1>(a, A, N); conv_u32(a); store_i<0, 2>(a, A, N); } else
    if constexpr(op == 7) { load_i<2>(a, &A[N * int64_t(2)], N); conv_u43(a); store_i<0, 1>(a, &A[N * int64_t(3)], N); store_i<1, 2>(a, E, N); load_i<2>(a, A, N); conv_u43(a); store_i<0, 3>(a, A, N); } else
    if constexpr(op == 8) { load_i<2>(a, &A[N * int64_t(2)], N); conv_u32(a); store_i<0, 4>(a, E, N); load_i<2>(a, A, N); conv_u32(a); store_i<0, 4>(a, A, N); } else
    if constexpr(op == 9) { load_i<3>(a, &A[N * int64_t(3)], N); conv_u48(a); store_i<0, 2>(a, &A[N * int64_t(4)], N); store_i<2, 2>(a, E, N); load_i<3>(a, A, N); conv_u48(a); store_i<0, 4>(a, A, N); } else
    if constexpr(op == 10) { load_i<3>(a, &A[N * int64_t(3)], N); conv_u38(a); store_i<0, 1>(a, &A[N * int64_t(5)], N); store_i<1, 4>(a, E, N); load_i<3>(a, A, N); conv_u38(a); store_i<0, 5>(a, A, N); } else
    if constexpr(op == 11) { load_i<3>(a, &A[N * int64_t(3)], N); conv_u32(a); store_i<0, 6>(a, E, N); load_i<3>(a, A, N); conv_u32(a); store_i<0, 6>(a, A, N); }
  }
}

template <int32_t SFT, int32_t OFF, int32_t LEN, int32_t ORDER> __device__ __forceinline__ void accum_i(uint64_t (&a)[ORDER], const uint64_t* in, int64_t stride) {
  if constexpr(0 == OFF && 0 < LEN) { a[0] = *in; if constexpr(1 < ORDER) a[1] = uint64_t(0); if constexpr(2 < ORDER) a[2] = uint64_t(0); }
    else if constexpr(0 < LEN) { constexpr uint32_t s = uint32_t(SFT * OFF); device::int8::add_shifted(a, int64_t(*in), s); }
  if constexpr(1 < LEN) { constexpr uint32_t s = uint32_t(SFT * (OFF + 1)); device::int8::add_shifted(a, int64_t(*(in += stride)), s); }
  if constexpr(2 < LEN) { constexpr uint32_t s = uint32_t(SFT * (OFF + 2)); device::int8::add_shifted(a, int64_t(*(in += stride)), s); }
  if constexpr(3 < LEN) { constexpr uint32_t s = uint32_t(SFT * (OFF + 3)); device::int8::add_shifted(a, int64_t(*(in += stride)), s); }
  if constexpr(4 < LEN) { constexpr uint32_t s = uint32_t(SFT * (OFF + 4)); device::int8::add_shifted(a, int64_t(*(in += stride)), s); }
  if constexpr(5 < LEN) { constexpr uint32_t s = uint32_t(SFT * (OFF + 5)); device::int8::add_shifted(a, int64_t(*(in += stride)), s); }
}

template <int32_t op> __global__ void limbs_accum_kernel(int64_t N, uint64_t* __restrict__ A, const uint64_t* __restrict__ E) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    constexpr int32_t orderA_lis[] = { 1, 2, 2, 3, 3, 3, 1, 2, 2, 3, 3, 3 };
    constexpr int32_t orderA = orderA_lis[op];
    uint64_t a[orderA]; A = &A[i]; E = &E[i];
    if constexpr(op == 0) { accum_i<32, 0, 1>(a, A, N); accum_i<32, 1, 1>(a, E, N); store_i<0, 1>(a, A, N); } else
    if constexpr(op == 1) { accum_i<43, 0, 2>(a, A, N); accum_i<43, 2, 1>(a, E, N); store_i<0, 2>(a, A, N); } else
    if constexpr(op == 2) { accum_i<32, 0, 2>(a, A, N); accum_i<32, 2, 2>(a, E, N); store_i<0, 2>(a, A, N); } else
    if constexpr(op == 3) { accum_i<48, 0, 3>(a, A, N); accum_i<48, 3, 1>(a, E, N); store_i<0, 3>(a, A, N); } else
    if constexpr(op == 4) { accum_i<38, 0, 3>(a, A, N); accum_i<38, 3, 2>(a, E, N); store_i<0, 3>(a, A, N); } else
    if constexpr(op == 5) { accum_i<32, 0, 3>(a, A, N); accum_i<32, 3, 3>(a, E, N); store_i<0, 3>(a, A, N); } else
    if constexpr(op == 6) { accum_i<32, 0, 2>(a, A, N); store_i<0, 1>(a, A, N); accum_i<32, 0, 2>(a, E, N); store_i<0, 1>(a, &A[N], N); } else
    if constexpr(op == 7) { accum_i<43, 0, 3>(a, A, N); store_i<0, 2>(a, A, N); accum_i<43, 0, 1>(a, &A[N * int64_t(3)], N); accum_i<43, 1, 2>(a, E, N); store_i<0, 2>(a, &A[N * int64_t(2)], N); } else
    if constexpr(op == 8) { accum_i<32, 0, 4>(a, A, N); store_i<0, 2>(a, A, N); accum_i<32, 0, 4>(a, E, N); store_i<0, 2>(a, &A[N * int64_t(2)], N); } else
    if constexpr(op == 9) { accum_i<48, 0, 4>(a, A, N); store_i<0, 3>(a, A, N); accum_i<48, 0, 2>(a, &A[N * int64_t(4)], N); accum_i<48, 2, 2>(a, E, N); store_i<0, 3>(a, &A[N * int64_t(3)], N); } else
    if constexpr(op == 10) { accum_i<38, 0, 5>(a, A, N); store_i<0, 3>(a, A, N); accum_i<38, 0, 1>(a, &A[N * int64_t(5)], N); accum_i<38, 1, 4>(a, E, N); store_i<0, 3>(a, &A[N * int64_t(3)], N); } else
    if constexpr(op == 11) { accum_i<32, 0, 6>(a, A, N); store_i<0, 3>(a, A, N); accum_i<32, 0, 6>(a, E, N); store_i<0, 3>(a, &A[N * int64_t(3)], N); }
  }
}

template <int32_t op> inline void conv_reduction(cudaStream_t stream, int64_t N, uint64_t* A, int64_t lenA, uint64_t* E, int64_t lenE, ncclComm_t col_comm) {
  constexpr int32_t block_threads = 512;
  int32_t grid = int32_t((N + int64_t(511)) >> 9);
  limbs_convert_kernel<op> <<< grid, block_threads, 0, stream >>> (N, A, E);
  ncclGroupStart();
  ncclAllReduce(A, A, lenA, ncclUint64, ncclSum, col_comm, stream);
  ncclAllReduce(E, E, lenE, ncclUint64, ncclSum, col_comm, stream);
  ncclGroupEnd();
  limbs_accum_kernel<op> <<< grid, block_threads, 0, stream >>> (N, A, E);
}

extern "C" void hyacinXAllReduce1Drow(hyacinHandle_t handle, int32_t Complex, int32_t orderA, int64_t N, uint64_t* A) {
  if (Complex <= 0 || orderA <= 0 || N <= int64_t(0) || handle.col_comm == nullptr) { return; }

  Timer::register_comm(handle.cudaStream, handle.timer);
  int32_t comm_size; ncclCommCount(handle.col_comm, &comm_size);
  if (comm_size == 1) { return; } else if (comm_size <= 0) { throw std::runtime_error("Invalid NCCL communicator at All-reduce"); }

  int32_t LimbCount = orderA + 1 + int32_t(orderA == 2 && 1048576 < comm_size) + int32_t(orderA == 3 && 32768 < comm_size) + int32_t(orderA == 3 && 33554432 < comm_size);
  int64_t lenA = int64_t(Complex) * int64_t(orderA) * N, lenE = int64_t(Complex) * int64_t(LimbCount - orderA) * N;
  uint64_t* devE = nullptr;
  if (cudaSuccess != cudaMallocAsync((void**)&devE, uint64_t(lenE) * uint64_t(sizeof(uint64_t)), handle.cudaStream))
    throw std::runtime_error("Workspace allocation failed at All-reduce.");

  if (Complex == 1 && orderA == 1 && LimbCount == 2) { conv_reduction<0>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 1 && orderA == 2 && LimbCount == 3) { conv_reduction<1>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 1 && orderA == 2 && LimbCount == 4) { conv_reduction<2>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 1 && orderA == 3 && LimbCount == 4) { conv_reduction<3>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 1 && orderA == 3 && LimbCount == 5) { conv_reduction<4>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 1 && orderA == 3 && LimbCount == 6) { conv_reduction<5>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 1 && LimbCount == 2) { conv_reduction<6>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 2 && LimbCount == 3) { conv_reduction<7>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 2 && LimbCount == 4) { conv_reduction<8>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 3 && LimbCount == 4) { conv_reduction<9>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 3 && LimbCount == 5) { conv_reduction<10>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); } else
  if (Complex == 2 && orderA == 3 && LimbCount == 6) { conv_reduction<11>(handle.cudaStream, N, A, lenA, devE, lenE, handle.col_comm); }
  cudaFreeAsync(devE, handle.cudaStream);
}

#else
extern "C" void hyacinXAllReduce1Drow(hyacinHandle_t, int32_t, int32_t, int64_t, uint64_t*) {}
#endif
