
#include <hyacinth.hpp>
#include <internal.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t beta, uint32_t N> struct normalize_i32 {
  int32_t* __restrict__ A;
  int64_t M;
  normalize_i32(int64_t M, int32_t* A) : A(A), M(M) {}

  __device__ __forceinline__ void operator()(int64_t i) {
    constexpr uint32_t BASE = device::Config::exp_base;
    constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;

    int32_t A_i = A[i];
    A[i] = A_i & iBASE; A_i = A_i >> BASE;

    #pragma unroll
    for (uint32_t k = 1; k < N; ++k) {
      int64_t j = i + int64_t(k) * M;
      int32_t A_k = A[j], val = A_i + A_k;
      A[j] = val & iBASE; A_i = val >> BASE;
    }

    int64_t j = i + int64_t(N) * M;
    if constexpr (beta) { int32_t A_k = A[j]; A[j] = A_i + A_k; }
      else { A[j] = A_i; }
  }
};

template <uint32_t order>
inline void normalization_dispatcher(cudaStream_t stream, int64_t M, int32_t beta, int32_t* A) {
  thrust::counting_iterator<int64_t> iter(0);
  if (beta)
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, normalize_i32<1, order>(M, A));
  else
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, M, normalize_i32<0, order>(M, A));
}

void internal::int8::i32_normalization(cudaStream_t stream, int64_t M, int32_t order, int32_t beta, int32_t* A) {
  switch(order) {
    case 1: normalization_dispatcher<1>(stream, M, beta, A); break;
    case 2: normalization_dispatcher<2>(stream, M, beta, A); break;
    case 3: normalization_dispatcher<3>(stream, M, beta, A); break;
    case 4: normalization_dispatcher<4>(stream, M, beta, A); break;
    case 5: normalization_dispatcher<5>(stream, M, beta, A); break;
    case 6: normalization_dispatcher<6>(stream, M, beta, A); break;
    case 7: normalization_dispatcher<7>(stream, M, beta, A); break;
    case 8: normalization_dispatcher<8>(stream, M, beta, A); break;
    case 9: normalization_dispatcher<9>(stream, M, beta, A); break;
    default: break;
  }

  if constexpr (device::Config::exp_base < 7)
    switch(order) {
      case 10: normalization_dispatcher<10>(stream, M, beta, A); break;
      case 11: normalization_dispatcher<11>(stream, M, beta, A); break;
      case 12: normalization_dispatcher<12>(stream, M, beta, A); break;
      default: break;
    }

  if constexpr (device::Config::exp_base < 5)
    switch(order) {
      case 13: normalization_dispatcher<13>(stream, M, beta, A); break;
      case 14: normalization_dispatcher<14>(stream, M, beta, A); break;
      case 15: normalization_dispatcher<15>(stream, M, beta, A); break;
      default: break;
    }
}

