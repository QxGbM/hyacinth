
#include <hyacinth.hpp>
#include <internal.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

template <uint32_t beta, uint32_t N> struct normalize_i32 {
  int4* __restrict__ A;
  uint64_t stride;
  normalize_i32(uint64_t M, int32_t* A) : A((int4*)A), stride(M >> 2) {}

  __device__ __forceinline__ void operator()(uint64_t i) {
    constexpr uint32_t BASE = device::Config::exp_base;
    constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;

    int4 A_i = A[i];
    A[i] = make_int4(A_i.x & iBASE, A_i.y & iBASE, A_i.z & iBASE, A_i.w & iBASE);
    A_i = make_int4(A_i.x >> BASE, A_i.y >> BASE, A_i.z >> BASE, A_i.w >> BASE);

    #pragma unroll
    for (uint32_t k = 1; k < N; ++k) {
      uint64_t j = i + uint64_t(k) * stride;
      int4 A_k = A[j];
      int4 val = make_int4(A_i.x + A_k.x, A_i.y + A_k.y, A_i.z + A_k.z, A_i.w + A_k.w);

      A[j] = make_int4(val.x & iBASE, val.y & iBASE, val.z & iBASE, val.w & iBASE);
      A_i = make_int4(val.x >> BASE, val.y >> BASE, val.z >> BASE, val.w >> BASE);
    }

    uint64_t j = i + uint64_t(N) * stride;
    if constexpr (beta) {
      int4 A_k = A[j];
      A[j] = make_int4(A_i.x + A_k.x, A_i.y + A_k.y, A_i.z + A_k.z, A_i.w + A_k.w);
    }
    else
      A[j] = A_i;
  }
};

template <uint32_t order>
void normalization_dispatcher(cudaStream_t stream, uint64_t M, int32_t beta, int32_t* A) {
  thrust::counting_iterator<uint64_t> iter(0);
  if (beta) {
    normalize_i32<1, order> normalize(M, A);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, normalize.stride, normalize);
  }
  else {
    normalize_i32<0, order> normalize(M, A);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, normalize.stride, normalize);
  }
}

void internal::int8::i32_normalization(cudaStream_t stream, uint64_t M, int32_t order, int32_t beta, int32_t* A) {
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

