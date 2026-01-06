
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_selector.hpp>
#include <algorithm>

template<int32_t len> struct i32_array { int32_t arr[len]; };

__global__ void i32_remainder_normalize_kernel(int64_t N, int32_t* __restrict__ X) {
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < N) {
    i += N * int64_t(blockIdx.y);
    if (int32_t(blockIdx.y) & 1) 
    { int32_t r255 = X[i]; X[i] = int32_t(device::int8::fast_rem_u32_255(uint32_t(r255))) - int32_t(r255 < 0); }
    else
      X[i] = int32_t(uint8_t(X[i]));
  }
}

template<int32_t orderA, int32_t orderPD, int32_t orderM, uint64_t MO, uint64_t MINV, uint64_t P0, uint64_t P1, uint64_t P2, uint64_t P3, int32_t beta, int32_t pd_len>
__global__ void i32_crt_accum_kernel(i32_array<pd_len> pd, int64_t N, const int32_t* __restrict__ X, uint64_t* __restrict__ A) {
  int64_t i = int64_t(blockIdx.x) * int64_t(blockDim.x) + int64_t(threadIdx.x);
  if (i < N) {
    uint64_t acc[orderA]; int32_t rem[8];
    int64_t iter = i;

    if constexpr(beta) {
      #pragma unroll
      for (int32_t r = 0; r < orderA; ++r)
      { acc[r] = A[iter]; iter += N; } iter = i;
    }
    else {
      #pragma unroll
      for (int32_t r = 0; r < orderA; ++r)
      { acc[r] = uint64_t(0); }
    }

    constexpr int32_t r_len = orderM << 1;
    #pragma unroll
    for (int32_t r = 0; r < r_len; ++r)
    { rem[r] = X[iter]; iter += N; } iter = i;
    device::int8::crt_recover<MO, MINV>(rem);

    #pragma unroll
    for (int32_t r = 0; r < orderM; ++r) {
      int32_t ri = (acc[orderA - 1] >> 63) ? rem[r << 1] : rem[(r << 1) + 1];

      #pragma unroll
      for (int32_t p = 0; p < orderPD; ++p) {
        int32_t idx = r * orderPD + p;
        uint32_t lo = uint32_t(ri * pd.arr[idx]), hi = uint32_t(__mulhi(ri, pd.arr[idx]));
        int64_t prod = int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
        device::int8::add_shifted(acc, prod, uint32_t(p * 31));
      }
    }

    if constexpr(P0) 
      if (acc[orderA - 1] >> 63) {
        device::int8::add_shifted(acc, P0, uint32_t(0));
        if constexpr(1 < orderA) device::int8::add_shifted(acc, P1, uint32_t(63));
        if constexpr(2 < orderA) device::int8::add_shifted(acc, P2, uint32_t(126));
        if constexpr(3 < orderA) device::int8::add_shifted(acc, P3, uint32_t(189));
      }

    #pragma unroll
    for (int32_t r = 0; r < orderA; ++r)
    { A[iter] = acc[r]; iter += N; }
  }
}

constexpr int32_t block_threads = 512;

void internal::int8::normalize_remainder_i32tensor(cudaStream_t stream, int64_t N, int32_t* X, int32_t nbatch) {
  dim3 grid((uint32_t(N) + uint32_t(block_threads - 1)) / uint32_t(block_threads), uint32_t(nbatch) << 1);
  i32_remainder_normalize_kernel <<< grid, block_threads, 0, stream >>> (N, X);
}

template <int32_t n_moduli, int32_t iter>
inline void crt_acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, const int32_t* X, uint64_t* A) {
  constexpr int32_t orderM = CRT::active_moduli(n_moduli, iter);

  if constexpr(0 < orderM) {
    constexpr int32_t orderA = CRT::order_p(n_moduli), orderPD = CRT::order_pd(n_moduli);
    constexpr uint64_t MO = CRT::modular(iter), MINV = CRT::modular_inv(n_moduli, iter);

    constexpr int32_t pd_len = orderM * orderPD;
    i32_array<pd_len> pd; std::copy_n(CRT::p_div(n_moduli, iter), pd_len, &pd.arr[0]);
    int32_t grid = int32_t((N + int64_t(block_threads - 1)) / int64_t(block_threads));

    if (option & 2) {
      constexpr uint64_t P0 = CRT::domain_p(n_moduli, 0), P1 = CRT::domain_p(n_moduli, 1);
      constexpr uint64_t P2 = CRT::domain_p(n_moduli, 2), P3 = CRT::domain_p(n_moduli, 3);
      if (option & 1)
        i32_crt_accum_kernel<orderA, orderPD, orderM, MO, MINV, P0, P1, P2, P3, 1> <<< grid, block_threads, 0, stream >>> (pd, N, X, A);
      else
        i32_crt_accum_kernel<orderA, orderPD, orderM, MO, MINV, P0, P1, P2, P3, 0> <<< grid, block_threads, 0, stream >>> (pd, N, X, A);
    }
    else {
      constexpr uint64_t z = uint64_t(0);
      if (option & 1)
        i32_crt_accum_kernel<orderA, orderPD, orderM, MO, MINV, z, z, z, z, 1> <<< grid, block_threads, 0, stream >>> (pd, N, X, A);
      else
        i32_crt_accum_kernel<orderA, orderPD, orderM, MO, MINV, z, z, z, z, 0> <<< grid, block_threads, 0, stream >>> (pd, N, X, A);
    }
  }
}

template <int32_t iter>
inline void crt_acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, int32_t n_moduli, const int32_t* X, uint64_t* A) {
  switch (n_moduli) {
    case 2: crt_acc_dispatcher<2, iter>(stream, option, N, X, A); break;
    case 3: crt_acc_dispatcher<3, iter>(stream, option, N, X, A); break;
    case 4: crt_acc_dispatcher<4, iter>(stream, option, N, X, A); break;
    case 5: crt_acc_dispatcher<5, iter>(stream, option, N, X, A); break;
    case 6: crt_acc_dispatcher<6, iter>(stream, option, N, X, A); break;
    case 7: crt_acc_dispatcher<7, iter>(stream, option, N, X, A); break;
    case 8: crt_acc_dispatcher<8, iter>(stream, option, N, X, A); break;
    case 9: crt_acc_dispatcher<9, iter>(stream, option, N, X, A); break;
    case 10: crt_acc_dispatcher<10, iter>(stream, option, N, X, A); break;
    case 11: crt_acc_dispatcher<11, iter>(stream, option, N, X, A); break;
    case 12: crt_acc_dispatcher<12, iter>(stream, option, N, X, A); break;
    case 13: crt_acc_dispatcher<13, iter>(stream, option, N, X, A); break;
    case 14: crt_acc_dispatcher<14, iter>(stream, option, N, X, A); break;
    default: break;
  }
}

void internal::int8::accumulate_remainder_i32tensor(cudaStream_t stream, int32_t option, int64_t N, int32_t n_moduli, int32_t iter, const int32_t* X, uint64_t* A) {
  switch (iter) {
    case 0: crt_acc_dispatcher<0>(stream, option, N, n_moduli, X, A); break;
    case 1: crt_acc_dispatcher<1>(stream, option, N, n_moduli, X, A); break;
    case 2: crt_acc_dispatcher<2>(stream, option, N, n_moduli, X, A); break;
    case 3: crt_acc_dispatcher<3>(stream, option, N, n_moduli, X, A); break;
    default: break;
  }
}
