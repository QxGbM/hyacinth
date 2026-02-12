
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <crt_selector.hpp>
#include <algorithm>

template<int32_t len> struct i32_array { int32_t arr[len]; };

template<int32_t orderM, int32_t orderA, int32_t orderPD, uint64_t MO, uint64_t MINV, uint64_t R32, int64_t P0, int64_t P1, int64_t P2, int32_t beta, int32_t pd_len>
__global__ void i32_crt_accum_kernel(i32_array<pd_len> pd, int64_t N, const int32_t* __restrict__ X, int64_t incx, uint64_t* __restrict__ A, int64_t inca) {
  int64_t i = (int64_t(blockIdx.x) << 9) + int64_t(threadIdx.x);
  if (i < N) {
    constexpr int32_t m[8]{ uint8_t(MO) ? uint32_t(uint8_t(MO)) : uint32_t(256), 
      uint8_t(MO >> 8), uint8_t(MO >> 16), uint8_t(MO >> 24), uint8_t(MO >> 32), uint8_t(MO >> 40), uint8_t(MO >> 48), uint8_t(MO >> 56) };
    uint64_t acc[orderA]; int64_t iter = i;
    int32_t rem[8], (*rem_lo)[4] = (int32_t (*)[4])(&rem[0]), (*rem_hi)[4] = (int32_t (*)[4])(&rem[4]);

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
    for (int32_t r = 0; r < orderM; ++r)
    { rem[r] = X[iter]; iter += incx; } iter = i;

    constexpr uint32_t m_lo = uint32_t(MO), m_hi = uint32_t(MO >> 32);
    constexpr uint32_t i_lo = uint32_t(MINV), i_hi = uint32_t(MINV >> 32);
    constexpr uint32_t r_lo = uint32_t(R32), r_hi = uint32_t(R32 >> 32);
    device::int8::crt_recover<m_lo, i_lo, r_lo>(*rem_lo);
    device::int8::crt_recover<m_hi, i_hi, r_hi>(*rem_hi);
    const int32_t* pdx = &pd.arr[0];

    #pragma unroll
    for (int32_t r = 0; r < orderM; ++r) {
      int32_t ri = (acc[orderA - 1] >> 63) ? rem[r] : (rem[r] - m[r]);

      #pragma unroll
      for (int32_t p = 0; p < orderPD; ++p) {
        int32_t pdi = *pdx; ++pdx;
        uint32_t lo = uint32_t(ri * pdi), hi = uint32_t(__mulhi(ri, pdi));
        int64_t prod = int64_t(uint64_t(lo) | (uint64_t(hi) << 32));
        device::int8::add_shifted(acc, prod, uint32_t(p * 31));
      }
    }

    if constexpr(P0) 
      if (acc[orderA - 1] >> 63) {
        device::int8::add_shifted(acc, P0, uint32_t(0));
        if constexpr(P1) device::int8::add_shifted(acc, P1, uint32_t(63));
        if constexpr(P2) device::int8::add_shifted(acc, P2, uint32_t(126));
      }

    #pragma unroll
    for (int32_t r = 0; r < orderA; ++r)
    { A[iter] = acc[r]; iter += inca; }
  }
}

template <int32_t orderX, int32_t orderA, int32_t orderPD, int32_t iter>
inline void crt_acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  constexpr int32_t orderM = CRT::active_moduli(orderX, iter), pd_len = orderM * orderPD, block_threads = 512;

  if constexpr(0 < orderM) {
    constexpr uint64_t MO = CRT::modular(iter), MINV = CRT::modular_inv(orderX, iter), R32 = CRT::inv_r32(orderX, iter);
    constexpr int64_t P0 = CRT::domain_p(orderX, 0), P1 = CRT::domain_p(orderX, 1), P2 = CRT::domain_p(orderX, 2), z = int64_t(0);

    i32_array<pd_len> pd; std::copy_n(CRT::p_div(orderX, iter), pd_len, &pd.arr[0]);
    int32_t grid = int32_t((N + int64_t(511)) >> 9);
    int32_t accum = int32_t(option & 1 == 1), last = int32_t(option & 2 == 2);

    if (accum && last)
      i32_crt_accum_kernel<orderM, orderA, orderPD, MO, MINV, R32, P0, P1, P2, 1> <<< grid, block_threads, 0, stream >>> (pd, N, X, incx, A, inca);
    else if (accum)
      i32_crt_accum_kernel<orderM, orderA, orderPD, MO, MINV, R32, z, z, z, 1> <<< grid, block_threads, 0, stream >>> (pd, N, X, incx, A, inca);
    else if (last)
      i32_crt_accum_kernel<orderM, orderA, orderPD, MO, MINV, R32, P0, P1, P2, 0> <<< grid, block_threads, 0, stream >>> (pd, N, X, incx, A, inca);
    else
      i32_crt_accum_kernel<orderM, orderA, orderPD, MO, MINV, R32, z, z, z, 0> <<< grid, block_threads, 0, stream >>> (pd, N, X, incx, A, inca);
  }
}

template <int32_t iter>
inline void crt_acc_dispatcher(cudaStream_t stream, int32_t option, int64_t N, int32_t orderX, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  switch (orderX) {
    case 2: crt_acc_dispatcher<2, 1, 1, iter>(stream, option, N, X, incx, A, inca); break;
    case 3: crt_acc_dispatcher<3, 1, 1, iter>(stream, option, N, X, incx, A, inca); break;
    case 4: crt_acc_dispatcher<4, 1, 1, iter>(stream, option, N, X, incx, A, inca); break;
    case 5: crt_acc_dispatcher<5, 1, 2, iter>(stream, option, N, X, incx, A, inca); break;
    case 6: crt_acc_dispatcher<6, 1, 2, iter>(stream, option, N, X, incx, A, inca); break;
    case 7: crt_acc_dispatcher<7, 1, 2, iter>(stream, option, N, X, incx, A, inca); break;
    case 8: crt_acc_dispatcher<8, 2, 2, iter>(stream, option, N, X, incx, A, inca); break;
    case 9: crt_acc_dispatcher<9, 2, 3, iter>(stream, option, N, X, incx, A, inca); break;
    case 10: crt_acc_dispatcher<10, 2, 3, iter>(stream, option, N, X, incx, A, inca); break;
    case 11: crt_acc_dispatcher<11, 2, 3, iter>(stream, option, N, X, incx, A, inca); break;
    case 12: crt_acc_dispatcher<12, 2, 3, iter>(stream, option, N, X, incx, A, inca); break;
    case 13: crt_acc_dispatcher<13, 2, 4, iter>(stream, option, N, X, incx, A, inca); break;
    case 14: crt_acc_dispatcher<14, 2, 4, iter>(stream, option, N, X, incx, A, inca); break;
    case 15: crt_acc_dispatcher<15, 2, 4, iter>(stream, option, N, X, incx, A, inca); break;
    case 16: crt_acc_dispatcher<16, 3, 4, iter>(stream, option, N, X, incx, A, inca); break;
    case 17: crt_acc_dispatcher<17, 3, 5, iter>(stream, option, N, X, incx, A, inca); break;
    case 18: crt_acc_dispatcher<18, 3, 5, iter>(stream, option, N, X, incx, A, inca); break;
    case 19: crt_acc_dispatcher<19, 3, 5, iter>(stream, option, N, X, incx, A, inca); break;
    case 20: crt_acc_dispatcher<20, 3, 5, iter>(stream, option, N, X, incx, A, inca); break;
    case 21: crt_acc_dispatcher<21, 3, 6, iter>(stream, option, N, X, incx, A, inca); break;
    case 22: crt_acc_dispatcher<22, 3, 6, iter>(stream, option, N, X, incx, A, inca); break;
    case 23: crt_acc_dispatcher<23, 3, 6, iter>(stream, option, N, X, incx, A, inca); break;
    default: break;
  }
}

void internal::int8::accumulate_remainder_i32tensor(cudaStream_t stream, int32_t option, int64_t N, int32_t orderX, int32_t iter, const int32_t* X, int64_t incx, uint64_t* A, int64_t inca) {
  switch (iter) {
    case 0: crt_acc_dispatcher<0>(stream, option, N, orderX, X, incx, A, inca); break;
    case 1: crt_acc_dispatcher<1>(stream, option, N, orderX, X, incx, A, inca); break;
    case 2: crt_acc_dispatcher<2>(stream, option, N, orderX, X, incx, A, inca); break;
    default: break;
  }
}
