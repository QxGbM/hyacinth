
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device {

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[1], int64_t i) {
    if constexpr(expon == uint32_t(0)) { a[0] += uint64_t(i); } else
    if constexpr(expon <= uint32_t(63)) { a[0] += uint64_t(i) << expon; }
  } 

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[2], int64_t i) {
    constexpr uint64_t i63 = 0x7fffffffffffffffllu;
    if constexpr(expon == uint32_t(0)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[0] += uint64_t(i) & i63; a[1] += sign + (a[0] >> 63); a[0] &= i63; 
    } else if constexpr(expon == uint32_t(63)) {
      a[1] += uint64_t(i); 
    } else if constexpr(expon < uint32_t(63)) {
      constexpr uint32_t left_sft = expon, sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(63) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[0] += q0 & i63; a[1] += q1 + (a[0] >> 63); a[0] &= i63;
    } else if constexpr(expon <= uint32_t(126)) {
      constexpr uint32_t left_sft = expon - uint32_t(63);
      a[1] += uint64_t(i) << left_sft;
    }
  }

  template <uint32_t expon> __host__ __device__ __forceinline__ void add_shifted(uint64_t (&a)[3], int64_t i) {
    constexpr uint64_t i63 = 0x7fffffffffffffffllu;
    if constexpr(expon == uint32_t(0)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[0] += uint64_t(i) & i63;
      a[1] += (sign & i63) + (a[0] >> 63); a[0] &= i63; a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon == uint32_t(63)) {
      uint64_t sign = -(uint64_t(i) >> 63); a[1] += uint64_t(i) & i63;
      a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon == uint32_t(126)) {
      a[2] += uint64_t(i);
    } else if constexpr(expon < uint32_t(63)) {
      constexpr uint32_t left_sft = expon, sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(63) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[0] += q0 & i63; a[1] += (q1 & i63) + (a[0] >> 63); a[0] &= i63; a[2] += sign + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon < uint32_t(126)) {
      constexpr uint32_t left_sft = expon - uint32_t(63), sign_sft = uint32_t(1) + left_sft, right_sft = uint32_t(126) - expon;
      uint64_t q0 = uint64_t(i) << left_sft, sign = -(uint64_t(i) >> 63), q1 = (sign << sign_sft) | (uint64_t(i) >> right_sft);
      a[1] += q0 & i63; a[2] += q1 + (a[1] >> 63); a[1] &= i63;
    } else if constexpr(expon <= uint32_t(189)) {
      constexpr uint32_t left_sft = expon - uint32_t(126);
      a[2] += uint64_t(i) << left_sft;
    }
  }

  template<uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t barrett_reduc(uint32_t x) {
#ifdef __CUDA_ARCH__
    if constexpr(DIV) {
      constexpr uint32_t d = uint32_t((0x100000000llu / uint64_t(DIV))), minus_div = -DIV;
      x += __umulhi(x, d) * minus_div; return __viaddmin_u32(minus_div, x, x);
    } else return uint32_t(0);
#else
    if constexpr(DIV) return x % DIV; else return uint32_t(0);
#endif
  }

  template<uint32_t MUL, uint32_t DIV>
  __host__ __device__ __forceinline__ uint32_t mulx_reduc(uint32_t x) {
    if constexpr(MUL && DIV) {
      static_assert(DIV <= 4096u, "Modular needs to be smaller than 4096");
      constexpr uint32_t mul = MUL % DIV, mul_r16 = (mul << 16) % DIV;
      return (mul_r16 * (x >> 16)) + (mul * (x & uint32_t(0xffff)));
    } else return uint32_t(0);
  }

};
