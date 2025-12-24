#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

namespace device::int8 {

  const uint16_t div1 = 65280; // 2,3,5,17
  const uint16_t div2 = 65279; // 29,2251
  const uint16_t div3 = 65273; // 13,5021
  const uint16_t div4 = 65269; // prime
  const uint16_t div5 = 65267; // prime
  const uint16_t div6 = 65261; // 7,9323
  const uint16_t div7 = 65257; // prime
  const uint16_t div8 = 65251; // 23,2837
  const uint16_t div9 = 65249; // 71,919
  const uint16_t div10 = 65243; // 53,1231
  const uint16_t div11 = 65239; // prime
  const uint16_t div12 = 65237; // 89,733
  const uint16_t div13 = 65231; // 37,41,43
  const uint16_t div14 = 65227; // 19,3433

  template<uint16_t div>
  __host__ __device__ __forceinline__ uint32_t fast_rem_u32(uint32_t x) {
    constexpr uint32_t m_num = uint32_t(0x100000000llu / uint64_t(div));
#ifdef __CUDA_ARCH__
    uint32_t quo = __umulhi(x, m_num);
#else
    uint32_t quo = (uint64_t(x) * uint64_t(m_num)) >> 32;
#endif
    return x - quo * uint32_t(div);
  }

  template<uint16_t div, uint32_t ORDER>
  __host__ __device__ __forceinline__ uint16_t conv_u32i8_modular(const uint32_t (&code)[ORDER]) {
    static_assert(1 <= ORDER && ORDER <= 3, "Quantized Integer order must be in [1,3]");

    uint32_t lo = fast_rem_u32<div>(code[0]);
    if constexpr(1 < ORDER) {
      constexpr uint32_t rem_s16 = uint32_t(65536) % uint32_t(div);
      constexpr uint32_t rem_s32 = (rem_s16 * rem_s16) % uint32_t(div);
      lo += fast_rem_u32<div>(code[1]) * rem_s32; lo = fast_rem_u32<div>(lo);
      if constexpr(2 < ORDER) {
        constexpr uint32_t rem_s64 = (rem_s32 * rem_s32) % uint32_t(div);
        lo += fast_rem_u32<div>(code[2]) * rem_s64; lo = fast_rem_u32<div>(lo);
      }
    }
    
    uint32_t r255 = fast_rem_u32<uint16_t(255)>(lo);
    return uint16_t((lo & uint32_t(255)) | ((r255 + (r255 >> 7)) << 8));
  }

  template<uint16_t div>
  __host__ __device__ __forceinline__ uint32_t crt_recover_u16(int32_t r256, int32_t r255, uint32_t minv) {
    uint32_t rem = uint32_t(uint8_t(r256) * uint8_t(255)) * uint32_t(255);
    uint32_t u255 = (uint32_t(r255) >> 16) + uint32_t(uint16_t(r255)) + ((-uint32_t(r255 < 0)) & uint32_t(254));
    rem += fast_rem_u32<uint16_t(255)>(u255) << 8;
    rem += (-uint32_t(uint32_t(65280) < rem)) & uint32_t(-65280);
    return fast_rem_u32<div>(rem * minv);
  }


}