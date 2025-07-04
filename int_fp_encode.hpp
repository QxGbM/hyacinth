#pragma once

#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

#ifdef __CUDACC__
namespace device::int8 {

};
#endif

namespace host::int8 {
  inline double decode_scaled_int4_double(int4 code) {
    uint4 code_lo = make_uint4((uint32_t)code.x & 0x7F, (uint32_t)code.y & 0x3FFF, (uint32_t)code.z & 0x1FFFFF, (uint32_t)code.w & 0xFFFFFFF);
    int32_t hi = (code.x >> 7) + (code.y >> 14) + (code.z >> 21) + (code.w >> 28);
    uint32_t lo = (code_lo.x << 21) + (code_lo.y << 14) + (code_lo.z << 7) + code_lo.w;
    return (double)(((int64_t)hi << 28) + lo);
  }

  inline float2 decode_scaled_int3_float2(int3 code) {
    uint3 code_lo = make_uint3((uint32_t)code.x & 0x1FF, (uint32_t)code.y & 0xFFFF, (uint32_t)code.z & 0x7FFFFF);
    uint32_t lo = (code_lo.x << 14) + (code_lo.y << 7) + code_lo.z;
    int32_t hi = (code.x >> 9) + (code.y >> 16) + (code.z >> 23) + (lo >> 23);
    return make_float2((float)((int64_t)hi << 23), (float)(lo & 0x7FFFFF));
  }

};
