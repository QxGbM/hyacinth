
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(8) float_idx { float real; int32_t idx; };
struct __align__(16) double_idx { double real; int32_t idx; };
struct __align__(32) double2_idx { double2 real; int32_t idx; };
struct __align__(32) float4_idx { float4 real; int32_t idx; };

namespace device::cmp {

  __host__ __device__ __forceinline__ double_idx double_max(double_idx a, double_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    double val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
    return double_idx({ val, id });
  }

  __host__ __device__ __forceinline__ float_idx float_max(float_idx a, float_idx b) {
    bool less = a.real < b.real, par = a.real == b.real;
    float val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
    return float_idx({ val, id });
  }

  __host__ __device__ __forceinline__ double2_idx double2_max(double2_idx a, double2_idx b) {
    bool l1 = a.real.x < b.real.x, l2 = a.real.y < b.real.y, p1 = a.real.x == b.real.x;
    bool less = l1 || (p1 && l2), par = p1 && (a.real.y == b.real.y);
    double2 val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
    return double2_idx({ val, id });
  }

  __host__ __device__ __forceinline__ float4_idx float4_max(float4_idx a, float4_idx b) {
    bool l1 = a.real.x < b.real.x, l2 = a.real.y < b.real.y, l3 = a.real.z < b.real.z, l4 = a.real.w < b.real.w;
    bool p1 = a.real.x == b.real.x, p2 = p1 && (a.real.y == b.real.y), p3 = p2 && (a.real.z == b.real.z);
    bool less = l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); 
    bool par = p3 && (a.real.w == b.real.w);
    float4 val = less ? b.real : a.real;
    int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
    int32_t idx_ab = less ? b.idx : a.idx;
    int32_t id = par ? idx_min : idx_ab;
    return float4_idx({ val, id });
  }

};
