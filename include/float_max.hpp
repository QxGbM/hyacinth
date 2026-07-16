
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>

struct __align__(8) float_idx { float real; int32_t idx; };
struct __align__(16) double_idx { double real; int32_t idx; };
struct __align__(32) double2_idx { double2 real; int32_t idx; };
struct __align__(32) float4_idx { float4 real; int32_t idx; };

namespace device::cmp {
  __host__ __device__ __forceinline__ void cmp_double(double a, double b, bool& less, bool& par) {
    less = a < b; par = a == b;
  }

  __host__ __device__ __forceinline__ void cmp_float(float a, float b, bool& less, bool& par) {
    less = a < b; par = a == b;
  }

  __host__ __device__ __forceinline__ void cmp_double2(double2 a, double2 b, bool& less, bool& par) {
    bool l1 = a.x < b.x, l2 = a.y < b.y, p1 = a.x == b.x;
    less = l1 || (p1 && l2); par = p1 && (a.y == b.y);
  }

  __host__ __device__ __forceinline__ void cmp_float4(float4 a, float4 b, bool& less, bool& par) {
    bool l1 = a.x < b.x, l2 = a.y < b.y, l3 = a.z < b.z, l4 = a.w < b.w;
    bool p1 = a.x == b.x, p2 = p1 && (a.y == b.y), p3 = p2 && (a.z == b.z);
    less = l1 || (p1 && l2) || (p2 && l3) || (p3 && l4); 
    par = p3 && (a.w == b.w);
  }

  struct idx_max {
    __host__ __device__ __forceinline__ double_idx operator()(double_idx a, double_idx b) {
      bool less, par; cmp_double(a.real, b.real, less, par);
      double val = less ? b.real : a.real;
      int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
      int32_t idx_ab = less ? b.idx : a.idx;
      int32_t id = par ? idx_min : idx_ab;
      return double_idx({ val, id });
    }

    __host__ __device__ __forceinline__ float_idx operator()(float_idx a, float_idx b) {
      bool less, par; cmp_float(a.real, b.real, less, par);
      float val = less ? b.real : a.real;
      int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
      int32_t idx_ab = less ? b.idx : a.idx;
      int32_t id = par ? idx_min : idx_ab;
      return float_idx({ val, id });
    }

    __host__ __device__ __forceinline__ double2_idx operator()(double2_idx a, double2_idx b) {
      bool less, par; cmp_double2(a.real, b.real, less, par);
      double2 val = less ? b.real : a.real;
      int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
      int32_t idx_ab = less ? b.idx : a.idx;
      int32_t id = par ? idx_min : idx_ab;
      return double2_idx({ val, id });
    }

    __host__ __device__ __forceinline__ float4_idx operator()(float4_idx a, float4_idx b) {
      bool less, par; cmp_float4(a.real, b.real, less, par);
      float4 val = less ? b.real : a.real;
      int32_t idx_min = a.idx < b.idx ? a.idx : b.idx;
      int32_t idx_ab = less ? b.idx : a.idx;
      int32_t id = par ? idx_min : idx_ab;
      return float4_idx({ val, id });
    }
  };

};
