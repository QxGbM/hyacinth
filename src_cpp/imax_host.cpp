
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <float_max.hpp>

#include <numeric>
#include <execution>

struct real_max {
  inline double_idx operator()(double_idx a, double_idx b) { return device::cmp::double_max(a, b); }
  inline float_idx operator()(float_idx a, float_idx b) { return device::cmp::float_max(a, b); }
  inline double2_idx operator()(double2_idx a, double2_idx b) { return device::cmp::double2_max(a, b); }
  inline float4_idx operator()(float4_idx a, float4_idx b) { return device::cmp::float4_max(a, b); }

  inline void init(double_idx& a) { a = double_idx({ 0., -1 }); }
  inline void init(float_idx& a) { a = float_idx({ 0.f, -1 }); }
  inline void init(double2_idx& a) { a = double2_idx({ make_double2(0., 0.), -1 }); }
  inline void init(float4_idx& a) { a = float4_idx({ make_float4(0.f, 0.f, 0.f, 0.f), -1 }); }
};

inline void real_sqrt(double a, double& sq, double& rsq) { sq = std::sqrt(a); rsq = 1. / sq; }
inline void real_sqrt(float a, float& sq, float& rsq) { sq = std::sqrt(a); rsq = 1.f / sq; }
inline void real_sqrt(double2 a, double2& sq, double2& rsq) { rsq = device::dd::frsqrt(a); sq = device::dd::mul(a, rsq); }
inline void real_sqrt(float4 a, float4& sq, float4& rsq) { rsq = device::qf::frsqrt(a); sq = device::qf::mul(a, rsq); }

template <class idx_t, class real_t>
inline void imax_host(int32_t maxN, int32_t lenX, real_t* X_rl) {
  idx_t* X = reinterpret_cast<idx_t*>(X_rl), init;
  int32_t* piv = reinterpret_cast<int32_t*>(&X_rl[2]);
  
  real_max cmp; cmp.init(init);
  idx_t res = std::reduce(std::execution::unseq, X, &X[lenX], init, cmp);
  res = (0 <= res.idx && res.idx < maxN) ? res : init;
  real_sqrt(res.real, X_rl[0], X_rl[1]);
  *piv = res.idx;
}

void internal::Cholesky::imax_f64_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, double* X) {
  cudaStreamSynchronize(stream);
  imax_host<double_idx>(maxN, lenX, X);
}

void internal::Cholesky::imax_f32_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, float* X) {
  cudaStreamSynchronize(stream);
  imax_host<float_idx>(maxN, lenX, X);
}

void internal::Cholesky::imax_f128_dd_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, double2* X) {
  cudaStreamSynchronize(stream);
  imax_host<double2_idx>(maxN, lenX, X);
}

void internal::Cholesky::imax_f128_qf_host_sync(cudaStream_t stream, int32_t maxN, int32_t lenX, float4* X) {
  cudaStreamSynchronize(stream);
  imax_host<float4_idx>(maxN, lenX, X);
}

