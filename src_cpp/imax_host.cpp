
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <float_max.hpp>

inline void real_sqrt(double a, double& sq, double& rsq) { sq = std::sqrt(a); rsq = 1. / sq; }
inline void real_sqrt(float a, float& sq, float& rsq) { sq = std::sqrt(a); rsq = 1.f / sq; }
inline void real_sqrt(double2 a, double2& sq, double2& rsq) { device::dd::frsqrt(a, sq, rsq); }
inline void real_sqrt(float4 a, float4& sq, float4& rsq) { device::qf::frsqrt(a, sq, rsq); }

template <class idx_t>
inline void imax_host(idx_t* X) {
  idx_t res = (1 <= X[0].idx) ? X[0] : idx_t();
  real_sqrt(res.real, X[0].real, X[1].real);
}

void internal::Cholesky::imax_f64_host_sync(cudaStream_t stream, double_idx* X) {
  cudaStreamSynchronize(stream);
  imax_host(X);
}

void internal::Cholesky::imax_f32_host_sync(cudaStream_t stream, float_idx* X) {
  cudaStreamSynchronize(stream);
  imax_host(X);
}

void internal::Cholesky::imax_f128_dd_host_sync(cudaStream_t stream, double2_idx* X) {
  cudaStreamSynchronize(stream);
  imax_host(X);
}

void internal::Cholesky::imax_f128_qf_host_sync(cudaStream_t stream, float4_idx* X) {
  cudaStreamSynchronize(stream);
  imax_host(X);
}

