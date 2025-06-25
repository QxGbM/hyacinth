
#include <hyacinth.hpp>
#include <limits>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

template <class real_t> struct real_imax {
  const int32_t incx;
  const real_t* ptr;
  real_imax(int32_t incx, const real_t* ptr) : incx(incx), ptr(ptr) {}

  __device__ thrust::pair<real_t, int32_t> operator()(int32_t i) {
    return thrust::pair<real_t, int32_t>(ptr[i * incx], i);
  }

  __device__ thrust::pair<real_t, int32_t> operator()(thrust::pair<real_t, int32_t> e1, thrust::pair<real_t, int32_t> e2) {
    thrust::pair<real_t, int32_t> val = e1.first < e2.first ? e2 : e1;
    int32_t id_tie = min(e1.second, e2.second);
    return thrust::pair<real_t, int32_t>(val.first, e1.first == e2.first ? id_tie : val.second);
  }
};

std::pair<float, int32_t> real_imax_float(cudaStream_t stream, int32_t N, const float* x, int32_t incx) {
  const real_imax<float> imax_func(incx, x);
  auto diag_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), imax_func);
  thrust::pair<float, int32_t> max = thrust::reduce(thrust::cuda::par_nosync.on(stream), diag_iter, diag_iter + N, thrust::pair<float, int32_t>(-std::numeric_limits<float>::infinity(), 0), imax_func);
  return std::make_pair(max.first, max.second);
}

std::pair<double, int32_t> real_imax_double(cudaStream_t stream, int32_t N, const double* x, int32_t incx) {
  const real_imax<double> imax_func(incx, x);
  auto diag_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), imax_func);
  thrust::pair<double, int32_t> max = thrust::reduce(thrust::cuda::par_nosync.on(stream), diag_iter, diag_iter + N, thrust::pair<double, int32_t>(-std::numeric_limits<double>::infinity(), 0), imax_func);
  return std::make_pair(max.first, max.second);
}

