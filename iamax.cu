#include <hyacinth.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/counting_iterator.h>

struct Ismax {
  const int32_t incx;
  const float* ptr;
  Ismax(int32_t incx, const float* ptr) : incx(incx), ptr(ptr) {}
  __device__ thrust::pair<float, int32_t> operator()(int32_t i) {
    return thrust::pair<float, int32_t>(ptr[i * incx], i);
  }
  __device__ thrust::pair<float, int32_t> operator()(thrust::pair<float, int32_t> e1, thrust::pair<float, int32_t> e2) {
    float e = fmaxf(e1.first, e2.first);
    int32_t i1 = e1.second & ~(__float_as_int(e1.first - e) >> 31);
    int32_t i2 = e2.second & ~(__float_as_int(e2.first - e) >> 31);
    return thrust::pair<float, int32_t>(e, i1 | i2);
  }
};

std::pair<float, int32_t> Iamax_float(cudaStream_t stream, int32_t N, const float* x, int32_t incx) {
  const Ismax imax_func(incx, x);
  auto diag_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0), imax_func);
  thrust::pair<float, int32_t> max = thrust::reduce(thrust::cuda::par_nosync.on(stream), diag_iter, diag_iter + N, thrust::pair<float, int32_t>(0.f, 0), imax_func);
  return std::make_pair(max.first, max.second);
}

