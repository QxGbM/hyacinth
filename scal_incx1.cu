
#include <hyacinth.hpp>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <float4.hpp>

struct negate {
  __device__ double operator()(double f) { return -f; }
  __device__ float operator()(float f) { return -f; }
  __device__ __half operator()(__half f) { return -f; }
  __device__ float4 operator()(float4 f) { return device::f4::negate(f); }
};

template <class real_t> struct mult {
  real_t scale;
  mult(real_t scale) : scale(scale) {}
  __device__ real_t operator()(real_t f) { return f * scale; }
};

struct mult_float4 {
  float4 scale;
  mult_float4(float4 scale) : scale(scale) {}
  __device__ float4 operator()(float4 f) { 
    return device::f4::fma(scale, f, make_float4(0.f, 0.f, 0.f, 0.f));
  }
};

void scal_incx1_float(cudaStream_t stream, float scale, int32_t N, float* x) {
  thrust::device_ptr<float> dev_x(x);

  if (scale == -1.f)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, negate());
  else if (scale != 1.f)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, mult<float>(scale));
}

void scal_incx1_double(cudaStream_t stream, double scale, int32_t N, double* x) {
  thrust::device_ptr<double> dev_x(x);

  if (scale == -1.)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, negate());
  else if (scale != 1.)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, mult<double>(scale));
}

void scal_incx1_float4(cudaStream_t stream, float4 scale, int32_t N, float4* x) {
  thrust::device_ptr<float4> dev_x(x);

  if (scale.x == -1.f && scale.y == 0.f && scale.z == 0.f && scale.w == 0.f)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, negate());
  else if (!(scale.x == 1.f && scale.y == 0.f && scale.z == 0.f && scale.w == 0.f))
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_x, mult_float4(scale));
}
