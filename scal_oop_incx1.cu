#include <hyacinth.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

struct negate {
  __device__ float operator()(float f) {
    return -f;
  }
};

struct mult {
  float scale;
  mult(float scale) : scale(scale) {}
  __device__ float operator()(float f) {
    return f * scale;
  }
};

void scal_oop_incx1_float(cudaStream_t stream, float scale, int32_t N, const float* x, float* y) {
  thrust::device_ptr<const float> dev_x(x);
  thrust::device_ptr<float> dev_y(y);

  if (scale == -1.f)
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_y, negate());
  else
    thrust::transform(thrust::cuda::par_nosync.on(stream), dev_x, dev_x + N, dev_y, mult(scale));
}

