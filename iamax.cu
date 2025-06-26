
#include <hyacinth.hpp>

#include <cub/cub.cuh>
#include <cuComplex.h>
#include <float4.hpp>

struct less_real {
  __device__ bool operator()(double a, double b) { return a < b; }
  __device__ bool operator()(float a, float b) { return a < b; }
  __device__ bool operator()(float4 a, float4 b) { return device::f4::a_less_than_b(a, b); }
};

struct eq_real {
  __device__ bool operator()(double a, double b) { return a == b; }
  __device__ bool operator()(float a, float b) { return a == b; }
  __device__ bool operator()(float4 a, float4 b) { return device::f4::a_eq_to_b(a, b); }
};

struct get_real {
  __device__ double operator()(cuDoubleComplex f) { return f.x; }
  __device__ float operator()(cuComplex f) { return f.x; }
  __device__ float4 operator()(complex_float4 f) { return f.real; }
};

struct init_real {
  __device__ operator double() { return 0.; }
  __device__ operator float() { return 0.f; }
  __device__ operator float4() { return make_float4(0.f, 0.f, 0.f, 0.f); }
};

template <class real_t> struct real_pair {
  real_t first;
  int32_t second;
};

template <class real_t> struct real_pair_max {
  __device__ real_pair<real_t> operator()(real_pair<real_t> e1, real_pair<real_t> e2) const {
    less_real cmp_less; eq_real cmp_eq;
    real_pair<real_t> val = cmp_less(e1.first, e2.first) ? e2 : e1;
    int32_t id_tie = min(e1.second, e2.second);
    return real_pair<real_t>({ val.first, cmp_eq(e1.first, e2.first) ? id_tie : val.second });
  }
};

template <class real_t, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD>
__global__ void reduce_real(int32_t N, const real_t* A, int32_t* i_out, real_t* val_out) {
  using BlockLoad = cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>;
  using BlockReduce = cub::WarpReduce<real_pair<real_t>, BLOCK_THREADS>;
  constexpr int32_t elements = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ typename BlockLoad::TempStorage temp_load;
  __shared__ typename BlockReduce::TempStorage temp_reduce;

  real_t thread_data[ITEMS_PER_THREAD], init = init_real();
  real_pair<real_t> thread_pair[ITEMS_PER_THREAD];
  real_pair_max<real_t> cmp_max;

  #pragma unroll
  for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
    thread_pair[j] = real_pair<real_t>({ init, -1 });

  for (int32_t i = 0; i < N; i += elements) {
    int32_t num_items = min(elements, N - i);
    BlockLoad(temp_load).Load(&A[i], thread_data, num_items, init);

    #pragma unroll
    for (int32_t j = 0; j < ITEMS_PER_THREAD; ++j)
      thread_pair[j] = cmp_max(thread_pair[j], real_pair<real_t>({ thread_data[j], int32_t(threadIdx.x * ITEMS_PER_THREAD + i + j) }));
  }

  real_pair<real_t> thread_res = cub::ThreadReduce(thread_pair, cmp_max, real_pair<real_t>({ init, -1 }));
  real_pair<real_t> block_res = BlockReduce(temp_reduce).Reduce(thread_res, cmp_max);
  if (threadIdx.x == 0) {
    *i_out = block_res.second;
    *val_out = block_res.first;
  }
}

std::pair<double, int32_t> imax_double(cudaStream_t stream, int32_t N, const double* X) {
  constexpr int32_t block_threads = 8 * 32;
  constexpr int32_t items_per_thread = 4;
  int32_t* d_i;
  double* d_v;

  cudaMallocManaged(reinterpret_cast<void**>(&d_i), sizeof(int32_t), cudaMemAttachGlobal);
  cudaMallocManaged(reinterpret_cast<void**>(&d_v), sizeof(double), cudaMemAttachGlobal);

  reduce_real <double, block_threads, items_per_thread> <<< 1, block_threads, 0, stream >>> (N, X, d_i, d_v);

  cudaStreamSynchronize(stream);
  int32_t id;
  double val;
  cudaMemcpy(&id, d_i, sizeof(int32_t), cudaMemcpyDefault);
  cudaMemcpy(&val, d_v, sizeof(double), cudaMemcpyDefault);
  printf("%d %f\n", id, val);
  cudaFree(d_i);
  cudaFree(d_v);
  return std::make_pair(val, id);
}

