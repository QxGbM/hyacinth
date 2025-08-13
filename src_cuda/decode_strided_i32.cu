
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>

struct acc_i32 {
  __device__ __forceinline__ void operator()(double& f, int32_t i, int32_t expon) {
    f += scalbn(double(i), expon);
  }

  __device__ __forceinline__ void operator()(float& f, int32_t i, int32_t expon) {
    f += scalbnf(float(i), expon);
  }

  __device__ __forceinline__ void operator()(double2& f, int32_t i, int32_t expon) {
    f = device::dd::add_double(f, scalbn(double(i), expon));
  }

  __device__ __forceinline__ void operator()(float4& f, int32_t i, int32_t expon) {
    float c1 = float(i);
    float c2 = float(i - int32_t(c1));
    f = device::qf::add_float2(f, make_float2(scalbnf(c1, expon), scalbnf(c2, expon)));
  }
};

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE>
__global__ void decode_kernel(int32_t k_begin, int32_t k_len, uint64_t N, real_ptr A, const int32_t* __restrict__ B) {
  constexpr uint64_t elements_block = ITEMS_PER_THREAD * BLOCK_THREADS;
  constexpr uint64_t elements = GRID_BLOCKS * elements_block;
  uint64_t block_offset = blockIdx.x * elements_block;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load_real;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load_real(temp_load_real);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  int32_t c[ITEMS_PER_THREAD];
  real_t val[ITEMS_PER_THREAD];
  acc_i32 acc_f;

  for (uint64_t i = block_offset; i < N; i += elements) {
    uint64_t num_items = min(elements, N - i);
    block_load_real.Load(&A[i], val, num_items);

    for (int32_t k = 0; k < k_len; ++k) {
      int32_t expon_k = BASE * (k + k_begin);
      uint64_t A_k = i + uint64_t(k) * N;

      block_load.Load(&B[A_k], c, num_items);
      
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        acc_f(val[i], c[i], expon_k);
    }

    block_store.Store(&A[i], val, num_items);
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

void internal::int8::decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double* A, const int32_t* B, int32_t ld) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(ld);
  decode_kernel <double, double* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, strideA, A, B);
}

void internal::int8::decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float* A, const int32_t* B, int32_t ld) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(ld);
  decode_kernel <float, float* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, strideA, A, B);
}

void internal::int8::decode_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double2* A, const int32_t* B, int32_t ld) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(ld);
  decode_kernel <double2, double2* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, strideA, A, B);
}

void internal::int8::decode_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float4* A, const int32_t* B, int32_t ld) {
  int32_t order = order_hi - order_lo;
  uint64_t strideA = uint64_t(N) * uint64_t(ld);
  decode_kernel <float4, float4* __restrict__, grid_blocks, block_threads, items_per_thread, device::Config::exp_base>
    <<< grid_blocks, block_threads, 0, stream >>> (order_lo, order, strideA, A, B);
}

