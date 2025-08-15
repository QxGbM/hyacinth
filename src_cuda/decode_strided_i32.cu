
#include <hyacinth.hpp>
#include <internal.hpp>
#include <int_fp_encode.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>
#include <cub/cub.cuh>

struct acc_func {
  template <uint32_t ORDER>
  __device__ __forceinline__ void operator()(double& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f += device::dd::conv_a31_f64(a, expon);
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void operator()(float& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f += device::qf::conv_a31_f32(a, expon);
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void operator()(double2& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f = device::dd::add(f, device::dd::conv_a31_dd(a, expon));
  }

  template <uint32_t ORDER>
  __device__ __forceinline__ void operator()(float4& f, uint32_t const (&a)[ORDER], int32_t expon) {
    f = device::qf::add(f, device::qf::conv_a31_qf(a, expon));
  }
};

template <class real_t, class real_ptr, int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE, int32_t ORDER>
__global__ void decode_kernel(uint64_t M, int32_t N, int32_t expon, real_ptr A, uint64_t strideA, const int32_t* __restrict__ B) {
  constexpr uint64_t elements_block = ITEMS_PER_THREAD * BLOCK_THREADS;
  constexpr uint64_t elements = GRID_BLOCKS * elements_block;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load;
  __shared__ typename cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED>::TempStorage temp_load_real;
  __shared__ typename cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load(temp_load);
  cub::BlockLoad<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_STRIPED> block_load_real(temp_load_real);
  cub::BlockStore<real_t, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_STORE_STRIPED> block_store(temp_store);
  int32_t threadA[ITEMS_PER_THREAD];
  real_t val[ITEMS_PER_THREAD];
  acc_func acc_f;

  for (uint64_t row = (blockIdx.x * elements_block); row < M; row += elements) {
    uint32_t acc[ITEMS_PER_THREAD][ORDER]{};
    uint64_t num_items = min(elements, M - row);

    for (int32_t col = 0; col < N; ++col) {
      int32_t e = BASE * col;
      uint64_t A_col = row + uint64_t(col) * M;
      block_load.Load(&B[A_col], threadA, num_items);
      
      #pragma unroll
      for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
        device::int8::add_shifted(acc[i], threadA[i], e);
    }

    block_load_real.Load(&A[row], val, num_items);

    #pragma unroll
    for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
      acc_f(val[i], acc[i], expon);
    block_store.Store(&A[row], val, num_items);
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

template <class real_t, class real_ptr>
inline void decode_dispatcher(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, real_ptr A, const int32_t* B, int32_t ld) {
  constexpr int32_t acc_bits = 31;
  int32_t order = order_hi - order_lo;
  int32_t acc_order = 1 + ((order * device::Config::exp_base) + (acc_bits - 1)) / acc_bits;
  int32_t expon = device::Config::exp_base * order_lo;
  uint64_t M = uint64_t(N) * uint64_t(ld);
  uint64_t strideA = M * uint64_t(order);

  switch (acc_order) {
    case 1: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 1>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    case 2: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 2>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    case 3: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 3>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    case 4: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 4>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    case 5: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 5>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    case 6: decode_kernel <real_t, real_ptr, grid_blocks, block_threads, items_per_thread, device::Config::exp_base, 6>
      <<< grid_blocks, block_threads, 0, stream >>> (M, order, expon, A, strideA, B); break;
    default: break;
  }
}

void internal::int8::decode_f64_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<double, double* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f32_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<float, float* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f128_dd_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, double2* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<double2, double2* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

void internal::int8::decode_f128_qf_strided_i32(cudaStream_t stream, int32_t order_lo, int32_t order_hi, int32_t N, float4* A, const int32_t* B, int32_t ld) {
  decode_dispatcher<float4, float4* __restrict__>(stream, order_lo, order_hi, N, A, B, ld);
}

