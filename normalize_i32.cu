
#include <internal.hpp>
#include <int_fp_encode.hpp>

#include <cub/cub.cuh>

template <int32_t BASE, int32_t ITEMS_PER_THREAD>
__device__ void signed_normalize(int32_t (&a)[ITEMS_PER_THREAD], int32_t (&c)[ITEMS_PER_THREAD]) {
  constexpr uint32_t iBASE = (uint32_t(1) << BASE) - 1;
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i) {
    int32_t val = a[i] + c[i];
    int32_t sign = val >> 31;
    c[i] = (val >> BASE) - sign;
    a[i] = (val & iBASE) | (sign & ~iBASE);
  }
}

template <int32_t ITEMS_PER_THREAD>
__device__ void signed_sum(int32_t (&a)[ITEMS_PER_THREAD], int32_t const (&c)[ITEMS_PER_THREAD]) {
  #pragma unroll
  for (int32_t i = 0; i < ITEMS_PER_THREAD; ++i)
    a[i] += c[i];
}

template <int32_t GRID_BLOCKS, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, int32_t BASE, int32_t SET_HIGH>
__global__ void normalize_columns_i32(int32_t M, int32_t N, int32_t* __restrict__ A) {
  constexpr int32_t elements_block = ITEMS_PER_THREAD * BLOCK_THREADS;
  constexpr int32_t elements = GRID_BLOCKS * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;
  int32_t M2 = M & (elements_block), M1 = M - M2;

  __shared__ typename cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;

  cub::BlockLoad<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<int32_t, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  int32_t a[ITEMS_PER_THREAD];

  for (int32_t row = block_offset; row < M1; row += elements) {
    int32_t* A_row = &A[row];
    int32_t* A_tl = &A[row + M * N];
    int32_t c[ITEMS_PER_THREAD]{};

    for (int32_t k = 0; k < N; ++k) {
      int32_t* A_k = &A_row[k * M];
      block_load.Load(A_k, a);
      signed_normalize<BASE>(a, c);
      block_store.Store(A_k, a);
    }

    if constexpr (SET_HIGH)
      block_store.Store(A_tl, c);
    else {
      block_load.Load(A_tl, a);
      signed_sum(a, c);
      block_store.Store(A_tl, a);
    }
  }

  if (0 < M2 && blockIdx.x == 0) {
    int32_t* A_row = &A[M1];
    int32_t* A_tl = &A[M1 + M * N];
    int32_t c[ITEMS_PER_THREAD]{};

    for (int32_t k = 0; k < N; ++k) {
      int32_t* A_k = &A_row[k * M];
      block_load.Load(A_k, a, M2, 0);
      signed_normalize<BASE>(a, c);
      block_store.Store(A_k, a, M2);
    }

    if constexpr (SET_HIGH)
      block_store.Store(A_tl, c);
    else {
      block_load.Load(A_tl, a);
      signed_sum(a, c);
      block_store.Store(A_tl, a);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_blocks = 2048;
constexpr int32_t items_per_thread = 4;

void internal::int8::normalize_i32(cudaStream_t stream, int32_t M, int32_t N, int32_t* A) {
  normalize_columns_i32 <grid_blocks, block_threads, items_per_thread, exp_base, 0>
    <<< grid_blocks, block_threads, 0, stream >>> (M, N, A);
}

void internal::int8::normalize_i32_set_high(cudaStream_t stream, int32_t M, int32_t N, int32_t* A) {
  normalize_columns_i32 <grid_blocks, block_threads, items_per_thread, exp_base, 1>
    <<< grid_blocks, block_threads, 0, stream >>> (M, N, A);
}
