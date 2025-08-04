
#include <internal.hpp>
#include <cub/cub.cuh>

struct convert_fp {
  __device__ __forceinline__ void operator()(double a, double& b) { b = a; }
  __device__ __forceinline__ void operator()(float a, double& b) { b = double(a); }
  __device__ __forceinline__ void operator()(double2 a, double& b) { b = a.x + a.y; }
  __device__ __forceinline__ void operator()(float4 a, double& b) { b = double(a.x) + double(a.y) + double(a.z) + double(a.w); }
  __device__ __forceinline__ void operator()(double a, float& b) { b = float(a); }
  __device__ __forceinline__ void operator()(float a, float& b) { b = a; }
};

template <int32_t GRID_X, int32_t GRID_Y, int32_t BLOCK_THREADS, int32_t ITEMS_PER_THREAD, class typeA, class constPtrA, class typeB, class ptrB>
__global__ void copy_convert_upper(int32_t items, int32_t N, constPtrA A, int32_t lda, ptrB B, int32_t ldb) {
  constexpr int32_t elements_block = BLOCK_THREADS * ITEMS_PER_THREAD;
  constexpr int32_t elements = GRID_Y * elements_block;
  int32_t block_offset = blockIdx.x * elements_block;

  __shared__ typename cub::BlockLoad<typeA, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_load;
  __shared__ typename cub::BlockStore<typeB, BLOCK_THREADS, ITEMS_PER_THREAD>::TempStorage temp_store;
  typeA threadA[ITEMS_PER_THREAD];
  typeB threadB[ITEMS_PER_THREAD];

  cub::BlockLoad<typeA, BLOCK_THREADS, ITEMS_PER_THREAD> block_load(temp_load);
  cub::BlockStore<typeB, BLOCK_THREADS, ITEMS_PER_THREAD> block_store(temp_store);
  convert_fp conv_f;

  for (int32_t col = blockIdx.y; col < N; col += GRID_X) {
    int32_t M = (col + 1) * items;
    int32_t M2 = M & (elements_block - 1);
    int32_t M1 = M - M2;

    constPtrA A_col = &A[uint64_t(col) * uint64_t(lda)];
    ptrB B_col = &B[uint64_t(col) * uint64_t(ldb)];

    for (int32_t k = block_offset; k < M1; k += elements) {
      block_load.Load(&A_col[k], threadA);

      #pragma unroll
      for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
        conv_f(threadA[l], threadB[l]);
      block_store.Store(&B_col[k], threadB);
    }

    if (0 < M2 && blockIdx.x == 0) {
      block_load.Load(&A_col[M1], threadA, M2);

      #pragma unroll
      for (int32_t l = 0; l < ITEMS_PER_THREAD; ++l)
        conv_f(threadA[l], threadB[l]);
      block_store.Store(&B_col[M1], threadB, M2);
    }
  }
}

constexpr int32_t block_threads = 128;
constexpr int32_t grid_x = 512;
constexpr int32_t grid_y = 4;
constexpr int32_t items_per_thread = 4;

void internal::Cholesky::copy_convert_upper_f64_f64(cudaStream_t stream, int32_t items, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, double, const double* __restrict__, double, double* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}

void internal::Cholesky::copy_convert_upper_f32_f64(cudaStream_t stream, int32_t items, int32_t N, const float* A, int32_t lda, double* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, float, const float* __restrict__, double, double* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}

void internal::Cholesky::copy_convert_upper_dd_f64(cudaStream_t stream, int32_t items, int32_t N, const double2* A, int32_t lda, double* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, double2, const double2* __restrict__, double, double* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}

void internal::Cholesky::copy_convert_upper_qf_f64(cudaStream_t stream, int32_t items, int32_t N, const float4* A, int32_t lda, double* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, float4, const float4* __restrict__, double, double* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}

void internal::Cholesky::copy_convert_upper_f64_f32(cudaStream_t stream, int32_t items, int32_t N, const double* A, int32_t lda, float* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, double, const double* __restrict__, float, float* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}

void internal::Cholesky::copy_convert_upper_f32_f32(cudaStream_t stream, int32_t items, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb) {
  copy_convert_upper <grid_x, grid_y, block_threads, items_per_thread, float, const float* __restrict__, float, float* __restrict__>
    <<< dim3(grid_y, grid_x, 1), block_threads, 0, stream >>> (items, N, A, lda, B, ldb);
}
