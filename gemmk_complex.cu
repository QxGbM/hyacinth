
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct fma_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c) {
    return make_cuDoubleComplex(fma(a.x, b.x, fma(-a.y, b.y, c.x)), fma(a.x, b.y, fma(a.y, b.x, c.y))); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, cuComplex c) {
    return make_cuComplex(fmaf(a.x, b.x, fmaf(-a.y, b.y, c.x)), fmaf(a.x, b.y, fmaf(a.y, b.x, c.y))); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, complex_double2 c) { 
    return device::dd::fma(a, b, c); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, complex_float4 c) { 
    return device::qf::fma(a, b, c); }
};

struct minus_conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(-f.x, f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(-f.x, f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::conj(device::dd::negate(f)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::conj(device::qf::negate(f)); }
};

template <class real_t, class real_ptr, class real_const_ptr, class complex_t, 
  class complex_ptr, class complex_const_ptr, int32_t DIM_K, int32_t BLOCK_WARPS, int32_t TILE_ORDER>
__global__ void minus_AHA_plusC_compile_time_k_complex(int32_t N, complex_const_ptr A, complex_ptr C, int32_t ld) {  
  constexpr int32_t TILE_SIZE = 1 << TILE_ORDER;
  constexpr int32_t TILE_SIZE_MASK = (1 << TILE_ORDER) - 1;
  constexpr int32_t ITEM_K = DIM_K >> 5;

  int32_t warp_id = (int32_t(threadIdx.x) >> 5);
  int32_t lane_id = int32_t(threadIdx.x) & 31;
  int32_t tile_id = warp_id + (int32_t(blockIdx.x) * BLOCK_WARPS);
  int32_t tile_y = (tile_id >> TILE_ORDER) << 5;
  int32_t tile_x = tile_id & TILE_SIZE_MASK;

  __shared__ typename cub::WarpLoad<complex_t, ITEM_K>::TempStorage temp_loadK[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, ITEM_K>::TempStorage temp_storeK[BLOCK_WARPS];
  __shared__ typename cub::WarpLoad<complex_t, 1>::TempStorage temp_load_one[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, 1>::TempStorage temp_store_one[BLOCK_WARPS];
  __shared__ complex_t spaceA[DIM_K * 33], spaceB[DIM_K * BLOCK_WARPS];
  complex_t regA[ITEM_K], regB[ITEM_K];
  complex_ptr spaceA_th = &spaceA[lane_id * 33 * ITEM_K], spaceB_warp = &spaceB[warp_id * DIM_K];

  cub::WarpLoad<complex_t, ITEM_K> warp_load_k(temp_loadK[warp_id]);
  cub::WarpStore<complex_t, ITEM_K> warp_store_k(temp_storeK[warp_id]);
  cub::WarpLoad<complex_t, 1> warp_load_one(temp_load_one[warp_id]);
  cub::WarpStore<complex_t, 1> warp_store_one(temp_store_one[warp_id]);
  minus_conj conj_f;
  fma_complex fma_f;

  // TILE_Y = tiles_on_y * 32 (all warps maps to the same row)
  for (int32_t row = tile_y; row < N; row += TILE_SIZE) {
    int32_t valid_rows = min(N - row, 32);

    for (int32_t i = warp_id; i < valid_rows; i += BLOCK_WARPS) {
      warp_load_k.Load(&A[(row + i) * ld], regA);
      __syncwarp();

      #pragma unroll
      for (int32_t k = 0; k < ITEM_K; ++k)
        spaceA_th[(k << 5) + (i + k)] = conj_f(regA[k]);
    }
    __syncthreads(); // thread barrier to guarantee all A read is correct

    // TILE_X = warps_on_x * block_warps (one warp writes to same column)
    for (int32_t col = tile_x; col < N; col += TILE_SIZE) {
      complex_t regC;
      complex_ptr C_ij = &C[row + col * ld];
      warp_load_one.Load(C_ij, *reinterpret_cast<complex_t(*)[1]>(&regC), valid_rows, complex_t());
      warp_load_k.Load(&A[col * ld], regB);
      warp_store_k.Store(spaceB_warp, regB);
      __syncwarp(); // warp barrier to ensure shared-memory broadcast happen

      #pragma unroll
      for (int32_t i = 0; i < DIM_K; ++i) {
        complex_t Ai = spaceB_warp[i], AHi;
        warp_load_one.Load(&spaceA[(i << 5) + i], *reinterpret_cast<complex_t(*)[1]>(&AHi));
        regC = fma_f(AHi, Ai, regC);
      }

      warp_store_one.Store(C_ij, *reinterpret_cast<complex_t(*)[1]>(&regC), valid_rows);
    }
    __syncthreads(); // extra sync to prevent read overwritten A by other warps
  }
}

constexpr int32_t block_warps = 8;
constexpr int32_t tile_order = 10; // 2^10 = 1024
constexpr int32_t grid_y = 1 << (tile_order - 5);
constexpr int32_t grid_x = (1 << tile_order) / block_warps;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::minus_AHA_gemmk64_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, std::complex<double>* C, int32_t ld) {
  constexpr int32_t dim_k = 32 * (thread_bytes / sizeof(std::complex<double>));

  minus_AHA_plusC_compile_time_k_complex <double, double* __restrict__, const double* __restrict__, cuDoubleComplex,
    cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, dim_k, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, (const cuDoubleComplex*)A, (cuDoubleComplex*)C, ld);
}

