
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cuComplex.h>
#include <cub/cub.cuh>

struct add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b) { return make_cuDoubleComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b) { return make_cuComplex(a.x + b.x, a.y + b.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b) { return device::dd::add(a, b); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b) { return device::qf::add(a, b); }
};

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
  class complex_ptr, class complex_const_ptr, int32_t DIM_K, int32_t DIM_L, int32_t BLOCK_WARPS, int32_t TILE_ORDER>
__global__ void minus_AHA_plusC_compile_time_k_complex(int32_t N, complex_const_ptr A, complex_ptr C, int32_t ld) {
  constexpr int32_t ITEM_K = DIM_K >> 5;
  constexpr int32_t ITEM_L = (DIM_L >> 5) < 1 ? 1 : (DIM_L >> 5);
  constexpr int32_t WARP_L = ITEM_L * 32;
  constexpr int32_t TILE_SIZE = 1 << TILE_ORDER;
  constexpr int32_t TILE_SIZE_MASK = (1 << TILE_ORDER) - 1;

  int32_t warp_id = (int32_t(threadIdx.x) >> 5);
  int32_t lane_first = int32_t((threadIdx.x & 31) == 0);
  int32_t tile_id = warp_id + int32_t(blockIdx.x) * BLOCK_WARPS;
  int32_t tile_y = (tile_id >> TILE_ORDER) * DIM_L;
  int32_t tile_x = tile_id & TILE_SIZE_MASK;

  __shared__ typename cub::WarpLoad<complex_t, ITEM_K>::TempStorage temp_loadK[BLOCK_WARPS];
  __shared__ typename cub::WarpLoad<complex_t, ITEM_L>::TempStorage temp_loadL[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, ITEM_K>::TempStorage temp_storeK[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, ITEM_L>::TempStorage temp_storeL[BLOCK_WARPS];
  __shared__ typename cub::WarpReduce<complex_t>::TempStorage temp_reduce[BLOCK_WARPS];
  __shared__ complex_t spaceA[DIM_K * DIM_L];
  __shared__ complex_t spaceC[WARP_L * BLOCK_WARPS];
  complex_t regA[ITEM_K], spaceB[ITEM_K], regC[ITEM_L];
  complex_ptr spaceC_warp = &spaceC[warp_id * WARP_L];

  cub::WarpLoad<complex_t, ITEM_K> warp_load_k(temp_loadK[warp_id]);
  cub::WarpLoad<complex_t, ITEM_L> warp_load_l(temp_loadL[warp_id]);
  cub::WarpStore<complex_t, ITEM_K> warp_store_k(temp_storeK[warp_id]);
  cub::WarpStore<complex_t, ITEM_L> warp_store_l(temp_storeL[warp_id]);
  cub::WarpReduce<complex_t> warp_reduce(temp_reduce[warp_id]);
  minus_conj conj_f;
  add_complex add_f;
  fma_complex fma_f;

  // TILE_Y = tiles_on_y * DIM_L (all warps maps to the same row)
  for (int32_t row = tile_y; row < N; row += TILE_SIZE) {
    int32_t valid_rows = min(N - row, DIM_L);

    #pragma unroll
    for (int32_t i = 0; i < DIM_L; i += BLOCK_WARPS) {
      int32_t items = i < valid_rows ? DIM_K : 0;
      warp_load_k.Load(&A[(row + i) * ld], regA, items, complex_t());
      #pragma unroll
      for (int32_t k = 0; k < ITEM_K; ++k)
        regA[k] = conj_f(regA[k]);
      warp_store_k.Store(&spaceA[i * DIM_K], regA);
    }
    __syncthreads();

    // TILE_X = warps_on_x * block_warps (one warp writes to same column)
    for (int32_t col = tile_x; col < N; col += TILE_SIZE) {
      warp_load_k.Load(&A[col * ld], spaceB);

      complex_ptr C_ij = &C[row + col * ld];
      warp_load_l.Load(C_ij, regC, valid_rows, complex_t());
      warp_store_l.Store(spaceC_warp, regC);

      #pragma unroll
      for (int32_t i = 0; i < DIM_L; ++i) {
        complex_t warp_res = lane_first ? spaceC_warp[i] : complex_t();
        warp_load_k.Load(&spaceA[i * DIM_K], regA);

        #pragma unroll
        for (int32_t k = 0; k < ITEM_K; ++k)
          warp_res = fma_f(regA[k], spaceB[k], warp_res);
        warp_res = warp_reduce.Reduce(warp_res, add_f);

        if (lane_first)
          spaceC_warp[i] = warp_res;
      }

      warp_load_l.Load(spaceC_warp, regC);
      warp_store_l.Store(C_ij, regC, valid_rows);
    }
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t write_items = 2;
constexpr int32_t tile_order = 8; // 2^8 = 256
constexpr int32_t grid_y = (1 << tile_order) / write_items;
constexpr int32_t grid_x = (1 << tile_order) / block_warps;
constexpr int32_t thread_bytes = 32;

void internal::Cholesky::minus_AHA_gemmk64_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, std::complex<double>* C, int32_t ld) {
  constexpr int32_t read_items = 32 * (thread_bytes / sizeof(std::complex<double>));

  minus_AHA_plusC_compile_time_k_complex <double, double* __restrict__, const double* __restrict__, cuDoubleComplex,
    cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, read_items, write_items, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, (const cuDoubleComplex*)A, (cuDoubleComplex*)C, ld);
}

