
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

struct scal_add_complex {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex a, cuDoubleComplex b, double s) { return make_cuDoubleComplex(s * (a.x + b.x), s * (a.y + b.y)); }
  __device__ __forceinline__ cuComplex operator()(cuComplex a, cuComplex b, float s) { return make_cuComplex(s * (a.x + b.x), s * (a.y + b.y)); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 a, complex_double2 b, double2 s) { 
    return device::dd::make_complex_double2(device::dd::mul(s, device::dd::add(a.real, b.real)), device::dd::mul(s, device::dd::add(a.imag, b.imag))); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 a, complex_float4 b, float4 s) { 
    return device::qf::make_complex_float4(device::qf::mul(s, device::qf::add(a.real, b.real)), device::qf::mul(s, device::qf::add(a.imag, b.imag))); }
};

struct minus_conj {
  __device__ __forceinline__ cuDoubleComplex operator()(cuDoubleComplex f) { return make_cuDoubleComplex(f.x, -f.y); }
  __device__ __forceinline__ cuComplex operator()(cuComplex f) { return make_cuComplex(f.x, -f.y); }
  __device__ __forceinline__ complex_double2 operator()(complex_double2 f) { return device::dd::conj(device::dd::negate(f)); }
  __device__ __forceinline__ complex_float4 operator()(complex_float4 f) { return device::qf::conj(device::qf::negate(f)); }
};

template <class real_t, class real_ptr, class real_const_ptr, class complex_t, class complex_ptr, class complex_const_ptr,
  int32_t DIM_K, int32_t DIM_L, int32_t BLOCK_WARPS, int32_t GRID_X, int32_t GRID_Y>
__global__ void minus_AHA_plusC_K32_scale_complex(real_const_ptr scale, int32_t N, complex_const_ptr A, complex_ptr C, int32_t ld) {
  constexpr int32_t ITEM_K = DIM_K >> 5;
  constexpr int32_t ITEM_L = (DIM_L >> 5) < 1 ? 1 : (DIM_L >> 5);
  constexpr int32_t TILE_X = GRID_X * BLOCK_WARPS;
  constexpr int32_t TILE_Y = GRID_Y * DIM_L;

  int32_t warp_id = (int32_t(threadIdx.x) >> 5);
  int32_t lane_id = int32_t(threadIdx.x) & 31;

  __shared__ complex_t spaceA[DIM_K * DIM_L];
  __shared__ complex_t spaceD[DIM_L * BLOCK_WARPS];
  complex_t regA[ITEM_K], spaceB[ITEM_K], spaceC[ITEM_L], regD[ITEM_L];
  complex_ptr spaceD_warp = &spaceD[warp_id * DIM_L];

  __shared__ typename cub::WarpLoad<complex_t, ITEM_K>::TempStorage temp_loadK[BLOCK_WARPS];
  __shared__ typename cub::WarpLoad<complex_t, ITEM_L>::TempStorage temp_loadL[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, ITEM_K>::TempStorage temp_storeK[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, ITEM_L>::TempStorage temp_storeL[BLOCK_WARPS];
  __shared__ typename cub::WarpReduce<complex_t>::TempStorage temp_reduce[BLOCK_WARPS];

  int32_t tile_id = warp_id + int32_t(blockIdx.x) * BLOCK_WARPS;
  int32_t tile_y = (tile_id / GRID_X) * DIM_L;
  int32_t tile_x = (tile_id % GRID_X) * BLOCK_WARPS;
  int32_t iter_N = ((N + TILE_X - 1) / TILE_X) * TILE_X;

  minus_conj conj_f;
  add_complex add_f;
  fma_complex fma_f;
  scal_add_complex scal_add_f;

  // TILE_X = warps_on_x * block_warps (all warps maps to the same row)
  for (int32_t col = tile_x; col < iter_N; col += TILE_X) {
    bool col_valid = (col < N); // invalid col warp needs to enter the row_loop to write shmA and sync
    cub::WarpLoad<complex_t, ITEM_K>(temp_loadK[warp_id]).Load(&A[col * ld], spaceB, col_valid ? DIM_K : 0);

    // TILE_Y = tiles_on_y * DIM_L (one warp writes to same column)
    for (int32_t row = tile_y; row < N; row += TILE_Y) {

      int32_t valid_rows = min(N - row, DIM_L);
      #pragma unroll
      for (int32_t i = 0; i < DIM_L; i += BLOCK_WARPS) {
        int32_t items = i < valid_rows ? DIM_K : 0;
        cub::WarpLoad<complex_t, ITEM_K>(temp_loadK[warp_id]).Load(&A[(row + i) * ld], regA, items, complex_t());
        #pragma unroll
        for (int32_t k = 0; k < ITEM_K; ++k)
          regA[k] = conj_f(regA[k]);
        cub::WarpStore<complex_t, ITEM_K>(temp_storeK[warp_id]).Store(&spaceA[i * DIM_K], regA);
      }
      __syncthreads();

      #pragma unroll
      for (int32_t i = 0; i < DIM_L; ++i) {
        complex_t warp_res = complex_t();
        cub::WarpLoad<complex_t, ITEM_K>(temp_loadK[warp_id]).Load(&spaceA[i * DIM_K], regA);

        #pragma unroll
        for (int32_t k = 0; k < ITEM_K; ++k)
          warp_res = fma_f(regA[k], spaceB[k]);
        warp_res = cub::WarpReduce<complex_t>(temp_reduce[warp_id]).Reduce(warp_res, add_f);

        if (lane_id == 0)
          spaceD_warp[i] = warp_res;
      }

      int32_t valid_C = col_valid ? valid_rows : 0;
      complex_ptr C_ij = &C[row + col * ld];
      cub::WarpLoad<complex_t, ITEM_L>(temp_loadL[warp_id]).Load(C_ij, spaceC, valid_C, complex_t());
      cub::WarpLoad<complex_t, ITEM_L>(temp_loadL[warp_id]).Load(spaceD_warp, regD, valid_C, complex_t());
      
      #pragma unroll
      for (int32_t i = 0; i < ITEM_L; ++i)
        spaceC[i] = scal_add_f(scale, regD[i], spaceC[i]);
      cub::WarpStore<complex_t, ITEM_L>(temp_storeL[warp_id]).Store(C_ij, spaceC, valid_C);
    }
  }
}

constexpr int32_t block_warps = 4;
constexpr int32_t grid_size = 1024;
constexpr int32_t grid_warps = grid_size * block_warps;
constexpr int32_t thread_bytes = 32;


