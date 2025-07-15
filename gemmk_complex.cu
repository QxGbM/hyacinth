
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

struct minus_only {
  __device__ __forceinline__ double operator()(double f) { return -f; }
  __device__ __forceinline__ float operator()(float f) { return -f; }
  __device__ __forceinline__ double2 operator()(double2 f) { return device::dd::negate(f); }
  __device__ __forceinline__ float4 operator()(float4 f) { return device::qf::negate(f); }
};

template <class complex_t, class complex_ptr, class complex_const_ptr, int32_t ITER_K, int32_t BLOCK_WARPS, int32_t GRID_ORDER>
__global__ void minus_AHA_plusC_k32_complex(int32_t N, complex_const_ptr A, complex_ptr C, int32_t ld) {  
  int32_t warp_id = (int32_t(threadIdx.x) >> 5);
  int32_t lane_id = int32_t(threadIdx.x) & 31;
  int32_t blocks_on_y_order = (N <= 32) ? 0 : min(GRID_ORDER, 32 - __clz((N >> 5) - 1));
  int32_t blocks_on_x_order = GRID_ORDER - blocks_on_y_order;
  int32_t warps_on_x = BLOCK_WARPS << blocks_on_x_order;
  int32_t inc_y = 32 << blocks_on_y_order;

  int32_t tile_y = (int32_t(blockIdx.x) >> blocks_on_x_order) << 5;
  int32_t tile_x = (warp_id + (int32_t(blockIdx.x) * BLOCK_WARPS)) & (warps_on_x - 1);

  __shared__ typename cub::WarpLoad<complex_t, 1>::TempStorage temp_load[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<complex_t, 1>::TempStorage temp_store[BLOCK_WARPS];
  __shared__ complex_t spaceA[32 * 33], spaceB[32 * BLOCK_WARPS];
  complex_ptr spaceA_th = &spaceA[lane_id * 33], spaceB_warp = &spaceB[warp_id * 32];

  cub::WarpLoad<complex_t, 1> warp_load(temp_load[warp_id]);
  cub::WarpStore<complex_t, 1> warp_store(temp_store[warp_id]);
  minus_conj conj_f;
  fma_complex fma_f;

  for (int32_t iter_k = 0; iter_k < ITER_K; ++iter_k) {
    // TILE_Y = tiles_on_y * 32 (all warps maps to the same row)
    for (int32_t row = tile_y; row < N; row += inc_y) {
      int32_t valid_rows = min(N - row, 32);

      for (int32_t i = warp_id; i < valid_rows; i += BLOCK_WARPS) {
        complex_t regA = A[(row + i) * ld + lane_id];
        spaceA_th[i] = conj_f(regA);
      }
      __syncthreads(); // thread barrier to guarantee all A read is correct

      // TILE_X = warps_on_x * block_warps (one warp writes to same column)
      for (int32_t col = tile_x; col < N; col += warps_on_x) {
        complex_t regC;
        int32_t col_loc = col * ld;
        spaceB_warp[lane_id] = A[lane_id + col_loc];
        complex_ptr C_ij = &C[row + col_loc];
        warp_load.Load(C_ij, *reinterpret_cast<complex_t(*)[1]>(&regC), valid_rows, complex_t());
        __syncwarp(); // warp barrier to ensure shared-memory broadcast happen

        #pragma unroll
        for (int32_t i = 0; i < 32; ++i) {
          complex_t Ai = spaceB_warp[i];
          complex_t AHi = spaceA[(i << 5) + (i + lane_id)];
          regC = fma_f(AHi, Ai, regC);
        }

        warp_store.Store(C_ij, *reinterpret_cast<complex_t(*)[1]>(&regC), valid_rows);
      }
      __syncthreads(); // extra sync to prevent read overwritten A by other warps
    }

    A = &A[32]; // Move to next series of k32;
  }
}

constexpr int32_t block_warps = 8;
constexpr int32_t grid_order = 14; // 2^10 = 1024
constexpr int32_t grid_blocks = 1 << grid_order;
//constexpr int32_t grid_y = 1 << (grid_order - 5);
//constexpr int32_t grid_x = (1 << grid_order) / block_warps;
constexpr int32_t iter_k = 256 / 32;

void internal::Cholesky::minus_AHA_gemmk_double_complex(cudaStream_t stream, int32_t N, const std::complex<double>* A, std::complex<double>* C, int32_t ld) {
  minus_AHA_plusC_k32_complex <cuDoubleComplex, cuDoubleComplex* __restrict__, const cuDoubleComplex* __restrict__, iter_k, block_warps, grid_order> 
    <<< grid_blocks, block_warps * 32, 0, stream >>> (N, (const cuDoubleComplex*)A, (cuDoubleComplex*)C, ld);
}

void internal::Cholesky::minus_AHA_gemmk_float_complex(cudaStream_t stream, int32_t N, const std::complex<float>* A, std::complex<float>* C, int32_t ld) {
  minus_AHA_plusC_k32_complex <cuComplex, cuComplex* __restrict__, const cuComplex* __restrict__, iter_k, block_warps, grid_order> 
    <<< grid_blocks, block_warps * 32, 0, stream >>> (N, (const cuComplex*)A, (cuComplex*)C, ld);
}

void internal::Cholesky::minus_AHA_gemmk_double2_complex(cudaStream_t stream, int32_t N, const complex_double2* A, complex_double2* C, int32_t ld) {
  minus_AHA_plusC_k32_complex <complex_double2, complex_double2* __restrict__, const complex_double2* __restrict__, iter_k, block_warps, grid_order> 
    <<< grid_blocks, block_warps * 32, 0, stream >>> (N, A, C, ld);
}

void internal::Cholesky::minus_AHA_gemmk_float4_complex(cudaStream_t stream, int32_t N, const complex_float4* A, complex_float4* C, int32_t ld) {
  minus_AHA_plusC_k32_complex <complex_float4, complex_float4* __restrict__, const complex_float4* __restrict__, iter_k, block_warps, grid_order> 
    <<< grid_blocks, block_warps * 32, 0, stream >>> (N, A, C, ld);
}
