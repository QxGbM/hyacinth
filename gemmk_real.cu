
#include <internal.hpp>
#include <quad_float.hpp>
#include <double_double.hpp>

#include <cub/cub.cuh>

struct fma_real {
  __device__ __forceinline__ double operator()(double a, double b, double c) { return fma(a, b, c); }
  __device__ __forceinline__ float operator()(float a, float b, float c) { return fmaf(a, b, c); }
  __device__ __forceinline__ double2 operator()(double2 a, double2 b, double2 c) { return device::dd::fma(a, b, c); }
  __device__ __forceinline__ float4 operator()(float4 a, float4 b, float4 c) { return device::qf::fma(a, b, c); }
};

struct minus_only {
  __device__ __forceinline__ double operator()(double f) { return -f; }
  __device__ __forceinline__ float operator()(float f) { return -f; }
  __device__ __forceinline__ double2 operator()(double2 f) { return device::dd::negate(f); }
  __device__ __forceinline__ float4 operator()(float4 f) { return device::qf::negate(f); }
};

template <class real_t, class real_ptr, class real_const_ptr, int32_t BLOCK_WARPS, int32_t TILE_ORDER>
__global__ void minus_ATA_plusC_k32(int32_t N, real_const_ptr A, real_ptr C, int32_t ld) {  
  constexpr int32_t TILE_SIZE = 1 << TILE_ORDER;
  constexpr int32_t TILE_SIZE_MASK = (1 << TILE_ORDER) - 1;

  int32_t warp_id = (int32_t(threadIdx.x) >> 5);
  int32_t lane_id = int32_t(threadIdx.x) & 31;
  int32_t tile_id = warp_id + (int32_t(blockIdx.x) * BLOCK_WARPS);
  int32_t tile_y = (tile_id >> TILE_ORDER) << 5;
  int32_t tile_x = tile_id & TILE_SIZE_MASK;

  __shared__ typename cub::WarpLoad<real_t, 1>::TempStorage temp_load[BLOCK_WARPS];
  __shared__ typename cub::WarpStore<real_t, 1>::TempStorage temp_store[BLOCK_WARPS];
  __shared__ real_t spaceA[32 * 33], spaceB[32 * BLOCK_WARPS];
  real_ptr spaceA_th = &spaceA[lane_id * 33], spaceB_warp = &spaceB[warp_id * 32];

  cub::WarpLoad<real_t, 1> warp_load(temp_load[warp_id]);
  cub::WarpStore<real_t, 1> warp_store(temp_store[warp_id]);
  minus_only negate_f;
  fma_real fma_f;

  // TILE_Y = tiles_on_y * 32 (all warps maps to the same row)
  for (int32_t row = tile_y; row < N; row += TILE_SIZE) {
    int32_t valid_rows = min(N - row, 32);

    for (int32_t i = warp_id; i < valid_rows; i += BLOCK_WARPS) {
      real_t regA = A[(row + i) * ld + lane_id];
      spaceA_th[i] = negate_f(regA);
    }
    __syncthreads(); // thread barrier to guarantee all A read is correct

    // TILE_X = warps_on_x * block_warps (one warp writes to same column)
    for (int32_t col = tile_x; col < N; col += TILE_SIZE) {
      real_t regC;
      int32_t col_loc = col * ld;
      spaceB_warp[lane_id] = A[lane_id + col_loc];
      real_ptr C_ij = &C[row + col_loc];
      warp_load.Load(C_ij, *reinterpret_cast<real_t(*)[1]>(&regC), valid_rows, real_t());
      __syncwarp(); // warp barrier to ensure shared-memory broadcast happen

      #pragma unroll
      for (int32_t i = 0; i < 32; ++i) {
        real_t Ai = spaceB_warp[i];
        real_t AHi = spaceA[(i << 5) + (i + lane_id)];
        regC = fma_f(AHi, Ai, regC);
      }

      warp_store.Store(C_ij, *reinterpret_cast<real_t(*)[1]>(&regC), valid_rows);
    }
    __syncthreads(); // extra sync to prevent read overwritten A by other warps
  }
}

constexpr int32_t block_warps = 16;
constexpr int32_t tile_order = 10; // 2^10 = 1024
constexpr int32_t grid_y = 1 << (tile_order - 5);
constexpr int32_t grid_x = (1 << tile_order) / block_warps;

void internal::Cholesky::minus_ATA_gemmk_double(cudaStream_t stream, int32_t N, const double* A, double* C, int32_t ld) {
  minus_ATA_plusC_k32 <double, double* __restrict__, const double* __restrict__, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, A, C, ld);
}

void internal::Cholesky::minus_ATA_gemmk_float(cudaStream_t stream, int32_t N, const float* A, float* C, int32_t ld) {
  minus_ATA_plusC_k32 <float, float* __restrict__, const float* __restrict__, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, A, C, ld);
}

void internal::Cholesky::minus_ATA_gemmk_double2(cudaStream_t stream, int32_t N, const double2* A, double2* C, int32_t ld) {
  minus_ATA_plusC_k32 <double2, double2* __restrict__, const double2* __restrict__, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, A, C, ld);
}

void internal::Cholesky::minus_ATA_gemmk_float4(cudaStream_t stream, int32_t N, const float4* A, float4* C, int32_t ld) {
  minus_ATA_plusC_k32 <float4, float4* __restrict__, const float4* __restrict__, block_warps, tile_order> 
    <<< grid_y * grid_x, block_warps * 32, 0, stream >>> (N, A, C, ld);
}
