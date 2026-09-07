
#include <internal.hpp>
#include <int_fp_quantize.hpp>
#include <cub/cub.cuh>
#include <limits>
#include <stdexcept>

constexpr int32_t int_max = std::numeric_limits<int32_t>::max();
struct u64_add {
  __device__ __forceinline__ ulonglong3 operator()(ulonglong3 a, ulonglong3 b) { a.x += b.x; a.y += b.y; a.z += b.z; return a; }
};

template <int32_t beta, int32_t sign>
__device__ __forceinline__ uint64_t* conv_acc(ulonglong3 acc, int32_t M, uint32_t corr, uint64_t* out, int32_t stride) {
  uint64_t a[2]{ uint64_t(acc.x), uint64_t(acc.z) };
  device::int8::add_shifted(a, int64_t(M), corr);
  device::int8::add_shifted(a, int64_t(acc.y), uint32_t(32));
  if constexpr(beta) { device::int8::add_shifted(a, int64_t(out[0]), uint32_t(0)); device::int8::add_shifted(a, int64_t(out[stride]), uint32_t(63)); }
  if constexpr(sign) { out[0] = -a[0]; out[stride] = -a[1]; } else { out[0] = a[0]; out[stride] = a[2]; }
  return &out[stride];
}

template <int32_t beta, int32_t BLOCK_THREADS, class matrix_t>
__global__ void vector_sum_kernel(int32_t M, const matrix_t* __restrict__ A, int64_t lda, uint32_t corr, const int32_t* __restrict__ vexp, uint64_t* __restrict__ vsum) {
  constexpr int32_t Complex = std::is_same_v<matrix_t, cuDoubleComplex> || std::is_same_v<matrix_t, cuComplex> || std::is_same_v<matrix_t, __half2>;
  int32_t expon = vexp[blockIdx.x]; M = (expon == int_max) ? int64_t(0) : M;
  A = &A[int64_t(blockIdx.x) * lda];

  if constexpr(Complex) {
    uint64_t rl[2]{}, im[2]{};
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS) {
      matrix_t A_i = A[i]; int32_t e;
      int64_t q_rl = device::int8::round_i64(A_i.x, expon, e); device::int8::add_shifted(rl, q_rl, uint32_t(e));
      int64_t q_im = device::int8::round_i64(A_i.y, expon, e); device::int8::add_shifted(im, q_im, uint32_t(e));
    }

    __shared__ typename cub::BlockReduce<ulonglong3, BLOCK_THREADS>::TempStorage temp_reduce[2];
    ulonglong3 threadA = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce[0]).Reduce(make_ulonglong3(uint32_t(rl[0]), uint32_t(rl[0] >> 32), rl[1]), u64_add());
    ulonglong3 threadB = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce[1]).Reduce(make_ulonglong3(uint32_t(im[0]), uint32_t(im[0] >> 32), im[1]), u64_add());
    if (threadIdx.x == 0) { conv_acc<beta, 0>(threadB, M, corr, conv_acc<beta, 0>(threadA, M, corr, &vsum[blockIdx.x], int32_t(gridDim.x)), int32_t(gridDim.x)); }
  } else {
    uint64_t rl[2]{};
    for (int32_t i = int32_t(threadIdx.x); i < M; i += BLOCK_THREADS)
    { int32_t e; int64_t q_rl = device::int8::round_i64(A[i], expon, e); device::int8::add_shifted(rl, q_rl, uint32_t(e)); }

    __shared__ typename cub::BlockReduce<ulonglong3, BLOCK_THREADS>::TempStorage temp_reduce;
    ulonglong3 threadA = cub::BlockReduce<ulonglong3, BLOCK_THREADS>(temp_reduce).Reduce(make_ulonglong3(uint32_t(rl[0]), uint32_t(rl[0] >> 32), rl[1]), u64_add());
    if (threadIdx.x == 0) { conv_acc<beta, 1>(threadA, M, corr, &vsum[blockIdx.x], int32_t(gridDim.x)); }
  }
}

constexpr int32_t block_threads = 512;

namespace internal::int8 {

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const double* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const float* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const cuComplex* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

  void vector_sums(cudaStream_t stream, int32_t M, int32_t N, const __half2* A, int32_t lda, uint32_t corr, const int32_t* vexp, uint64_t* vsum)
  { vector_sum_kernel<0, block_threads> <<< N, block_threads, 0, stream >>> (M, A, int64_t(lda), corr, vexp, vsum); }

}
