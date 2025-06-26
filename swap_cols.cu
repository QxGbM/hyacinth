
#include <hyacinth.hpp>

#include <thrust/device_ptr.h>
#include <thrust/swap.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <cuComplex.h>
#include <float4.hpp>

struct swap_3row {
  int32_t locs[6];
  __device__ int32_t operator()(int32_t i) { return locs[i]; }
};

template<class T>
inline void swap_cols(cudaStream_t stream, int32_t i, int32_t j, int32_t N, T* A, int32_t lda) {
  swap_3row swap({ i * (lda + 1), j * lda + i, N * lda + i, i * lda + j, j * (lda + 1), N * lda + j });
  thrust::device_ptr<T> devA(A);
  auto row_iter = thrust::make_permutation_iterator(devA, thrust::make_transform_iterator(thrust::make_counting_iterator(0), swap));
  thrust::swap_ranges(thrust::cuda::par_nosync.on(stream), row_iter, row_iter + 3, row_iter + 3);
  thrust::swap_ranges(thrust::cuda::par_nosync.on(stream), devA + (i * lda), devA + (i * lda + N), devA + (j * lda));
}

void swap_cols_double(cudaStream_t stream, int32_t i, int32_t j, int32_t N, double* A, int32_t lda) {
  swap_cols<double>(stream, i, j, N, A, lda);
  copy_col_to_row_double(stream, j, N, A, lda);
}

void swap_cols_double_complex(cudaStream_t stream, int32_t i, int32_t j, int32_t N, std::complex<double>* A, int32_t lda) {
  swap_cols<cuDoubleComplex>(stream, i, j, N, (cuDoubleComplex*)A, lda);
  copy_col_to_row_double_complex(stream, j, N, A, lda);
}
