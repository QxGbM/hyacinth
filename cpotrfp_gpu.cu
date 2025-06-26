
#include <hyacinth.hpp>

#include <thrust/universal_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <cuComplex.h>

struct stride {
  int32_t ld;
  stride(int32_t ld) : ld(ld) {}
  __device__ int32_t operator()(int32_t i) { return i * ld; }
};

void zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  thrust::universal_host_pinned_vector<int32_t> pivots(N);
  thrust::device_vector<double> diag_scale(1);
  double* scale = thrust::raw_pointer_cast(diag_scale.data());

  double* diag = (double*)(&A[N * lda]);
  thrust::device_ptr<double> devA((double*)A), devD(diag);
  auto Adiag_idx = thrust::make_permutation_iterator(devA, thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride(2 * lda + 2)));
  thrust::copy(thrust::cuda::par_nosync.on(stream), Adiag_idx, Adiag_idx + N, devD);

  for (int32_t i = 0; i < N; ++i) {
    imax_double(stream, N - i, &diag[i], &pivots[i], scale);
    cudaStreamSynchronize(stream);
    pivots[i] += 1 + i;

    if ((pivots[i] - 1) != i)
      swap_cols_double_complex(stream, i, pivots[i] - 1, N, A, lda);
  
    minus_adjAx_plusB_scale_double_complex(stream, scale, N - i, i, &A[i * lda], lda, &A[i * lda], &A[i * (lda + 1)], &diag[i]);
    copy_col_to_row_double_complex(stream, i, N, A, lda);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, thrust::raw_pointer_cast(pivots.data()), N * sizeof(int32_t), cudaMemcpyDefault);
}
