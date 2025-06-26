
#include <hyacinth.hpp>

#include <thrust/host_vector.h>
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
  thrust::host_vector<int32_t> pivots(N);

  double* diag = (double*)(&A[N * lda]);
  thrust::device_ptr<double> devA((double*)A), devD(diag);
  auto Adiag_idx = thrust::make_permutation_iterator(devA, thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride(2 * lda + 2)));
  thrust::copy(thrust::cuda::par_nosync.on(stream), Adiag_idx, Adiag_idx + N, devD);

  for (int32_t i = 0; i < 2; ++i) {
    std::pair<double, int32_t> p = imax_double(stream, N - i, &diag[i]);
    double s = 1. / std::sqrt(p.first);
    pivots[i] = p.second + 1 + i;

    if (p.second != 0)
      swap_cols_double_complex(stream, i, i + p.second, N, A, lda);
  
    minus_adjAx_plusB_scale_double_complex(stream, s, N - i, i, &A[i * lda], lda, &A[i * lda], &A[i * (lda + 1)], &diag[i]);
    copy_col_to_row_double_complex(stream, i, N, A, lda);
  }

  thrust::copy(pivots.begin(), pivots.end(), ipiv);
}
