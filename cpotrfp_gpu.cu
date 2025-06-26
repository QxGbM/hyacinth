
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
  __device__ __forceinline__ int32_t operator()(int32_t i) { return i * ld; }
};

template<class real_t> struct pinned_space {
  real_t scale;
  int32_t pivot;
};

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  thrust::host_vector<int32_t> pivots(N);
  pinned_space<double>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double>));
  int32_t* pivot = &(work->pivot);
  double* scale = &(work->scale);

  double* diag = (double*)(&A[N * lda]);
  thrust::device_ptr<double> devA((double*)A), devD(diag);
  auto Adiag_idx = thrust::make_permutation_iterator(devA, thrust::make_transform_iterator(thrust::make_counting_iterator(0), stride(2 * lda + 2)));
  thrust::copy(thrust::cuda::par_nosync.on(stream), Adiag_idx, Adiag_idx + N, devD);

  for (int32_t i = 0; i < N; ++i) {
    imax_double(stream, N - i, &diag[i], pivot, scale);
    cudaStreamSynchronize(stream);
    pivots[i] = *pivot + 1 + i;

    if (std::isnan(*scale)) {
      cudaMemcpy(ipiv, thrust::raw_pointer_cast(pivots.data()), i * sizeof(int32_t), cudaMemcpyDefault);
      return i + 1;
    }
    if ((pivots[i] - 1) != i)
      swap_cols_double_complex(stream, i, pivots[i] - 1, N, A, lda);
  
    minus_adjAx_plusB_scale_double_complex(stream, scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)], &diag[i]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, thrust::raw_pointer_cast(pivots.data()), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}
