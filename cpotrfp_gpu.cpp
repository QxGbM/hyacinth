
#include <hyacinth.hpp>

#include <vector>

template<class real_t> struct pinned_space {
  real_t scale;
  int32_t pivot;
};

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  pinned_space<double>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double>));
  int32_t* pivot = &(work->pivot);
  double* scale = &(work->scale);

  double* diag = (double*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(double), A, (2 * lda + 2) * sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    if (0 < i)
      imax_update_double_complex(stream, N - i, &A[i + (i - 1) * lda], &diag[i], pivot, scale);
    else
      imax_double(stream, N - i, &diag[i], pivot, scale);
    cudaStreamSynchronize(stream);
    pivots[i] = *pivot + 1 + i;

    if (std::isnan(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), i * sizeof(int32_t), cudaMemcpyDefault);
      return i + 1;
    }

    if (0 < *pivot)
      swap_cols_double_complex(stream, i, *pivot + i, N, A, lda);
    minus_adjAx_plusB_scale_double_complex(stream, scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}
