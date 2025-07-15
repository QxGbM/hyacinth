
#include <hyacinth.hpp>
#include <internal.hpp>

#include <vector>
#include <numeric>

template<class real_t> struct pinned_space {
  real_t scale;
  int32_t pivot;
};

constexpr int32_t gemmk = 256;

int32_t zpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<double>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double>));
  int32_t* pivot = &(work->pivot);
  double* scale = &(work->scale);

  double* diag = (double*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(double), A, (2 * lda + 2) * sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice, stream);

  int32_t panel = 0;
  for (int32_t i = 0; i < N; ++i) {
    if (0 < i)
      internal::Cholesky::imax_update_double_complex(stream, N - i, &A[i + (i - 1) * lda], &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    else
      internal::Cholesky::imax_double_complex(stream, N - i, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_double_complex(stream, i, j, N, A, lda);
    }

    if (gemmk < i - panel) {
      internal::Cholesky::minus_AHA_gemmk_double_complex(stream, N - i, &A[panel + i * lda], &A[i * (lda + 1)], lda);
      panel += gemmk;
    }
    internal::Cholesky::minus_adjAx_plusB_scale_double_complex(stream, scale, N - i, i - panel, &A[panel + i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t cpotrfp_gpu(cudaStream_t stream, int32_t N, std::complex<float>* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<float>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<float>));
  int32_t* pivot = &(work->pivot);
  float* scale = &(work->scale);

  float* diag = (float*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(float), A, (2 * lda + 2) * sizeof(float), sizeof(float), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    if (0 < i)
      internal::Cholesky::imax_update_float_complex(stream, N - i, &A[i + (i - 1) * lda], &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    else
      internal::Cholesky::imax_float_complex(stream, N - i, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_float_complex(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_adjAx_plusB_scale_float_complex(stream, scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t complex_double_double_potrfp_gpu(cudaStream_t stream, int32_t N, complex_double2* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<double2>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double2>));
  int32_t* pivot = &(work->pivot);
  double2* scale = &(work->scale);

  double2* diag = (double2*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(double2), A, (2 * lda + 2) * sizeof(double2), sizeof(double2), N, cudaMemcpyDeviceToDevice, stream);

  int32_t panel = 0;
  for (int32_t i = 0; i < N; ++i) {
    if (0 < i)
      internal::Cholesky::imax_update_double2_complex(stream, N - i, &A[i + (i - 1) * lda], &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    else
      internal::Cholesky::imax_double2_complex(stream, N - i, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!host::dd::isnormal(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_double2_complex(stream, i, j, N, A, lda);
    }

    if (gemmk < i - panel) {
      internal::Cholesky::minus_AHA_gemmk_double2_complex(stream, N - i, &A[panel + i * lda], &A[i * (lda + 1)], lda);
      panel += gemmk;
    }
    internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(stream, scale, N - i, i - panel, &A[panel + i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t complex_quad_float_potrfp_gpu(cudaStream_t stream, int32_t N, complex_float4* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<float4>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<float4>));
  int32_t* pivot = &(work->pivot);
  float4* scale = &(work->scale);

  float4* diag = (float4*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(float4), A, (2 * lda + 2) * sizeof(float4), sizeof(float4), N, cudaMemcpyDeviceToDevice, stream);

  int32_t panel = 0;
  for (int32_t i = 0; i < N; ++i) {
    if (0 < i)
      internal::Cholesky::imax_update_float4_complex(stream, N - i, &A[i + (i - 1) * lda], &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    else
      internal::Cholesky::imax_float4_complex(stream, N - i, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!host::qf::isnormal(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_float4_complex(stream, i, j, N, A, lda);
    }

    if (gemmk < i - panel) {
      internal::Cholesky::minus_AHA_gemmk_float4_complex(stream, N - i, &A[panel + i * lda], &A[i * (lda + 1)], lda);
      panel += gemmk;
    }
    internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(stream, scale, N - i, i - panel, &A[panel + i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}
