
#include <hyacinth.hpp>
#include <internal.hpp>

#include <vector>
#include <numeric>

template<class real_t> struct pinned_space {
  real_t scale;
  int32_t pivot;
};

int32_t dpotrfp_gpu(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<double>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double>));
  int32_t* pivot = &(work->pivot);
  double* scale = &(work->scale);

  double* diag = (double*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(double), A, (lda + 1) * sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    double* Aprev = (0 < i) ? &A[i + (i - 1) * lda] : nullptr;
    internal::Cholesky::imax_double(stream, N - i, Aprev, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!std::isfinite(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_double(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_transAx_plusB_scale_double(stream, *scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t spotrfp_gpu(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<float>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<float>));
  int32_t* pivot = &(work->pivot);
  float* scale = &(work->scale);

  float* diag = (float*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(float), A, (lda + 1) * sizeof(float), sizeof(float), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    float* Aprev = (0 < i) ? &A[i + (i - 1) * lda] : nullptr;
    internal::Cholesky::imax_float(stream, N - i, Aprev, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!std::isfinite(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_float(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_transAx_plusB_scale_float(stream, *scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t double_double_potrfp_gpu(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<double2>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<double2>));
  int32_t* pivot = &(work->pivot);
  double2* scale = &(work->scale);

  double2* diag = (double2*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(double2), A, (lda + 1) * sizeof(double2), sizeof(double2), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    double2* Aprev = (0 < i) ? &A[i + (i - 1) * lda] : nullptr;
    internal::Cholesky::imax_double2(stream, N - i, Aprev, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!device::dd::fisfinite(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_double2(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_transAx_plusB_scale_double2(stream, *scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}

int32_t quad_float_potrfp_gpu(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t* ipiv) {
  std::vector<int32_t> pivots(N);
  std::iota(pivots.begin(), pivots.end(), 1);
  pinned_space<float4>* work;
  cudaMallocHost((void**)&work, sizeof(pinned_space<float4>));
  int32_t* pivot = &(work->pivot);
  float4* scale = &(work->scale);

  float4* diag = (float4*)(&A[N * lda]);
  cudaMemcpy2DAsync(diag, sizeof(float4), A, (lda + 1) * sizeof(float4), sizeof(float4), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    float4* Aprev = (0 < i) ? &A[i + (i - 1) * lda] : nullptr;
    internal::Cholesky::imax_float4(stream, N - i, Aprev, &diag[i], &A[i * (lda + 1)], lda, pivot, scale);
    cudaStreamSynchronize(stream);

    if (!device::qf::fisfinite(*scale)) {
      cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
      cudaFreeHost(work);
      return i + 1;
    }

    if (0 < *pivot) {
      int32_t j = *pivot + i;
      std::iter_swap(&pivots[i], &pivots[j]);
      internal::Cholesky::swap_cols_float4(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_transAx_plusB_scale_float4(stream, *scale, N - i, i, &A[i * lda], lda, &A[i * (lda + 1)]);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(ipiv, pivots.data(), N * sizeof(int32_t), cudaMemcpyDefault);
  cudaFreeHost(work);
  return 0;
}
