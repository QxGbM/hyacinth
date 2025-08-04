
#include <hyacinth.hpp>
#include <internal.hpp>

#include <numeric>

int32_t device::Cholesky::zpotrfp(cudaStream_t stream, int32_t N, std::complex<double>* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  double* scale = (double*)(&ipiv[algnN]), *diag = (double*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(double), A, (2 * lda + 2) * sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    std::complex<double>* Aprev = (0 < i) ? &A[uint64_t(i) + uint64_t(i - 1) * uint64_t(lda)] : nullptr;
    internal::Cholesky::imax_double_complex(stream, N - i, Aprev, &diag[i], &A[uint64_t(i) * uint64_t(lda + 1)], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isfinite(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_double_complex(stream, i, j, N, A, lda);
    }

    internal::Cholesky::minus_adjAx_plusB_scale_double_complex(stream, *scale, N - i, i, &A[uint64_t(i) * uint64_t(lda)], lda, &A[uint64_t(i) * uint64_t(lda + 1)]);
  }
  return 0;
}

int32_t device::Cholesky::cpotrfp(cudaStream_t stream, int32_t N, std::complex<float>* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  float* scale = (float*)(&ipiv[algnN]), *diag = (float*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(float), A, (2 * lda + 2) * sizeof(float), sizeof(float), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    std::complex<float>* Aprev = (0 < i) ? &A[uint64_t(i) + uint64_t(i - 1) * uint64_t(lda)] : nullptr;
    internal::Cholesky::imax_float_complex(stream, N - i, Aprev, &diag[i], &A[uint64_t(i) * uint64_t(lda + 1)], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isfinite(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_float_complex(stream, i, j, N, A, lda);
    }
    internal::Cholesky::minus_adjAx_plusB_scale_float_complex(stream, *scale, N - i, i, &A[uint64_t(i) * uint64_t(lda)], lda, &A[uint64_t(i) * uint64_t(lda + 1)]);
  }
  return 0;
}

int32_t device::Cholesky::complex_double_double_potrfp(cudaStream_t stream, int32_t N, complex_double2* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  double2* scale = (double2*)(&ipiv[algnN]), *diag = (double2*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(double2), A, (2 * lda + 2) * sizeof(double2), sizeof(double2), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    complex_double2* Aprev = (0 < i) ? &A[uint64_t(i) + uint64_t(i - 1) * uint64_t(lda)] : nullptr;
    internal::Cholesky::imax_double2_complex(stream, N - i, Aprev, &diag[i], &A[uint64_t(i) * uint64_t(lda + 1)], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!device::dd::fisfinite(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_double2_complex(stream, i, j, N, A, lda);
    }

    internal::Cholesky::minus_adjAx_plusB_scale_double2_complex(stream, *scale, N - i, i, &A[uint64_t(i) * uint64_t(lda)], lda, &A[uint64_t(i) * uint64_t(lda + 1)]);
  }
  return 0;
}

int32_t device::Cholesky::complex_quad_float_potrfp(cudaStream_t stream, int32_t N, complex_float4* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  float4* scale = (float4*)(&ipiv[algnN]), *diag = (float4*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(float4), A, (2 * lda + 2) * sizeof(float4), sizeof(float4), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    complex_float4* Aprev = (0 < i) ? &A[uint64_t(i) + uint64_t(i - 1) * uint64_t(lda)] : nullptr;
    internal::Cholesky::imax_float4_complex(stream, N - i, Aprev, &diag[i], &A[uint64_t(i) * uint64_t(lda + 1)], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!device::qf::fisfinite(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_float4_complex(stream, i, j, N, A, lda);
    }

    internal::Cholesky::minus_adjAx_plusB_scale_float4_complex(stream, *scale, N - i, i, &A[uint64_t(i) * uint64_t(lda)], lda, &A[uint64_t(i) * uint64_t(lda + 1)]);
  }
  return 0;
}
