
#include <hyacinth.hpp>
#include <internal.hpp>
#include <double_double.hpp>
#include <quad_float.hpp>

#include <numeric>

int32_t device::Cholesky::dpotrfp(cudaStream_t stream, int32_t N, double* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  double* scale = (double*)(&ipiv[algnN]), *diag = (double*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(double), A, (lda + 1) * sizeof(double), sizeof(double), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);

    internal::Cholesky::imax_double(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_double(stream, i, j, N, A, lda);
    }
    internal::Cholesky::gemv_scal_f64(stream, *scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);
  }
  return 0;
}

int32_t device::Cholesky::spotrfp(cudaStream_t stream, int32_t N, float* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  float* scale = (float*)(&ipiv[algnN]), *diag = (float*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(float), A, (lda + 1) * sizeof(float), sizeof(float), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);

    internal::Cholesky::imax_float(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(*scale))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_float(stream, i, j, N, A, lda);
    }
    internal::Cholesky::gemv_scal_f32(stream, *scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);
  }
  return 0;
}

int32_t device::Cholesky::double_double_potrfp(cudaStream_t stream, int32_t N, double2* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  double2* scale = (double2*)(&ipiv[algnN]), *diag = (double2*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(double2), A, (lda + 1) * sizeof(double2), sizeof(double2), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);

    internal::Cholesky::imax_double2(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(scale->x))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_double2(stream, i, j, N, A, lda);
    }
    internal::Cholesky::gemv_scal_f128_dd(stream, *scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);
  }
  return 0;
}

int32_t device::Cholesky::quad_float_potrfp(cudaStream_t stream, int32_t N, float4* A, int32_t lda, int32_t* ipiv) {
  int32_t algnN = (N + 3) & (~3), *pivot_i = &ipiv[algnN + 4];
  float4* scale = (float4*)(&ipiv[algnN]), *diag = (float4*)(&A[uint64_t(N) * uint64_t(lda)]);
  std::iota(ipiv, &ipiv[N], 1);
  cudaMemcpy2DAsync(diag, sizeof(float4), A, (lda + 1) * sizeof(float4), sizeof(float4), N, cudaMemcpyDeviceToDevice, stream);

  for (int32_t i = 0; i < N; ++i) {
    uint64_t A_diag = uint64_t(i) * uint64_t(lda + 1);
    uint64_t A_col = uint64_t(i) * uint64_t(lda);

    internal::Cholesky::imax_float4(stream, N - i, &diag[i], &A[A_diag], lda, pivot_i, scale);
    cudaStreamSynchronize(stream);

    if (!std::isnormal(scale->x))
      return i + 1;

    if (0 < *pivot_i) {
      int32_t j = *pivot_i + i;
      std::iter_swap(&ipiv[i], &ipiv[j]);
      internal::Cholesky::swap_cols_float4(stream, i, j, N, A, lda);
    }
    internal::Cholesky::gemv_scal_f128_qf(stream, *scale, N - i, i, &A[A_col], lda, &A[A_diag], &diag[i]);
  }
  return 0;
}
