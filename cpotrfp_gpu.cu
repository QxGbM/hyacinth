
#include <hyacinth.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <complex>

int32_t cpotrfp_gpu(cublasHandle_t handle, int32_t N, const cuComplex* A, int32_t lda, int32_t* ipiv, cuComplex* X, int32_t ldx, cuComplex* work) {
  int32_t ld = lda; //align_up(N, 8);
  if (work == nullptr)
    return ld * (N + 1);

  int32_t rank = N;
  cuComplex* L = &work[N * ld];

  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  cudaMemcpy2DAsync(work, ld * sizeof(std::complex<float>), A, lda * sizeof(std::complex<float>), N * sizeof(std::complex<float>), N, cudaMemcpyDefault, stream);

  std::vector<int32_t> piv(N);
  std::vector<float> diags(N);
  const std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f);
  float s0 = 0.f;
  for (int32_t i = 0; i < rank; ++i) {
    std::pair<float, int32_t> max = real_imax_float(stream, N, (float*)work, 2 * (ld + 1));
    float scale = 1.f / std::sqrt(max.first);
    diags[i] = scale;
    piv[i] = max.second;
    
    scal_incx1_float(stream, -scale, N * 2, (float*)(&work[max.second * ld]));
    scal_incx1_float(stream, scale, N * 2, (float*)(&work[max.second * ld]));
    cublasCgerc(handle, N, N, (const cuComplex*)&one, &X[i * ldx], 1, L, 1, work, ld);
    cudaMemsetAsync(&work[max.second * ld], 0, N * 2 * sizeof(float), stream);
    cublasCcopy(handle, N, &work[max.second * ld], 1, &work[max.second], ld);

    if (0 == i)
      s0 = scale * 2048.f;
    if (s0 < scale)
      rank = i;
  }

  cudaMemcpyAsync(ipiv, &piv[0], N * sizeof(int32_t), cudaMemcpyDefault, stream);
  for (int32_t i = rank - 1; 0 <= i; --i) {
    cublasCcopy(handle, i, &X[piv[i]], ldx, L, 1);
    float scale = diags[i];
    printf("%d %d %e\n", i, piv[i], scale);
    scal_incx1_float(stream, scale, N * 2, (float*)(&X[i * ldx]));
    cublasCgeru(handle, N, i, (const cuComplex*)&minus_one, &X[i * ldx], 1, L, 1, X, ldx);
  }

  return rank;
}
