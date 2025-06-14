
#include <hyacinth.h>
#include <cuda_runtime_api.h>

#include <vector>
#include <complex>

int32_t cpotrfp_gpu(cublasHandle_t handle, int32_t N, const cuComplex* A, int32_t lda, int32_t* ipiv, cuComplex* X, int32_t ldx, cuComplex* work) {
  int32_t ld = align_up(N, 8);
  if (work == nullptr)
    return ld * (N + 1);

  int32_t rank = N;
  cuComplex* L = &work[N * ld];

  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  cudaMemcpy2DAsync(work, ld * sizeof(std::complex<float>), A, lda * sizeof(std::complex<float>), N * sizeof(std::complex<float>), N, cudaMemcpyDefault, stream);

  std::vector<int32_t> piv(N);
  float s0 = 0.f;
  const std::complex<float> zero(0.f, 0.f), minus_one(-1.f, 0.f);
  for (int32_t i = 0; i < rank; ++i) {
    std::pair<float, int32_t> max = Iamax_float(stream, N, (float*)work, 2 * (ld + 1));
    std::complex<float> scale(1.f / max.first, 0.f);
    piv[i] = max.second;
    
    cublasCcopy(handle, N, &work[piv[i] * ld], 1, L, 1);
    cublasCgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, 1, (const cuComplex*)&scale, &work[piv[i] * ld], ld, (const cuComplex*)&zero, &X[i * ldx], ldx, &X[i * ldx], ldx);
    cublasCgerc(handle, N, N, (const cuComplex*)&minus_one, &X[i * ldx], 1, L, 1, work, ld);

    if (0 == i)
      s0 = scale.real() * 32768.f;
    if (s0 < scale.real())
      rank = i;
  }

  cudaMemcpyAsync(ipiv, &piv[0], N * sizeof(int32_t), cudaMemcpyDefault, stream);
  for (int32_t i = rank - 1; 0 < i; --i) {
    cublasCcopy(handle, i, &X[piv[i]], ldx, L, 1);
    cublasCgeru(handle, N, i, (const cuComplex*)&minus_one, &X[i * ldx], 1, L, 1, X, ldx);
  }

  return rank;
}
