
#include <hyacinLRA.hpp>
#include <internal.hpp>

template <class T> inline hyacinPrecision_t hyacin_prec();
template <> inline hyacinPrecision_t hyacin_prec<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t hyacin_prec<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t hyacin_prec<__half>() { return HYACIN_F16; };
template <> inline hyacinPrecision_t hyacin_prec<cuDoubleComplex>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<cuComplex>() { return HYACIN_F32_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<__half2>() { return HYACIN_F16_COMPLEX; };

template <class T> inline
int32_t interp_dispatcher(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpvt, T* U, int32_t ldu, T* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; localMv = localMv <= 0 ? N : localMv; batchIter = batchIter <= 0 ? M : std::min(batchIter, handle->BatchK);
  cudaSetDevice(handle->DeviceID); hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC, Bbytes;
  cudaMallocFromPoolAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle->mempool, handle->cudaStream);
  hyacinXGautoType(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScale(handle, M, N, Atype, A, lda, u, 0, vexp);
  hyacinAllReduceVExp(handle, uint64_t(N), vexp);

  uint64_t* C = nullptr; cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle->mempool, handle->cudaStream);
  if (handle->BatchK <= 0) { hyacinXherk(handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int8_t* Bdata = nullptr; hyacinXherkBatchInit(handle, epi, N, Atype, &Bbytes);
    cudaMallocFromPoolAsync((void**)&Bdata, Bbytes, handle->mempool, handle->cudaStream);

    int32_t beta = 0;
    T* Arena = (T*)hyacinXherkBatch(handle, 0, batchIter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    for (int32_t i = 0; i < M; i += batchIter) {
      int32_t rows = std::min(M - i, batchIter);
      internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, N, nullptr, &A[i], lda, Arena, handle->BatchK);
      Arena = (T*)hyacinXherkBatch(handle, rows, batchIter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    }
    hyacinXherkBatchFlush(handle, N, Atype, vexp, beta, lPanels, C, Bdata);
    cudaFreeAsync(Bdata, handle->cudaStream);
  }
  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);

  void* G = nullptr; cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream);
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  T* X = nullptr; cudaMallocFromPoolAsync((void**)&X, uint64_t(N) * uint64_t(K) * sizeof(T), handle->mempool, handle->cudaStream);
  int32_t rank = hyacinXGinterp(handle, 'A', epi, N, K, Atype, X, N, jpvt, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream);
  hyacinXtransform(handle, M, N, rank, Atype, A, lda, U, ldu, 'I', jpvt, 0);
  hyacinXtransform(handle, localMv, Nv, rank, Atype, V, ldv, V, ldv, 'F', &X[localVoffset], N);
  cudaFreeAsync(X, handle->cudaStream);
  return rank;
}

template <class T, class R> inline
int32_t svd_dispatcher(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, T* U, int32_t ldu, R* S, T* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; localMv = localMv <= 0 ? N : localMv; batchIter = batchIter <= 0 ? M : std::min(batchIter, handle->BatchK);
  cudaSetDevice(handle->DeviceID); hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC, Bbytes;
  cudaMallocFromPoolAsync((void**)&vexp, uint64_t(N) * sizeof(int32_t), handle->mempool, handle->cudaStream);
  hyacinXGautoType(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScale(handle, M, N, Atype, A, lda, u, 0, vexp);
  hyacinAllReduceVExp(handle, uint64_t(N), vexp);

  uint64_t* C = nullptr; cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle->mempool, handle->cudaStream);
  if (handle->BatchK <= 0) { hyacinXherk(handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int8_t* Bdata = nullptr; hyacinXherkBatchInit(handle, epi, N, Atype, &Bbytes);
    cudaMallocFromPoolAsync((void**)&Bdata, Bbytes, handle->mempool, handle->cudaStream);

    int32_t beta = 0;
    T* Arena = (T*)hyacinXherkBatch(handle, 0, batchIter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    for (int32_t i = 0; i < M; i += batchIter) {
      int32_t rows = std::min(M - i, batchIter);
      internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, N, nullptr, &A[i], lda, Arena, handle->BatchK);
      Arena = (T*)hyacinXherkBatch(handle, rows, batchIter, N, Atype, vexp, &beta, lPanels, C, Bdata);
    }
    hyacinXherkBatchFlush(handle, N, Atype, vexp, beta, lPanels, C, Bdata);
    cudaFreeAsync(Bdata, handle->cudaStream);
  }
  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);

  void* G = nullptr; cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream);
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  T* X = nullptr; cudaMallocFromPoolAsync((void**)&X, uint64_t(N) * uint64_t(K) * sizeof(T), handle->mempool, handle->cudaStream);
  int32_t rank = hyacinXGevd(handle, 'A', epi, N, K, Atype, X, N, S, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream);
  hyacinXtransform(handle, M, N, rank, Atype, A, lda, U, ldu, 'F', X, N);
  hyacinXtransform(handle, localMv, Nv, rank, Atype, V, ldv, V, ldv, 'F', &X[localVoffset], N);
  cudaFreeAsync(X, handle->cudaStream);
  return rank;
}

namespace hyacinLRA {
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, int32_t* jpvt, double* U, int32_t ldu, double* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, int32_t* jpvt, float* U, int32_t ldu, float* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, int32_t* jpvt, __half* U, int32_t ldu, __half* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, int32_t* jpvt, cuDoubleComplex* U, int32_t ldu, cuDoubleComplex* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, int32_t* jpvt, cuComplex* U, int32_t ldu, cuComplex* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, int32_t* jpvt, __half2* U, int32_t ldu, __half2* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, double* U, int32_t ldu, double* S, double* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, float* U, int32_t ldu, float* S, float* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, __half* U, int32_t ldu, __half* S, __half* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* U, int32_t ldu, double* S, cuDoubleComplex* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, cuComplex* U, int32_t ldu, float* S, cuComplex* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, __half2* U, int32_t ldu, __half* S, __half2* V, int32_t ldv, int32_t batchIter, int64_t globalM, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, M, N, K, A, lda, U, ldu, S, V, ldv, batchIter, globalM, localMv, Nv, localVoffset); }
};

