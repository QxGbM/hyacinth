
#include <hyacinLRA.hpp>
#include <internal.hpp>
#include <stdexcept>

constexpr uint64_t Gib8 = uint64_t(8) << 30;
template <class T> inline hyacinPrecision_t hyacin_prec();
template <> inline hyacinPrecision_t hyacin_prec<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t hyacin_prec<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t hyacin_prec<__half>() { return HYACIN_F16; };
template <> inline hyacinPrecision_t hyacin_prec<cuDoubleComplex>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<cuComplex>() { return HYACIN_F32_COMPLEX; };
template <> inline hyacinPrecision_t hyacin_prec<__half2>() { return HYACIN_F16_COMPLEX; };

template <class T> inline
int32_t interp_dispatcher(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpvt, T* U, int32_t ldu, T* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; localMv = localMv <= 0 ? N : localMv; batchIter = 0 < handle->BatchK ? (batchIter <= 0 ? handle->BatchK : std::min(batchIter, handle->BatchK)) : M;
  cudaSetDevice(handle->DeviceID); hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype;
  std::div_t divM = std::div(M, batchIter); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batchIter;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vexp, uint64_t(batches + 1) * uint64_t(N) * sizeof(int32_t), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Exponent vector allocation failed");
  hyacinXherkBatchInit(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp);
  hyacinAllReduceVExp(handle, uint64_t(N), vexp);

  uint64_t* C = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Accumulator allocation failed.");
  if (handle->BatchK <= 0) { hyacinXherk(handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int32_t beta = 0; T* Arena = (T*)hyacinXherkBatch(handle, 0, batchIter, N, Atype, vexp, &beta, lPanels, C);
    for (int32_t i = 0; i < M; i += batchIter) {
      int32_t rows = std::min(M - i, batchIter);
      internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, N, &A[i], lda, Arena, handle->BatchK);
      Arena = (T*)hyacinXherkBatch(handle, rows, batchIter, N, Atype, vexp, &beta, lPanels, C);
    }
    hyacinXherkBatchFlush(handle, N, Atype, vexp, beta, lPanels, C);
  }
  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);

  void* G = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Gram matrix allocation failed.");
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  T* X = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&X, uint64_t(N) * uint64_t(K) * sizeof(T), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Basis matrix allocation failed.");
  int32_t rank = hyacinXGinterp(handle, 'A', epi, N, K, Atype, X, N, jpvt, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream);
  if constexpr(std::is_same_v<T, double> || std::is_same_v<T, cuDoubleComplex>) { hyacinSetcuBLASWorkspaceSize(handle, Gib8); }
  if (U) { hyacinXtransform(handle, M, N, rank, Atype, A, lda, U, ldu, 'I', jpvt, 0); }
  hyacinXtransform(handle, localMv, Nv, rank, Atype, V, ldv, V, ldv, 'F', &X[localVoffset], N);
  if constexpr(std::is_same_v<T, double> || std::is_same_v<T, cuDoubleComplex>) { hyacinSetcuBLASWorkspaceSize(handle, uint64_t(0)); }
  cudaFreeAsync(X, handle->cudaStream);
  return rank;
}

template <class T, class R> inline
int32_t svd_dispatcher(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, T* U, int32_t ldu, R* S, T* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; localMv = localMv <= 0 ? N : localMv; batchIter = 0 < handle->BatchK ? (batchIter <= 0 ? handle->BatchK : std::min(batchIter, handle->BatchK)) : M;
  cudaSetDevice(handle->DeviceID); hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype;
  std::div_t divM = std::div(M, batchIter); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batchIter;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vexp, uint64_t(batches + 1) * uint64_t(N) * sizeof(int32_t), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Exponent vector allocation failed");
  hyacinXherkBatchInit(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC, &Gtype, &gElemBytes);
  hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp);
  hyacinAllReduceVExp(handle, uint64_t(N), vexp);

  uint64_t* C = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * sizeof(uint64_t), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Accumulator allocation failed.");
  if (handle->BatchK <= 0) { hyacinXherk(handle, M, N, Atype, A, lda, u, vexp, 0, lPanels, C); } else {
    int32_t beta = 0; T* Arena = (T*)hyacinXherkBatch(handle, 0, batchIter, N, Atype, vexp, &beta, lPanels, C);
    for (int32_t i = 0; i < M; i += batchIter) {
      int32_t rows = std::min(M - i, batchIter);
      internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, N, &A[i], lda, Arena, handle->BatchK);
      Arena = (T*)hyacinXherkBatch(handle, rows, batchIter, N, Atype, vexp, &beta, lPanels, C);
    }
    hyacinXherkBatchFlush(handle, N, Atype, vexp, beta, lPanels, C);
  }
  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);

  void* G = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Gram matrix allocation failed.");
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  T* X = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&X, uint64_t(N) * uint64_t(econV ? K : N) * sizeof(T), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Basis matrix allocation failed.");
  int32_t rank = hyacinXGevd(handle, 'A', epi, econV, N, K, Atype, X, N, S, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream);
  if constexpr(std::is_same_v<R, double>) { hyacinSetcuBLASWorkspaceSize(handle, Gib8); }
  if (U) { hyacinXtransform(handle, M, N, rank, Atype, A, lda, U, ldu, 'F', X, N); }
  hyacinXtransform(handle, localMv, Nv, econV ? rank : N, Atype, V, ldv, V, ldv, 'F', &X[localVoffset], N);
  if constexpr(std::is_same_v<R, double>) { hyacinSetcuBLASWorkspaceSize(handle, uint64_t(0)); }
  cudaFreeAsync(X, handle->cudaStream);
  return rank;
}

template <class T> inline
double check_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, int32_t rank, const T* U, int32_t ldu, const T* V, int32_t ldv, const T* A, int32_t lda) {
  if (M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuComplex> || std::is_same_v<T, __half2>;
  using type = typename std::conditional_t<Complex, cuDoubleComplex, double>;
  double err = 0.; constexpr int32_t block = 2048; constexpr uint64_t bytes = uint64_t(block) * uint64_t(block) * uint64_t(sizeof(type));
  if (rank <= 0) {
    type* matA = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matA, bytes, handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
      for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
        internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, cols, &A[int64_t(i) + int64_t(j) * int64_t(lda)], lda, matA, rows);
        double nrm = 0.; cublasDnrm2(handle->cublasHandle, rows * cols * (Complex + 1), (const double*)matA, 1, &nrm); err += std::pow(nrm, 2);
      }
    }
    cudaFreeAsync(matA, handle->cudaStream);
  } else {
    type* matA = nullptr, *matU = nullptr, *matV = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matA, bytes, handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matU, bytes, handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matV, bytes, handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    cuDoubleComplex minus_one = make_cuDoubleComplex(-1., 0.), one = make_cuDoubleComplex(1., 0.);
    cublasOperation_t opt = CUBLAS_OP_T; cudaDataType Atype = CUDA_R_64F; if constexpr(Complex) { opt = CUBLAS_OP_C; Atype = CUDA_C_64F; }
    for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
      for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
        internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, cols, &A[int64_t(i) + int64_t(j) * int64_t(lda)], lda, matA, rows);
        for (int32_t k = 0; k < rank; k += block) {int32_t reduc = std::min(rank - k, block);
          internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', rows, reduc, &U[int64_t(i) + int64_t(k) * int64_t(ldu)], ldu, matU, block);
          internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', cols, reduc, &V[int64_t(j) + int64_t(k) * int64_t(ldv)], ldv, matV, block);
          cublasGemmEx(handle->cublasHandle, CUBLAS_OP_N, opt, rows, cols, reduc, &minus_one, matU, Atype, block, matV, Atype, block, &one, matA, Atype, rows, CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
        }
        double nrm = 0.; cublasDnrm2(handle->cublasHandle, rows * cols * (Complex + 1), (const double*)matA, 1, &nrm); err += std::pow(nrm, 2);
      }
    }
    cudaFreeAsync(matA, handle->cudaStream); cudaFreeAsync(matU, handle->cudaStream); cudaFreeAsync(matV, handle->cudaStream);
  } return err;
}

namespace hyacinLRA {
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, int32_t* jpvt, double* U, int32_t ldu, double* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, int32_t* jpvt, float* U, int32_t ldu, float* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, int32_t* jpvt, __half* U, int32_t ldu, __half* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, int32_t* jpvt, cuDoubleComplex* U, int32_t ldu, cuDoubleComplex* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, int32_t* jpvt, cuComplex* U, int32_t ldu, cuComplex* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t interp_fit_transform(const hyacinHandle_t* handle, double epi, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, int32_t* jpvt, __half2* U, int32_t ldu, __half2* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return interp_dispatcher(handle, epi, M, N, K, A, lda, jpvt, U, ldu, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }

  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, double* U, int32_t ldu, double* S, double* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, float* U, int32_t ldu, float* S, float* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, __half* U, int32_t ldu, __half* S, __half* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* U, int32_t ldu, double* S, cuDoubleComplex* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, cuComplex* U, int32_t ldu, float* S, cuComplex* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }
  int32_t svd_fit_transform(const hyacinHandle_t* handle, double epi, int32_t econV, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, __half2* U, int32_t ldu, __half* S, __half2* V, int32_t ldv, int64_t globalM, int32_t batchIter, int32_t localMv, int32_t Nv, int32_t localVoffset)
  { return svd_dispatcher(handle, epi, econV, M, N, K, A, lda, U, ldu, S, V, ldv, globalM, batchIter, localMv, Nv, localVoffset); }

  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const double* A, int32_t lda, int32_t rank, const double* U, int32_t ldu, const double* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const float* A, int32_t lda, int32_t rank, const float* U, int32_t ldu, const float* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half* A, int32_t lda, int32_t rank, const __half* U, int32_t ldu, const __half* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuDoubleComplex* A, int32_t lda, int32_t rank, const cuDoubleComplex* U, int32_t ldu, const cuDoubleComplex* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const cuComplex* A, int32_t lda, int32_t rank, const cuComplex* U, int32_t ldu, const cuComplex* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
  double check_lra_answer(const hyacinHandle_t* handle, int32_t M, int32_t N, const __half2* A, int32_t lda, int32_t rank, const __half2* U, int32_t ldu, const __half2* V, int32_t ldv)
  { return check_answer(handle, M, N, rank, U, ldu, V, ldv, A, lda); }
};

