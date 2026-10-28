
#include <hyacin.h>
#include <internal.hpp>
#include <vector>
#include <numeric>
#include <stdexcept>

template <class T> static inline hyacinPrecision_t hyacin_prec() {
  if constexpr(std::is_same_v<T, double>) { return HYACIN_F64; } else
  if constexpr(std::is_same_v<T, float>) { return HYACIN_F32; } else
  if constexpr(std::is_same_v<T, __half>) { return HYACIN_F16; } else
  if constexpr(std::is_same_v<T, cuDoubleComplex>) { return HYACIN_F64_COMPLEX; } else
  if constexpr(std::is_same_v<T, cuComplex>) { return HYACIN_F32_COMPLEX; } else
  if constexpr(std::is_same_v<T, __half2>) { return HYACIN_F16_COMPLEX; } else
  { return hyacinPrecision_t(); }
}

template <class T> static inline
int32_t interp_dispatcher(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpvt, T* V, int32_t ldv, int64_t globalM) {
  K = std::min(N, K); if (K <= 0) { return 0; }
  int32_t batchIter = handle->GramSegmentK <= 0 ? M : std::min(handle->GramSegmentK, M);
  hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype; opA = opA == 't' ? 'T' : (opA == 'c' ? 'C' : opA);
  std::div_t divM = std::div(M, batchIter); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batchIter;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vexp, uint64_t(batches + 1) * uint64_t(N) * uint64_t(sizeof(int32_t)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Exponent vector allocation failed");
  hyacinXherkBatchInit(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC);
  uint64_t* C = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * uint64_t(sizeof(uint64_t)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Accumulator allocation failed.");

  if (opA == 'T' || opA == 'C') {
    T* AH = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&AH, uint64_t(M) * uint64_t(N) * uint64_t(sizeof(T)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("A-Transpose allocation failed");
    internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, opA, M, N, A, lda, AH, M);
    hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, AH, M, u, vexp);
    hyacinAllReduceVExp(handle, uint64_t(N), vexp);
    hyacinXherkBatch(handle, batches, batchIter, lastM, N, Atype, AH, M, u, vexp, 0, lPanels, C);
    cudaFreeAsync(AH, handle->cudaStream);
  } else {
    hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp);
    hyacinAllReduceVExp(handle, uint64_t(N), vexp);
    hyacinXherkBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp, 0, lPanels, C);
  }

  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);
  hyacinXdequantizeInit(handle, N, lPanels, C, vexp, Atype, &Gtype, &gElemBytes);
  void* G = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Gram matrix allocation failed.");
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  int32_t rank = hyacinXGinterp(handle, 'A', epi, N, K, Atype, V, ldv, jpvt, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream); return rank;
}

template <class T, class R> static inline
int32_t svd_dispatcher(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, R* S, T* V, int32_t ldv, int64_t globalM) {
  K = std::min(N, K); if (K <= 0) { return 0; }
  int32_t batchIter = handle->GramSegmentK <= 0 ? M : std::min(handle->GramSegmentK, M);
  hyacinPrecision_t Atype = hyacin_prec<T>(), Gtype; opA = opA == 't' ? 'T' : (opA == 'c' ? 'C' : opA);
  std::div_t divM = std::div(M, batchIter); int32_t batches = divM.quot + int32_t(0 < divM.rem), lastM = 0 < divM.rem ? divM.rem : batchIter;
  int32_t* vexp = nullptr, u, cPanels, lPanels, gElemBytes; uint64_t strideC;
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&vexp, uint64_t(batches + 1) * uint64_t(N) * uint64_t(sizeof(int32_t)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Exponent vector allocation failed");
  hyacinXherkBatchInit(handle, epi, globalM, N, Atype, &u, &cPanels, &lPanels, &strideC);
  uint64_t* C = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&C, uint64_t(cPanels) * uint64_t(lPanels) * strideC * uint64_t(sizeof(uint64_t)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Accumulator allocation failed.");
  
  if (opA == 'T' || opA == 'C') {
    T* AH = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&AH, uint64_t(M) * uint64_t(N) * uint64_t(sizeof(T)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("A-Transpose allocation failed");
    internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, opA, M, N, A, lda, AH, M);
    hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, AH, M, u, vexp);
    hyacinAllReduceVExp(handle, uint64_t(N), vexp);
    hyacinXherkBatch(handle, batches, batchIter, lastM, N, Atype, AH, M, u, vexp, 0, lPanels, C);
    cudaFreeAsync(AH, handle->cudaStream);
  } else {
    hyacinXquantizeScaleBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp);
    hyacinAllReduceVExp(handle, uint64_t(N), vexp);
    hyacinXherkBatch(handle, batches, batchIter, lastM, N, Atype, A, lda, u, vexp, 0, lPanels, C);
  }

  hyacinAllReduce1Drow(handle, cPanels, lPanels, strideC, C);
  hyacinXdequantizeInit(handle, N, lPanels, C, vexp, Atype, &Gtype, &gElemBytes);
  void* G = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&G, uint64_t(N) * uint64_t(N) * uint64_t(gElemBytes), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Gram matrix allocation failed.");
  hyacinXdequantize(handle, N, lPanels, C, vexp, Gtype, G, N);
  cudaFreeAsync(vexp, handle->cudaStream); cudaFreeAsync(C, handle->cudaStream);

  int32_t rank = hyacinXGevd(handle, 'A', epi, N, K, Atype, V, ldv, S, Gtype, G, N);
  cudaFreeAsync(G, handle->cudaStream); return rank;
}

static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc){
  double one = 1., zero = 0.;
  cublasGemmEx(handle->cublasHandle, opA == 'C' || opA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_64F, lda, B, CUDA_R_64F, ldb, &zero, C, CUDA_R_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}
static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc) {
  float one = 1.f, zero = 0.f;
  cublasGemmEx(handle->cublasHandle, opA == 'C' || opA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_32F, lda, B, CUDA_R_32F, ldb, &zero, C, CUDA_R_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
}
static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const __half* A, int32_t lda, const __half* B, int32_t ldb, __half* C, int32_t ldc) {
  __half one = CUDART_ONE_FP16, zero = CUDART_ZERO_FP16;
  cublasGemmEx(handle->cublasHandle, opA == 'C' || opA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, CUDA_R_16F, lda, B, CUDA_R_16F, ldb, &zero, C, CUDA_R_16F, ldc, CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT);
}
static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc) {
  cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
  cublasGemmEx(handle->cublasHandle, opA == 'C' ? CUBLAS_OP_C : (opA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N), CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_64F, lda, B, CUDA_C_64F, ldb, &zero, C, CUDA_C_64F, ldc, CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT);
}
static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc) {
  cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
  cublasGemmEx(handle->cublasHandle, opA == 'C' ? CUBLAS_OP_C : (opA == 'T' ? CUBLAS_OP_T : CUBLAS_OP_N), CUBLAS_OP_N, M, N, K, &one, A, CUDA_C_32F, lda, B, CUDA_C_32F, ldb, &zero, C, CUDA_C_32F, ldc, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUBLAS_GEMM_DEFAULT);
}
static inline void an_gemm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t K, const __half2* A, int32_t lda, const __half2* B, int32_t ldb, __half2* C, int32_t ldc) {
  internal::matmul_cf16(handle->cudaStream, handle->cublasHandle, handle->mempool, opA, M, N, K, A, lda, B, ldb, C, ldc);
}

template <class T, class R> static inline
int32_t svd_segN_dispatcher(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, R* S, T* V, int32_t ldv, int64_t globalM) {
  K = std::min(N, K); if (K <= 0) { return 0; }
  int32_t segments = 512 <= N ? int32_t(std::floor(std::sqrt(double(N) / double(K)))) : 1;
  if (segments <= 1) { return svd_dispatcher(handle, epi, opA, M, N, K, A, lda, S, V, ldv, globalM); }
  epi *= sqrt(0.5); std::div_t divN = std::div(N, segments); opA = opA == 't' ? 'T' : (opA == 'c' ? 'C' : opA);
  int32_t N_stage1 = divN.quot + int32_t(0 < divN.rem); uint64_t V_stage1_stride = uint64_t(N_stage1) * uint64_t(std::min(N_stage1, M));
  T* V_stage1 = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&V_stage1, V_stage1_stride * uint64_t(segments) * sizeof(T), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Stage-1 allocation failed.");
  R* S_stage1 = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&S_stage1, uint64_t(N_stage1) * uint64_t(sizeof(R)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Stage-1 allocation failed.");

  int64_t locA = int64_t(0); bool transA = opA == 'T' || opA == 'C';
  std::vector<int32_t> ranks_stage1;
  for (int32_t i = 0; i < segments; ++i) {
    int32_t cols = divN.quot + int32_t(i < divN.rem);
    ranks_stage1.emplace_back(svd_dispatcher(handle, epi, opA, M, cols, std::min(cols, M), &A[locA], lda, S_stage1, &V_stage1[uint64_t(i) * V_stage1_stride], N_stage1, globalM));
    if (transA) { locA += cols; } else { locA += int64_t(cols) * int64_t(lda); }
  }

  cudaFreeAsync(S_stage1, handle->cudaStream);
  int32_t N_stage2 = std::reduce(ranks_stage1.begin(), ranks_stage1.end());
  T* A_stage2 = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&A_stage2, uint64_t(M) * uint64_t(N_stage2) * uint64_t(sizeof(T)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Stage-2 allocation failed.");
  int64_t locA2 = (locA = int64_t(0));
  for (int32_t i = 0; i < segments; ++i) {
    int32_t cols = divN.quot + int32_t(i < divN.rem);
    an_gemm(handle, opA, M, ranks_stage1[i], cols, &A[locA], lda, &V_stage1[uint64_t(i) * V_stage1_stride], N_stage1, &A_stage2[locA2], M);
    if (transA) { locA += cols; } else { locA += int64_t(cols) * int64_t(lda); } locA2 += int64_t(ranks_stage1[i]) * int64_t(M);
  }

  K = std::min(N_stage2, K);
  T* V_stage2 = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&V_stage2, uint64_t(N_stage2) * uint64_t(K) * uint64_t(sizeof(T)), handle->mempool, handle->cudaStream))
    throw std::runtime_error("Stage-2 allocation failed.");
  int32_t rank = svd_dispatcher(handle, epi, 'N', M, N_stage2, K, A_stage2, M, S, V_stage2, N_stage2, globalM);
  cudaFreeAsync(A_stage2, handle->cudaStream);

  int32_t row = 0, rowV = 0;
  for (int32_t i = 0; i < segments; ++i) {
    int32_t cols = divN.quot + int32_t(i < divN.rem);
    an_gemm(handle, 'N', cols, rank, ranks_stage1[i], &V_stage1[uint64_t(i) * V_stage1_stride], N_stage1, &V_stage2[rowV], N_stage2, &V[row], ldv);
    row += cols; rowV += ranks_stage1[i];
  }
  cudaFreeAsync(V_stage1, handle->cudaStream); cudaFreeAsync(V_stage2, handle->cudaStream);
  return rank;
}

template <class T> static inline
double check_answer(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, int32_t rank, const T* V, int32_t ldv, const T* A, int32_t lda) {
  if (M <= 0 || N <= 0) { return 0.; }
  constexpr int32_t Complex = std::is_same_v<T, cuDoubleComplex> || std::is_same_v<T, cuComplex> || std::is_same_v<T, __half2>, block = 2048;
  using type = typename std::conditional_t<Complex, cuDoubleComplex, double>;
  opA = opA == 't' ? 'T' : (opA == 'c' ? 'C' : opA); bool transA = opA == 'T' || opA == 'C'; double err = 0.;
  if (rank <= 0) {
    type* matA = nullptr; if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matA, uint64_t(block) * uint64_t(block) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    for (int32_t i = 0; i < M; i += block) { int32_t rows = std::min(M - i, block);
      for (int32_t j = 0; j < N; j += block) { int32_t cols = std::min(N - j, block);
        internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, opA, rows, cols, transA ? &A[int64_t(j) + int64_t(i) * int64_t(lda)] : &A[int64_t(i) + int64_t(j) * int64_t(lda)], lda, matA, rows);
        double nrm = 0.; cublasDnrm2_64(handle->cublasHandle, int64_t(rows) * int64_t(cols) * int64_t(Complex + 1), (const double*)matA, 1, &nrm); err += std::pow(nrm, 2);
      }
    }
    cudaFreeAsync(matA, handle->cudaStream);
  } else {
    type* matQ = nullptr, *tau = nullptr, *work = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matQ, uint64_t(N) * uint64_t(rank) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&tau, uint64_t(N) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, 'A', N, rank, V, ldv, matQ, N);
    if constexpr(Complex) {
      int32_t Lwork0 = 0, Lwork1 = 0; cusolverDnZgeqrf_bufferSize(handle->cusolverHandle, N, rank, matQ, N, &Lwork0);
      cusolverDnZungqr_bufferSize(handle->cusolverHandle, N, rank, rank, matQ, N, tau, &Lwork1); Lwork0 = std::max(Lwork0, Lwork1);
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&work, uint64_t(Lwork0) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
        throw std::runtime_error("Workspace allocation failed at answer check");
      cusolverDnZgeqrf(handle->cusolverHandle, N, rank, matQ, N, tau, work, Lwork0, nullptr);
      cusolverDnZungqr(handle->cusolverHandle, N, rank, rank, matQ, N, tau, work, Lwork0, nullptr);
    } else {
      int32_t Lwork0 = 0, Lwork1 = 0; cusolverDnDgeqrf_bufferSize(handle->cusolverHandle, N, rank, matQ, N, &Lwork0);
      cusolverDnDorgqr_bufferSize(handle->cusolverHandle, N, rank, rank, matQ, N, tau, &Lwork1); Lwork0 = std::max(Lwork0, Lwork1);
      if (cudaSuccess != cudaMallocFromPoolAsync((void**)&work, uint64_t(Lwork0) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
        throw std::runtime_error("Workspace allocation failed at answer check");
      cusolverDnDgeqrf(handle->cusolverHandle, N, rank, matQ, N, tau, work, Lwork0, nullptr);
      cusolverDnDorgqr(handle->cusolverHandle, N, rank, rank, matQ, N, tau, work, Lwork0, nullptr);
    }
    cudaFreeAsync(tau, handle->cudaStream); cudaFreeAsync(work, handle->cudaStream);

    type* matA = nullptr, *matU = nullptr;
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matA, uint64_t(block) * uint64_t(N) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    if (cudaSuccess != cudaMallocFromPoolAsync((void**)&matU, uint64_t(block) * uint64_t(rank) * uint64_t(sizeof(type)), handle->mempool, handle->cudaStream))
      throw std::runtime_error("Workspace allocation failed at answer check");
    cuDoubleComplex minus_one = make_cuDoubleComplex(-1., 0.), one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
    cublasOperation_t opt = CUBLAS_OP_T; cudaDataType Atype = CUDA_R_64F; if constexpr(Complex) { opt = CUBLAS_OP_C; Atype = CUDA_C_64F; }
    for (int32_t i = 0; i < M; i += block) {
      int32_t rows = std::min(M - i, block);
      internal::scatter_matcopy(handle->cudaStream, handle->cublasHandle, opA, rows, N, transA ? &A[int64_t(i) * int64_t(lda)] : &A[i], lda, matA, rows);
      cublasGemmEx(handle->cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, rows, rank, N, &one, matA, Atype, rows, matQ, Atype, N, &zero, matU, Atype, block, CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
      cublasGemmEx(handle->cublasHandle, CUBLAS_OP_N, opt, rows, N, rank, &minus_one, matU, Atype, block, matQ, Atype, N, &one, matA, Atype, rows, CUBLAS_COMPUTE_64F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
      double nrm = 0.; cublasDnrm2_64(handle->cublasHandle, int64_t(rows) * int64_t(N) * int64_t(Complex + 1), (const double*)matA, 1, &nrm); err += std::pow(nrm, 2);
    }
    cudaFreeAsync(matA, handle->cudaStream); cudaFreeAsync(matU, handle->cudaStream); cudaFreeAsync(matQ, handle->cudaStream);
  } return err;
}

extern "C" int32_t hyacinInterpolateFit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const void* A, int32_t lda, int32_t* jpvt, void* V, int32_t ldv, int64_t globalM, hyacinPrecision_t ComputeType) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; switch (ComputeType) {
    case HYACIN_F64: return interp_dispatcher(handle, epi, opA, M, N, K, (const double*)A, lda, jpvt, (double*)V, ldv, globalM);
    case HYACIN_F32: return interp_dispatcher(handle, epi, opA, M, N, K, (const float*)A, lda, jpvt, (float*)V, ldv, globalM);
    case HYACIN_F16: return interp_dispatcher(handle, epi, opA, M, N, K, (const __half*)A, lda, jpvt, (__half*)V, ldv, globalM);
    case HYACIN_F64_COMPLEX: return interp_dispatcher(handle, epi, opA, M, N, K, (const cuDoubleComplex*)A, lda, jpvt, (cuDoubleComplex*)V, ldv, globalM);
    case HYACIN_F32_COMPLEX: return interp_dispatcher(handle, epi, opA, M, N, K, (const cuComplex*)A, lda, jpvt, (cuComplex*)V, ldv, globalM);
    case HYACIN_F16_COMPLEX: return interp_dispatcher(handle, epi, opA, M, N, K, (const __half2*)A, lda, jpvt, (__half2*)V, ldv, globalM);
    default: return 0;
  }
}

extern "C" int32_t hyacinTsvdFit(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const void* A, int32_t lda, void* S, void* V, int32_t ldv, int64_t globalM, hyacinPrecision_t ComputeType) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; switch (ComputeType) {
    case HYACIN_F64: return svd_dispatcher(handle, epi, opA, M, N, K, (const double*)A, lda, (double*)S, (double*)V, ldv, globalM);
    case HYACIN_F32: return svd_dispatcher(handle, epi, opA, M, N, K, (const float*)A, lda, (float*)S, (float*)V, ldv, globalM);
    case HYACIN_F16: return svd_dispatcher(handle, epi, opA, M, N, K, (const __half*)A, lda, (__half*)S, (__half*)V, ldv, globalM);
    case HYACIN_F64_COMPLEX: return svd_dispatcher(handle, epi, opA, M, N, K, (const cuDoubleComplex*)A, lda, (double*)S, (cuDoubleComplex*)V, ldv, globalM);
    case HYACIN_F32_COMPLEX: return svd_dispatcher(handle, epi, opA, M, N, K, (const cuComplex*)A, lda, (float*)S, (cuComplex*)V, ldv, globalM);
    case HYACIN_F16_COMPLEX: return svd_dispatcher(handle, epi, opA, M, N, K, (const __half2*)A, lda, (__half*)S, (__half2*)V, ldv, globalM);
    default: return 0;
  }
}

extern "C" int32_t hyacinTsvdFitSegmentN(const hyacinHandle_t* handle, double epi, char opA, int32_t M, int32_t N, int32_t K, const void* A, int32_t lda, void* S, void* V, int32_t ldv, int64_t globalM, hyacinPrecision_t ComputeType) {
  globalM = globalM <= int64_t(0) ? int64_t(M) : globalM; switch (ComputeType) {
    case HYACIN_F64: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const double*)A, lda, (double*)S, (double*)V, ldv, globalM);
    case HYACIN_F32: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const float*)A, lda, (float*)S, (float*)V, ldv, globalM);
    case HYACIN_F16: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const __half*)A, lda, (__half*)S, (__half*)V, ldv, globalM);
    case HYACIN_F64_COMPLEX: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const cuDoubleComplex*)A, lda, (double*)S, (cuDoubleComplex*)V, ldv, globalM);
    case HYACIN_F32_COMPLEX: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const cuComplex*)A, lda, (float*)S, (cuComplex*)V, ldv, globalM);
    case HYACIN_F16_COMPLEX: return svd_segN_dispatcher(handle, epi, opA, M, N, K, (const __half2*)A, lda, (__half*)S, (__half2*)V, ldv, globalM);
    default: return 0;
  }
}

extern "C" double hyacinCheckNorm(const hyacinHandle_t* handle, char opA, int32_t M, int32_t N, const void* A, int32_t lda, int32_t rank, const void* V, int32_t ldv, hyacinPrecision_t ComputeType) {
  switch (ComputeType) {
    case HYACIN_F64: return check_answer(handle, opA, M, N, rank, (const double*)V, ldv, (const double*)A, lda);
    case HYACIN_F32: return check_answer(handle, opA, M, N, rank, (const float*)V, ldv, (const float*)A, lda);
    case HYACIN_F16: return check_answer(handle, opA, M, N, rank, (const __half*)V, ldv, (const __half*)A, lda);
    case HYACIN_F64_COMPLEX: return check_answer(handle, opA, M, N, rank, (const cuDoubleComplex*)V, ldv, (const cuDoubleComplex*)A, lda);
    case HYACIN_F32_COMPLEX: return check_answer(handle, opA, M, N, rank, (const cuComplex*)V, ldv, (const cuComplex*)A, lda);
    case HYACIN_F16_COMPLEX: return check_answer(handle, opA, M, N, rank, (const __half2*)V, ldv, (const __half2*)A, lda);
    default: return 0.0;
  }
}
