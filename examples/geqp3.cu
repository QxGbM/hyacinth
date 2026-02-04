
#include <examples.hpp>
#include <hyacin.h>
#include <cuComplex.h>
#include <vector>
#include <numeric>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

const int32_t clen = 16384;
__constant__ int32_t cpiv[clen];

template <class elem_t> struct permute_copy {
  const elem_t* __restrict__ A;
  elem_t* __restrict__ B;
  int64_t M, lda, ldb;
  permute_copy(int64_t M, const elem_t* A, int64_t lda, elem_t* B, int64_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int64_t px = int64_t(cpiv[x] - 1);
    B[y + x * ldb] = A[y + px * lda];
  }
};

template <class elem_t>
inline void copy_gather(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  for (int32_t i = 0; i < N; i += clen) {
    int32_t cols = std::min(N - i, clen);
    int64_t elements = int64_t(cols) * int64_t(M);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(cols) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);
    permute_copy<elem_t> perm(M, (const elem_t*)A, lda, &((elem_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
  }
}

inline void workspace_realloc(cudaStream_t stream, void** ptr, uint64_t* bytes_old, uint64_t bytes_required) {
  if (*bytes_old < bytes_required) {
    void* workspace = nullptr;
    cudaStreamSynchronize(stream);
    cudaMalloc(&workspace, bytes_required);
    if (*ptr)
      cudaFree(*ptr);
    *ptr = workspace;
    *bytes_old = bytes_required;
  }
}

inline void qr_pp_f64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, double* A, int32_t lda, const int32_t* ipiv, double* tau, void** Workspace, uint64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  double* R = &A[int64_t(K) * int64_t(lda)];
  cusolverDnDgeqrf_bufferSize(cusolverH, M, K, A, lda, &l1);
  if (K < N)
    cusolverDnDormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, &l2);
  uint64_t bytes_required = std::max(uint64_t(M) * uint64_t(N), uint64_t(1 + std::max(l1, l2))) * uint64_t(sizeof(double));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  double* work = (double*)*Workspace;
  copy_gather<int64_t>(stream, M, N, (int32_t*)tau, A, lda, work, M);
  cudaMemcpy2DAsync(A, int64_t(lda) * sizeof(double), work, int64_t(M) * sizeof(double), int64_t(M) * sizeof(double), N, cudaMemcpyDeviceToDevice, stream);
  cusolverDnDgeqrf(cusolverH, M, K, A, lda, tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnDormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(double), stream);
  }
}

int32_t device::dgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, double* A, int32_t lda, int32_t* jpiv, double* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 

  if (mode == 'R' || mode == 'r')
    rank = hyacinXcpqrk(cublasH, 'R', epi, M, N, N, p, umax, HYACIN_F64, A, lda, &hpiv[0], HYACIN_F64, A, lda, precC, work, dpiv, alg);
  else if (mode != 'P' && mode != 'p') {
    rank = hyacinXcpqrk(cublasH, 'N', epi, M, N, N, p, umax, HYACIN_F64, A, lda, &hpiv[0], HYACIN_F64, nullptr, 0, precC, work, dpiv, alg);
    qr_pp_f64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

inline void qr_pp_f32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, float* A, int32_t lda, const int32_t* ipiv, float* tau, void** Workspace, uint64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  float* R = &A[int64_t(K) * int64_t(lda)];
  cusolverDnSgeqrf_bufferSize(cusolverH, M, K, A, lda, &l1);
  if (K < N)
    cusolverDnSormqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, &l2);
  uint64_t bytes_required = std::max(uint64_t(M) * uint64_t(N), uint64_t(1 + std::max(l1, l2))) * uint64_t(sizeof(float));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  float* work = (float*)*Workspace;
  copy_gather<int32_t>(stream, M, N, (int32_t*)tau, A, lda, work, M);
  cudaMemcpy2DAsync(A, int64_t(lda) * sizeof(float), work, int64_t(M) * sizeof(float), int64_t(M) * sizeof(float), N, cudaMemcpyDeviceToDevice, stream);
  cusolverDnSgeqrf(cusolverH, M, K, A, lda, tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnSormqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, M, N - K, K, A, lda, tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(float), stream);
  }
}

int32_t device::sgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, float* A, int32_t lda, int32_t* jpiv, float* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F32, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 

  if (mode == 'R' || mode == 'r')
    rank = hyacinXcpqrk(cublasH, 'R', epi, M, N, N, p, umax, HYACIN_F32, A, lda, &hpiv[0], HYACIN_F32, A, lda, precC, work, dpiv, alg);
  else if (mode != 'P' && mode != 'p') {
    rank = hyacinXcpqrk(cublasH, 'N', epi, M, N, N, p, umax, HYACIN_F32, A, lda, &hpiv[0], HYACIN_F32, nullptr, 0, precC, work, dpiv, alg);
    qr_pp_f32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

inline void qr_pp_cf64(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<double>* A, int32_t lda, const int32_t* ipiv, std::complex<double>* tau, void** Workspace, uint64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  cuDoubleComplex* R = (cuDoubleComplex*)&A[int64_t(K) * int64_t(lda)];
  cusolverDnZgeqrf_bufferSize(cusolverH, M, K, (cuDoubleComplex*)A, lda, &l1);
  if (K < N)
    cusolverDnZunmqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, M, N - K, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, R, lda, &l2);
  uint64_t bytes_required = std::max(uint64_t(M) * uint64_t(N), uint64_t(1 + std::max(l1, l2))) * uint64_t(sizeof(cuDoubleComplex));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  cuDoubleComplex* work = (cuDoubleComplex*)*Workspace;
  copy_gather<int4>(stream, M, N, (int32_t*)tau, A, lda, work, M);
  cudaMemcpy2DAsync(A, int64_t(lda) * sizeof(std::complex<double>), work, int64_t(M) * sizeof(std::complex<double>), int64_t(M) * sizeof(std::complex<double>), N, cudaMemcpyDeviceToDevice, stream);
  cusolverDnZgeqrf(cusolverH, M, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnZunmqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, M, N - K, K, (cuDoubleComplex*)A, lda, (cuDoubleComplex*)tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(cuDoubleComplex), stream);
  }
}

int32_t device::zgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F64_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 
  
  if (mode == 'R' || mode == 'r')
    rank = hyacinXcpqrk(cublasH, 'R', epi, M, N, N, p, umax, HYACIN_F64_COMPLEX, A, lda, &hpiv[0], HYACIN_F64_COMPLEX, A, lda, precC, work, dpiv, alg);
  else if (mode != 'P' && mode != 'p') {
    rank = hyacinXcpqrk(cublasH, 'N', epi, M, N, N, p, umax, HYACIN_F64_COMPLEX, A, lda, &hpiv[0], HYACIN_F64_COMPLEX, nullptr, 0, precC, work, dpiv, alg);
    qr_pp_cf64(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}

inline void qr_pp_cf32(cudaStream_t stream, cusolverDnHandle_t cusolverH, int32_t M, int32_t N, int32_t K, std::complex<float>* A, int32_t lda, const int32_t* ipiv, std::complex<float>* tau, void** Workspace, uint64_t* Lwork) {
  int32_t l1 = 0, l2 = 0;
  cuComplex* R = (cuComplex*)&A[int64_t(K) * int64_t(lda)];
  cusolverDnCgeqrf_bufferSize(cusolverH, M, K, (cuComplex*)A, lda, &l1);
  if (K < N)
    cusolverDnCunmqr_bufferSize(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, M, N - K, K, (cuComplex*)A, lda, (cuComplex*)tau, R, lda, &l2);
  uint64_t bytes_required = std::max(uint64_t(M) * uint64_t(N), uint64_t(1 + std::max(l1, l2))) * uint64_t(sizeof(cuComplex));
  workspace_realloc(stream, Workspace, Lwork, bytes_required);

  cudaMemcpyAsync(tau, ipiv, sizeof(int32_t) * N, cudaMemcpyDefault, stream);
  cuComplex* work = (cuComplex*)*Workspace;
  copy_gather<int64_t>(stream, M, N, (int32_t*)tau, A, lda, work, M);
  cudaMemcpy2DAsync(A, int64_t(lda) * sizeof(std::complex<float>), work, int64_t(M) * sizeof(std::complex<float>), int64_t(M) * sizeof(std::complex<float>), N, cudaMemcpyDeviceToDevice, stream);
  cusolverDnCgeqrf(cusolverH, M, K, (cuComplex*)A, lda, (cuComplex*)tau, work, l1, (int32_t*)&work[l1]);

  if (K < N) {
    cusolverDnCunmqr(cusolverH, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, M, N - K, K, (cuComplex*)A, lda, (cuComplex*)tau, R, lda, work, l2, (int32_t*)&work[l2]);
    cudaMemsetAsync(&tau[K], 0, int64_t(N - K) * sizeof(cuComplex), stream);
  }
}

int32_t device::cgeqp3(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, char mode, double epi, int32_t M, int32_t N, std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* tau) {
  cudaStream_t stream; cublasGetStream(cublasH, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F32_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* work = nullptr;
  int32_t* dpiv = nullptr;
  cudaMalloc(&work, dev_work_bytes);
  cudaMallocHost((void**)(&dpiv), pinned_work_bytes);
  std::vector<int32_t> hpiv(N);
  std::iota(hpiv.begin(), hpiv.end(), 1);

  int32_t rank = N, p = 0; 

  if (mode == 'R' || mode == 'r')
    rank = hyacinXcpqrk(cublasH, 'R', epi, M, N, N, p, umax, HYACIN_F32_COMPLEX, A, lda, &hpiv[0], HYACIN_F32_COMPLEX, A, lda, precC, work, dpiv, alg);
  else if (mode != 'P' && mode != 'p') {
    rank = hyacinXcpqrk(cublasH, 'N', epi, M, N, N, p, umax, HYACIN_F32_COMPLEX, A, lda, &hpiv[0], HYACIN_F32_COMPLEX, nullptr, 0, precC, work, dpiv, alg);
    qr_pp_cf32(stream, cusolverH, M, N, rank, A, lda, &hpiv[0], tau, &work, &dev_work_bytes);
  }

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, &hpiv[0], sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(work);
  cudaFreeHost(dpiv);
  return rank == N ? 0 : (rank + 1);
}
