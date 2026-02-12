
#include <examples.hpp>
#include <hyacin.h>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

const int32_t umax_exp_extra = 6; // extra bits for exponent difference;

enum class PostProcessPrecision { FP64, FP32, FP64_COMPLEX, FP32_COMPLEX };

template <PostProcessPrecision prec>
inline void cublas_trsm(cublasHandle_t handle, int32_t M, int32_t N, void* R, int32_t ldr) {
  if (M < N) {
    if constexpr(prec == PostProcessPrecision::FP64) {
      double one = 1., *r = (double*)R;
      cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, r, ldr, &r[M * ldr], ldr);
    }
    else if constexpr(prec == PostProcessPrecision::FP32) {
      float one = 1.f, *r = (float*)R;
      cublasStrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, &one, r, ldr, &r[M * ldr], ldr);
    }
    else if constexpr(prec == PostProcessPrecision::FP64_COMPLEX) {
      std::complex<double> one(1., 0.); cuDoubleComplex* r = (cuDoubleComplex*)R;
      cublasZtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuDoubleComplex*)&one, r, ldr, &r[M * ldr], ldr);
    }
    else if constexpr(prec == PostProcessPrecision::FP32_COMPLEX) {
      std::complex<float> one(1.f, 0.f); cuComplex* r = (cuComplex*)R;
      cublasCtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, M, N - M, (cuComplex*)&one, r, ldr, &r[M * ldr], ldr);
    }
  }
}

template <class elem_t> struct identity {
  elem_t* __restrict__ A;
  const elem_t one;
  int64_t M, lda;
  identity(int64_t M, elem_t* A, int64_t lda, elem_t one) :
    A(A), one(one), M(M), lda(lda) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int32_t diag = int32_t(x == y);
    A[y + x * lda] = diag ? one : elem_t();
  }
};

template <PostProcessPrecision prec>
inline void strided_identity(cudaStream_t stream, int32_t M, int32_t N, void* A, int32_t lda) {
  int64_t iter_items = int64_t(N) * int64_t(M);
  thrust::counting_iterator<int64_t> iter(0);

  if constexpr(prec == PostProcessPrecision::FP64) {
    identity<int64_t> id(M, (int64_t*)A, lda, 0x3FF0000000000000LL);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if constexpr(prec == PostProcessPrecision::FP32) {
    identity<int32_t> id(M, (int32_t*)A, lda, 0x3F800000);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if constexpr(prec == PostProcessPrecision::FP64_COMPLEX) {
    identity<int4> id(M, (int4*)A, lda, make_int4(0x0, 0x3FF00000, 0x0, 0x0));
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
  else if constexpr(prec == PostProcessPrecision::FP32_COMPLEX) {
    identity<int64_t> id(M, (int64_t*)A, lda, 0x3F800000LL);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, iter_items, id);
  }
}

const int32_t clen = 16384;
__constant__ int32_t cpiv[clen];

template <class elem_t> struct scatter_copy {
  const elem_t* __restrict__ A;
  elem_t* __restrict__ B;
  int64_t M, lda, ldb;
  scatter_copy(int64_t M, const elem_t* A, int64_t lda, elem_t* B, int64_t ldb) :
    A(A), B(B), M(M), lda(lda), ldb(ldb) {}
  
  __device__ __forceinline__ void operator()(int64_t i) {
    int64_t x = i / M, y = i - x * M;
    int64_t px = int64_t(cpiv[x] - 1);
    B[y + px * ldb] = A[y + x * lda];
  }
};

template <class elem_t>
inline void copy_scatter(cudaStream_t stream, int32_t M, int32_t N, const int32_t* jpiv, const void* A, int32_t lda, void* B, int32_t ldb) {
  thrust::counting_iterator<int64_t> iter(0);
  for (int32_t i = 0; i < N; i += clen) {
    int32_t cols = std::min(N - i, clen);
    int64_t elements = int64_t(cols) * int64_t(M);
    cudaMemcpyToSymbolAsync(cpiv, &jpiv[i], int64_t(cols) * sizeof(int32_t), 0, cudaMemcpyDefault, stream);
    scatter_copy<elem_t> perm(M, (const elem_t*)A, lda, &((elem_t*)B)[int64_t(i) * int64_t(ldb)], ldb);
    thrust::for_each_n(thrust::cuda::par_nosync.on(stream), iter, elements, perm);
  }
}

template <PostProcessPrecision prec>
inline void interp_pp_uni(cudaStream_t stream, cublasHandle_t handle, int32_t M, int32_t N, void* R, int32_t ldr, const int32_t* ipiv, void* X, int32_t ldx) {
  if (0 < M) {
    cublas_trsm<prec>(handle, M, N, R, ldr);
    strided_identity<prec>(stream, M, M, R, ldr);
    if constexpr(prec == PostProcessPrecision::FP64 || prec == PostProcessPrecision::FP32_COMPLEX)
      copy_scatter<int64_t>(stream, M, N, ipiv, R, ldr, X, ldx);
    else if constexpr(prec == PostProcessPrecision::FP32)
      copy_scatter<int32_t>(stream, M, N, ipiv, R, ldr, X, ldx);
    else if constexpr(prec == PostProcessPrecision::FP64_COMPLEX)
      copy_scatter<int4>(stream, M, N, ipiv, R, ldr, X, ldx);
  }
}

int32_t device::interp_decomp_f64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *R = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&R, int64_t(N) * int64_t(N) * sizeof(double));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, N, p, umax, HYACIN_F64, A, lda, (int32_t*)piv, HYACIN_F64, R, N, precC, dev_work, pinned_work, alg);
  interp_pp_uni<PostProcessPrecision::FP64>(stream, handle, rank, N, R, N, (int32_t*)piv, X, ldx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(R);
  cudaFree(piv);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t device::interp_decomp_f32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const float* A, int32_t lda, int32_t* jpiv, float* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F32, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *R = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&R, int64_t(N) * int64_t(N) * sizeof(float));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, N, p, umax, HYACIN_F32, A, lda, (int32_t*)piv, HYACIN_F32, R, N, precC, dev_work, pinned_work, alg);
  interp_pp_uni<PostProcessPrecision::FP32>(stream, handle, rank, N, R, N, (int32_t*)piv, X, ldx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(R);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t device::interp_decomp_cf64(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, int32_t* jpiv, std::complex<double>* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F64_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *R = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&R, int64_t(N) * int64_t(N) * sizeof(std::complex<double>));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, N, p, umax, HYACIN_F64_COMPLEX, A, lda, (int32_t*)piv, HYACIN_F64_COMPLEX, R, N, precC, dev_work, pinned_work, alg);
  interp_pp_uni<PostProcessPrecision::FP64_COMPLEX>(stream, handle, rank, N, R, N, (int32_t*)piv, X, ldx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(R);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t device::interp_decomp_cf32(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* X, int32_t ldx) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, umax_exp_extra, &umax, HYACIN_F32_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *R = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&R, int64_t(N) * int64_t(N) * sizeof(std::complex<float>));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, N, p, umax, HYACIN_F32_COMPLEX, A, lda, (int32_t*)piv, HYACIN_F32_COMPLEX, R, N, precC, dev_work, pinned_work, alg);
  interp_pp_uni<PostProcessPrecision::FP32_COMPLEX>(stream, handle, rank, N, R, N, (int32_t*)piv, X, ldx);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(R);
  cudaFreeHost(pinned_work);
  return rank;
}
