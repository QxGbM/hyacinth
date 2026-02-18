
#include <hyacin.h>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>
#include <complex>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t om, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(-(j + height));
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + om);
      double d = std::abs(vi - vj);
      A[i + j * lda] = std::cos(w * d) / d;
    }
  }
}

double check_answer(int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, const int32_t* jpiv, double* R, int32_t ldr) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, rank, N - rank, 1., R, ldr, &R[int64_t(rank) * int64_t(ldr)], ldr);
  LAPACKE_dlaset(LAPACK_COL_MAJOR, 'A', rank, rank, 0., 1., R, ldr);

  std::vector<double> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', M, N, A, lda, &matB[0], M);
  for (int32_t i = 0; i < N; ++i) {
    if (i < rank)
      cblas_dcopy(M, &matB[int64_t(jpiv[i] - 1) * int64_t(M)], 1, &matC[int64_t(i) * int64_t(M)], 1);
    cblas_dcopy(rank, &R[int64_t(i) * int64_t(ldr)], 1, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)], 1);
  }

  double nrm = cblas_dnrm2(matB.size(), &matB[0], 1);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, -1., &matC[0], M, &matR[0], rank, 1., &matB[0], M);
  double err = cblas_dnrm2(matB.size(), &matB[0], 1);
  return err / nrm;
}

int32_t geqp3_ronly(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const double* A, int32_t lda, int32_t* jpiv, double* R, int32_t ldr) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F64, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, N, p, umax, HYACIN_F64, A, lda, (int32_t*)piv, HYACIN_F64, R, ldr, precC, dev_work, pinned_work, alg);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(piv);
  cudaFreeHost(pinned_work);
  return rank;
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  double omega = 4 < argc ? std::atof(argv[4]) : 1.;

  std::vector<double> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, 0, M, N, &matA[0], M);

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(double));
  cudaMalloc((void**)(&d_X), N * N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  geqp3_ronly(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = geqp3_ronly(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  std::vector<double> matX(N * N);
  cudaMemcpy(matX.data(), d_X, N * N * sizeof(double), cudaMemcpyDeviceToHost);
  double rel_err = check_answer(rank, M, N, matA.data(), M, ipiv.data(), matX.data(), N);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // QR flops = 2mnk - nk^2 + 1/3k^3 + k(n-k)
  int64_t qr_flops = (int64_t(M) * int64_t(N) * int64_t(rank) * 2) - (int64_t(N) * int64_t(rank) * int64_t(rank)) + (int64_t(rank) * int64_t(rank) * int64_t(rank) / 3) + (int64_t(rank) * int64_t(N - rank));
  int64_t trsm_flops = int64_t(N) * int64_t(rank) * int64_t(rank);
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / milliseconds;

  std::cout << "D-LRA," << M << "," << N << "," << epi << "," << rel_err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_X);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
