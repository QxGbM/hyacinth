
#include <hyacin.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, double* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::cos(w * d) / d;
    }
  }
}

double check_answer(int32_t M, int32_t N, int32_t rank, const double* A, int32_t lda, const int32_t* jpiv, const double* tau, const double* B, int32_t ldb) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();
  std::vector<double> matQ(M * N, 0.);
  LAPACKE_dlacpy(LAPACK_COL_MAJOR, 'A', M, N, A, lda, &matQ[0], M);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, M, rank, rank, &matQ[0], M, tau);
  cblas_dtrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, 1., A, lda, &matQ[0], M);

  double err = 0., nrm = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      int32_t j2 = jpiv[j] - 1;
      err += std::norm(matQ[i + j * M] - B[i + j2 * ldb]);
      nrm += std::norm(B[i + j2 * ldb]);
  }
  return std::sqrt(err / nrm);
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
  int32_t sep = 5 < argc ? std::atoi(argv[5]) : 0;

  std::vector<double> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, sep, M, N, matA.data(), M);

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr, *d_tau = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(double));
  cudaMalloc((void**)(&d_tau), N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  device::dgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, ipiv.data(), d_tau);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::dgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, ipiv.data(), d_tau);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<double> matB(M * N), tau(N);
  cudaMemcpy(matB.data(), d_A, M * N * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(tau.data(), d_tau, N * sizeof(double), cudaMemcpyDeviceToHost);
  double err = check_answer(M, N, ret == 0 ? N : (ret - 1), &matB[0], M, &ipiv[0], &tau[0], &matA[0], M);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "DGEQP3," << M << "," << N << "," << epi << "," << err << "," << ret << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_tau);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
