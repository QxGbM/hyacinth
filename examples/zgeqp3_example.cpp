
#include <examples.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

double check_answer(int32_t M, int32_t N, int32_t rank, const std::complex<double>* A, int32_t lda, const int32_t* jpiv, const std::complex<double>* tau, const std::complex<double>* B, int32_t ldb) {
  if (rank <= 0)
    return std::numeric_limits<double>::quiet_NaN();
  std::vector<std::complex<double>> matQ(M * N, std::complex<double>(0., 0.));
  std::complex<double> one(1., 0.);
  LAPACKE_zlacpy(LAPACK_COL_MAJOR, 'A', M, N, (const lapack_complex_double*)A, lda, (lapack_complex_double*)&matQ[0], M);
  LAPACKE_zungqr(LAPACK_COL_MAJOR, M, rank, rank, (lapack_complex_double*)&matQ[0], M, (const lapack_complex_double*)tau);
  cblas_ztrmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, A, lda, &matQ[0], M);

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

  std::vector<std::complex<double>> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, sep, M, N, &matA[0], M);

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

  std::complex<double>* d_A = nullptr, *d_tau = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_tau), N * sizeof(std::complex<double>));
  cudaMemcpy(d_A, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  device::zgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, &ipiv[0], d_tau);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemset(d_tau, 0, N * sizeof(std::complex<double>));
  cudaMemcpy(d_A, &matA[0], M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::zgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, &ipiv[0], d_tau);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<std::complex<double>> matB(M * N), tau(N);
  cudaMemcpy(&matB[0], d_A, M * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  cudaMemcpy(&tau[0], d_tau, N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);
  double err = check_answer(M, N, ret == 0 ? N : (ret - 1), &matB[0], M, &ipiv[0], &tau[0], &matA[0], M);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "ZGEQP3," << M << "," << N << "," << epi << "," << err << "," << ret << "," << milliseconds << "," << gflops << std::endl;

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
