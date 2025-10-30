
#include <hyacin.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-6;
  std::vector<float> matA(M * N);
  std::vector<int32_t> ipiv(N);

  std::mt19937_64 gen(42);
  std::normal_distribution<float> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return dist(gen); });

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

  float* d_A = nullptr, *d_tau = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(float));
  cudaMalloc((void**)(&d_tau), N * sizeof(float));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  device::sgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, ipiv.data(), d_tau);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::sgeqp3(cublasH, cusolverH, 'Q', epi, M, N, d_A, M, ipiv.data(), d_tau);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  double err = 0.;
  std::vector<float> matB(M * N), tau(N), matQ(M * N, 0.);
  cudaMemcpy(matB.data(), d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  cudaMemcpy(tau.data(), d_tau, N * sizeof(float), cudaMemcpyDeviceToHost);

  int32_t rank = ret == 0 ? N : (ret - 1);
  std::copy_n(matB.begin(), M * rank, matQ.begin());
  LAPACKE_sorgqr(LAPACK_COL_MAJOR, M, rank, rank, matQ.data(), M, tau.data());
  cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, 1.f, matB.data(), M, matQ.data(), M);

  double nrm = 0.;
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      int32_t j2 = ipiv[j] - 1;
      err += std::norm(matQ[i + j * M] - matA[i + j2 * M]);
      nrm += std::norm(matA[i + j * M]);
  }
  err = std::sqrt(err / nrm);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "SGEQP3," << M << "," << N << "," << epi << "," << err << "," << ret << "," << milliseconds << "," << gflops << std::endl;

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
