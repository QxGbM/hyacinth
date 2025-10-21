
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
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float* d_A = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(float));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  device::sgeqp3_ronly(stream, handle, epi, M, N, d_A, M, ipiv.data());
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::sgeqp3_ronly(stream, handle, epi, M, N, d_A, M, ipiv.data());
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  int32_t err_int = 0;
  double err = 0.;
  if (M <= 2048 && N <= 2048) {
    std::vector<float> matB(M * N);
    cudaMemcpy(matB.data(), d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int32_t> jpiv(N, 0);
    std::vector<float> tau(N);
    LAPACKE_sgeqp3(LAPACK_COL_MAJOR, M, N, matA.data(), M, jpiv.data(), tau.data());

    for (int32_t i = 0; i < N; ++i) {
      err_int += int32_t(jpiv[i] != ipiv[i]);
      if (matA[i * (M + 1)] < 0.)
        cblas_sscal(N, -1.f, &(matA.data())[i], M);
    }
  
    double nrm = 0.;
    for (int32_t j = 0; j < N; ++j)
      for (int32_t i = 0; i <= j; ++i) {
        err += std::norm(matB[i + j * M] - matA[i + j * M]);
        nrm += std::norm(matA[i + j * M]);
    }
    err = std::sqrt(err / nrm);
  }

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "SGEQP3," << M << "," << N << "," << epi << "," << err_int << "," << err << "," << ret << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
