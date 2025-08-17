
#include <hyacinth.hpp>
#include <random>
#include <iostream>
#include <algorithm>

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

  int32_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int32_t N = std::min(M, 2 < argc ? std::atoi(argv[2]) : 128);
  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-6;
  std::vector<float> matA(M * N);
  std::vector<int32_t> ipiv(N);

  std::mt19937_64 gen;
  std::normal_distribution<float> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return dist(gen); });

  std::cout << "SGEQP3 <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << "\n";

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

  cudaEventRecord(start, stream);
  int32_t ret = device::sgeqp3_ronly(stream, handle, epi, M, N, d_A, M, ipiv.data());
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  if (M <= 2048 && N <= 2048) {
    std::vector<float> matB(M * N);
    cudaMemcpy(matB.data(), d_A, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<int32_t> jpiv(N, 0);
    std::vector<float> tau(N);
    LAPACKE_sgeqp3(LAPACK_COL_MAJOR, M, N, matA.data(), M, jpiv.data(), tau.data());

    int32_t err_int = 0;
    for (int32_t i = 0; i < N; ++i) {
      err_int += std::abs(jpiv[i] - ipiv[i]);
      if (matA[i * (M + 1)] < 0.)
        cblas_sscal(N, -1.f, &(matA.data())[i], M);
    }
  
    double nrm = 0., err = 0.;
    for (int32_t j = 0; j < N; ++j)
      for (int32_t i = 0; i <= j; ++i) {
        err += std::norm(matB[i + j * M] - matA[i + j * M]);
        nrm += std::norm(matA[i + j * M]);
    }

    std::cout << "Pivot Err: " << err_int << "\n";
    std::cout << "Err: " << std::sqrt(err / nrm) << "\n" << std::endl;
  }

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  std::cout << "Cholesky return: " << ret << std::endl;
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "GFLOPs: " << double(flops) * 1.e-6 / milliseconds << "\n";

  cudaFree(d_A);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
