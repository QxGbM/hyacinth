
#include <hyacinth.hpp>
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

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  std::vector<double> matA(M * N);
  std::vector<int32_t> ipiv(N);

  std::mt19937_64 gen(42);
  std::normal_distribution<double> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return dist(gen); });

  std::cout << "DGEQP3 <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << "\n";

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::dgeqp3_ronly(stream, handle, epi, M, N, d_A, M, ipiv.data());
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  if (M <= 2048 && N <= 2048) {
    std::vector<double> matB(M * N);
    cudaMemcpy(matB.data(), d_A, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<int32_t> jpiv(N, 0);
    std::vector<double> tau(N);
    LAPACKE_dgeqp3(LAPACK_COL_MAJOR, M, N, matA.data(), M, jpiv.data(), tau.data());

    int32_t err_int = 0;
    for (int32_t i = 0; i < N; ++i) {
      err_int += int32_t(jpiv[i] != ipiv[i]);
      if (matA[i * (M + 1)] < 0.)
        cblas_dscal(N, -1., &(matA.data())[i], M);
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
