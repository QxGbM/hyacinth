
#include <hyacinth.hpp>
#include <random>
#include <iostream>
#include <algorithm>
#include <eigen3/Eigen/Dense>

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
  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  std::vector<double> matA(M * N);
  std::vector<int32_t> ipiv(N);

  std::mt19937_64 gen(42);
  std::normal_distribution<double> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return dist(gen); });

  /*Eigen::Map<Eigen::MatrixXd> mapA(matA.data(), M, N);
  Eigen::MatrixXd AAT = mapA * mapA.adjoint();
  for (int32_t i = 0; i < 5; ++i) {
    AAT /= AAT.norm();
    AAT = AAT * AAT.adjoint();
  }
  mapA = AAT * mapA;*/

  std::cout << "F64 LR-APPROX <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << "\n";

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

  cudaEventRecord(start, stream);
  int32_t rank = device::interp_decomp_f64(stream, handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  if (M <= 2048 && N <= 2048) {
    std::vector<double> matB(M * N), matC(M * rank), matX(N * N);
    cudaMemcpy(matX.data(), d_X, N * N * sizeof(double), cudaMemcpyDeviceToHost);

    for (int32_t i = 0; i < rank; ++i) {
      int32_t col = (ipiv[i] - 1);
      std::copy_n(&matA[uint64_t(col) * uint64_t(M)], M, &matC[uint64_t(i) * uint64_t(M)]);
    }
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, 1., &matC[0], M, &matX[0], N, 0., &matB[0], M);
  
    double nrm = 0., err = 0.;
    for (int32_t j = 0; j < N; ++j)
      for (int32_t i = 0; i < M; ++i) {
        err += std::norm(matB[i + j * M] - matA[i + j * M]);
        nrm += std::norm(matA[i + j * M]);
    }
    
    std::cout << "Approx Err: " << std::sqrt(err / nrm) << "\n" << std::endl;
  }

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t qr_flops = (int64_t(N) * int64_t(N) * int64_t(N) * -2 / 3) + (int64_t(M) * int64_t(N) * int64_t(N) * 2);
  int64_t trsm_flops = int64_t(N) * int64_t(rank) * int64_t(rank);
  std::cout << "Matrix A rank: " << rank << std::endl;
  std::cout << "Time: " << milliseconds << " ms\n";
  std::cout << "Total GFLOPs: " << double(qr_flops + trsm_flops) * 1.e-6 / milliseconds << "\n";

  cudaFree(d_A);
  cudaFree(d_X);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
