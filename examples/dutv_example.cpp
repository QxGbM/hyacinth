
#include <common.hpp>
#include <iostream>

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  int64_t K = 3 < argc ? std::atoi(argv[3]) : N;
  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;
  K = std::min(K, N);

  std::vector<double> matA(M * N);
  make_2D_oscillatory(omega, 0, 0, M, N, &matA[0], M);

  cudaStream_t stream;
  cublasHandle_t cublasH;
  cusolverDnHandle_t cusolverH;
  cusolverDnParams_t params;

  cudaStreamCreate(&stream);
  cublasCreate(&cublasH);
  cublasSetStream(cublasH, stream);
  cusolverDnCreate(&cusolverH);
  cusolverDnSetStream(cusolverH, stream);
  cusolverDnCreateParams(&params);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  double* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(double));
  cudaMalloc((void**)(&d_V), K * N * sizeof(double));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  utv_factorize_phase1(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(double), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = utv_factorize_phase1(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<double> matU(M * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(double), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(M, N, rank, &matU[0], M, &matV[0], K, &matA[0], M);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(M) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(M) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / milliseconds;

  std::cout << "D-UTVK," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
