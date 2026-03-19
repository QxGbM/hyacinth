
#include <common.hpp>
#include <iostream>

template <class T> inline void run(char prec, int64_t M, int64_t N, int64_t K, double epi, double omega) {
  std::vector<T> matA(M * N);
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

  T* d_A = nullptr, *d_V = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_V), K * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = svd_fit_transform(stream, cublasH, cusolverH, params, epi, M, N, K, d_A, M, d_V, K);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  std::vector<T> matU(M * K), matV(K * N);
  cudaMemcpy(matU.data(), d_A, M * K * sizeof(T), cudaMemcpyDeviceToHost);
  cudaMemcpy(matV.data(), d_V, K * N * sizeof(T), cudaMemcpyDeviceToHost);

  std::pair<double, double> ret = check_answer_svd(M, N, rank, &matU[0], M, &matV[0], K, &matA[0], M);
  double err = std::sqrt(ret.first / ret.second);

  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
  int64_t flops = ((int64_t(M) + int64_t(N)) * int64_t(rank) * int64_t(2)) + (int64_t(M) * int64_t(N) * int64_t(rank) * int64_t(4));
  double gflops = double(flops) * 1.e-6 / double(milliseconds);

  std::cout << prec << "-SVD," << M << "," << N << "," << epi << "," << err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_V);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  cusolverDnDestroyParams(params);
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  char prec = 1 < argc ? *argv[1] : 'D';
  int64_t M = 2 < argc ? std::atoi(argv[2]) : 2048;
  int64_t N = 3 < argc ? std::atoi(argv[3]) : 2048;
  N = std::min(M, N);

  int64_t K = 4 < argc ? std::atoi(argv[4]) : N;
  double epi = 5 < argc ? std::atof(argv[5]) : 1.e-12;
  double omega = 6 < argc ? std::atof(argv[6]) : 1.;
  K = std::min(K, N);

  switch(prec) {
    case 'D': run<double>(prec, M, N, K, epi, omega); break;
    case 'S': run<float>(prec, M, N, K, epi, omega); break;
    case 'Z': run<std::complex<double>>(prec, M, N, K, epi, omega); break;
    case 'C': run<std::complex<float>>(prec, M, N, K, epi, omega); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
