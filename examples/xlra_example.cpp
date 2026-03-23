
#include <common.hpp>
#include <iostream>

template <class T> inline void run(char prec, int64_t M, int64_t N, double epi, double omega) {
  std::vector<T> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, 0, 0, M, N, &matA[0], M);

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  T* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(T));
  cudaMalloc((void**)(&d_X), N * N * sizeof(T));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  geqp3_ronly(handle, epi, M, N, N, d_A, M, ipiv.data(), d_X, N);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(T), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = geqp3_ronly(handle, epi, M, N, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  std::vector<T> matX(N * N);
  cudaMemcpy(matX.data(), d_X, N * N * sizeof(T), cudaMemcpyDeviceToHost);
  double rel_err = check_answer_lra(rank, M, N, matA.data(), M, ipiv.data(), matX.data(), N);

  float milliseconds = 0.0f; cudaEventElapsedTime(&milliseconds, start, stop);
  // QR flops = 2mnk - nk^2 + 1/3k^3 + k(n-k)
  int64_t qr_flops = (int64_t(M) * int64_t(N) * int64_t(rank) * 2) - (int64_t(N) * int64_t(rank) * int64_t(rank)) + (int64_t(rank) * int64_t(rank) * int64_t(rank) / 3) + (int64_t(rank) * int64_t(N - rank));
  int64_t trsm_flops = int64_t(N) * int64_t(rank) * int64_t(rank);
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / double(milliseconds);

  printf("%c-LRA,%ld,%ld,%.1le,%.12le,%d,%f,%lf\n", prec, M, N, epi, rel_err, rank, milliseconds, gflops);

  cudaFree(d_A);
  cudaFree(d_X);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
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

  double epi = 4 < argc ? std::atof(argv[4]) : 1.e-12;
  double omega = 5 < argc ? std::atof(argv[5]) : 1.;

  switch(prec) {
    case 'D': run<double>(prec, M, N, epi, omega); break;
    case 'S': run<float>(prec, M, N, epi, omega); break;
    case 'Z': run<std::complex<double>>(prec, M, N, epi, omega); break;
    case 'C': run<std::complex<float>>(prec, M, N, epi, omega); break;
    default: break;
  }

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
