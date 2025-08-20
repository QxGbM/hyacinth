
#include <hyacinth.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  auto translate_2d = [](int64_t i) { int64_t x = i / 128, y = i - 128 * x; return std::complex<double>(x, y); };
  sep = 128 * sep + ((M + 127) & (~127));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 128;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  double omega = 4 < argc ? std::atof(argv[4]) : 0.;
  int32_t sep = 5 < argc ? std::atoi(argv[5]) : 0;

  std::vector<std::complex<double>> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, sep, M, N, matA.data(), M);

  std::cout << "CF64 LR-APPROX <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << ", Omega: " << omega << ", Sep: " << sep << "\n";

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::complex<double>* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(std::complex<double>));
  cudaMalloc((void**)(&d_X), N * N * sizeof(std::complex<double>));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = device::interp_decomp_cf64(stream, handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  double rel_err = 0.;
  device::check_interp_decomp_cf64(stream, handle, rank, M, N, d_A, M, ipiv.data(), d_X, N, &rel_err);
  std::cout << "Rel Err: " << rel_err << std::endl;

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
