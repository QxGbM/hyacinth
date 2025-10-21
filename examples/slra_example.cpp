
#include <hyacin.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

void make_2D_oscillatory(double w, int32_t sep, int32_t M, int32_t N, float* A, int32_t lda) {
  constexpr int32_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };
  sep = height * sep + ((M + height - 1) & (~(height - 1)));

  for (int32_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + sep);
    for (int32_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i);
      double d = std::abs(vi - vj);
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = float(std::cos(w * d) / d);
    }
  }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int64_t M = 1 < argc ? std::atoi(argv[1]) : 2048;
  int64_t N = 2 < argc ? std::atoi(argv[2]) : 2048;
  N = std::min(M, N);

  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-6;
  double omega = 4 < argc ? std::atof(argv[4]) : 1.;
  int32_t sep = 5 < argc ? std::atoi(argv[5]) : 0;

  std::vector<float> matA(M * N);
  std::vector<int32_t> ipiv(N);
  make_2D_oscillatory(omega, sep, M, N, matA.data(), M);

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  float* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(float));
  cudaMalloc((void**)(&d_X), N * N * sizeof(float));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  device::interp_decomp_f32(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = device::interp_decomp_f32(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  double rel_err = 0.;
  device::check_interp_decomp_f32(handle, rank, M, N, d_A, M, ipiv.data(), d_X, N, &rel_err);

  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // QR flops = 2mnk - nk^2 + 1/3k^3 + k(n-k)
  int64_t qr_flops = (int64_t(M) * int64_t(N) * int64_t(rank) * 2) - (int64_t(N) * int64_t(rank) * int64_t(rank)) + (int64_t(rank) * int64_t(rank) * int64_t(rank) / 3) + (int64_t(rank) * int64_t(N - rank));
  int64_t trsm_flops = int64_t(N) * int64_t(rank) * int64_t(rank);
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / milliseconds;

  std::cout << "S-LRA," << M << "," << N << "," << epi << "," << omega << "," << sep << "," << rel_err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

  cudaFree(d_A);
  cudaFree(d_X);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);

  cu_err = cudaGetLastError();
  if (cu_err != cudaSuccess)
    std::cerr << cudaGetErrorString(cu_err) << std::endl;
  return 0;
}
