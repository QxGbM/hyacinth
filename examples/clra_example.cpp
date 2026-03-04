
#include <common.hpp>
#include <iostream>

void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, std::complex<float>* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      A[i + j * lda] = std::complex<float>(float(std::cos(w * d) / d), float(std::sin(w * d) / d));
    }
  }
}

int32_t geqp3_ronly(cublasHandle_t handle, double epi, int32_t rank, int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, int32_t* jpiv, std::complex<float>* R, int32_t ldr) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, HYACIN_F32_COMPLEX, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  rank = hyacinXcpqrk(handle, 'R', epi, M, N, rank, p, umax, HYACIN_F32_COMPLEX, A, lda, (int32_t*)piv, HYACIN_F32_COMPLEX, R, ldr, precC, dev_work, pinned_work, alg);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(piv);
  cudaFreeHost(pinned_work);
  return rank;
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

  std::vector<std::complex<float>> matA(M * N);
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

  std::complex<float>* d_A = nullptr, * d_X = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(std::complex<float>));
  cudaMalloc((void**)(&d_X), N * N * sizeof(std::complex<float>));
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  geqp3_ronly(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  std::fill(ipiv.begin(), ipiv.end(), 0);
  cudaMemcpy(d_A, matA.data(), M * N * sizeof(std::complex<float>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t rank = geqp3_ronly(handle, epi, N, M, N, d_A, M, ipiv.data(), d_X, N);
  cudaEventRecord(stop, stream);

  std::vector<std::complex<float>> matX(N * N);
  cudaMemcpy(matX.data(), d_X, N * N * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
  double rel_err = check_answer_clra(rank, M, N, matA.data(), M, ipiv.data(), matX.data(), N);
  
  float milliseconds = 0.0f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // QR flops = 2mnk - nk^2 + 1/3k^3 + k(n-k)
  int64_t qr_flops = (int64_t(M) * int64_t(N) * int64_t(rank) * 2) - (int64_t(N) * int64_t(rank) * int64_t(rank)) + (int64_t(rank) * int64_t(rank) * int64_t(rank) / 3) + (int64_t(rank) * int64_t(N - rank));
  int64_t trsm_flops = int64_t(N) * int64_t(rank) * int64_t(rank);
  double gflops = double(qr_flops + trsm_flops) * 1.e-6 / milliseconds;

  std::cout << "C-LRA," << M << "," << N << "," << epi << "," << rel_err << "," << rank << "," << milliseconds << "," << gflops << std::endl;

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
