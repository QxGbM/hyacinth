
#include <hyacinth.hpp>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <vector>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

void make_1D_oscilatory(double w, int32_t M, int32_t N, std::complex<double>* A, int32_t lda) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) {
      double d = std::abs(i - (j + M));
      A[uint64_t(i) + uint64_t(j) * uint64_t(lda)] = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d);
    }
}

int32_t main(int32_t argc, char* argv[]) {
  auto cu_err = cudaSetDevice(0);
  cudaDeviceReset();
  if (cu_err != cudaSuccess)
  { std::cerr << cudaGetErrorString(cu_err) << std::endl; return -1; }

  int32_t M = 1 < argc ? std::atoi(argv[1]) : 1024;
  int32_t N = std::min(M, 2 < argc ? std::atoi(argv[2]) : 128);
  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  std::vector<std::complex<double>> matA(M * N);
  std::vector<int32_t> ipiv(N);

  make_1D_oscilatory(200, M, N, matA.data(), M);

  std::cout << "CF64 LR-APPROX <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << "\n";

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

  cudaDeviceSynchronize();

  if (M <= 2048 && N <= 2048) {
    std::vector<std::complex<double>> matB(M * N), matC(M * rank), matX(N * N);
    cudaMemcpy(matX.data(), d_X, N * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    for (int32_t i = 0; i < rank; ++i) {
      int32_t col = (ipiv[i] - 1);
      std::copy_n(&matA[uint64_t(col) * uint64_t(M)], M, &matC[uint64_t(i) * uint64_t(M)]);
    }
    std::complex<double> zero(0., 0.), one(1., 0.);
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, rank, &one, &matC[0], M, &matX[0], N, &zero, &matB[0], M);
  
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
