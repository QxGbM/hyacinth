
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
  double epi = 3 < argc ? std::atof(argv[3]) : 1.e-12;
  std::vector<std::complex<double>> matA(M * N);
  int32_t* ipiv;

  std::mt19937_64 gen;
  std::normal_distribution<double> dist(0, 32);
  std::generate(matA.begin(), matA.end(), [&](){ return std::complex<double>(dist(gen), dist(gen)); });

  device::QR::geqp3_params params;
  device::QR::zgeqp3_ronly_params_query(&params, epi, M, N);
  //device::QR::set_quad_float_as_fp128(&params);

  std::cout << "ZGEQP3 <" << M << ", " << N << ">\n";
  std::cout << "Epi: " << epi << "\n";
  std::cout << "acc-bits: " << params.acc_bits << "\n";
  std::cout << "orderA: " << params.orderA << "\n";
  std::cout << "orderC: " << params.orderC << "\n";
  std::cout << "IGEMM k=" << params.iter_k << "\n";
  std::cout << "i8 bytes: " << params.n_i8 << "\n";
  std::cout << "i32 bytes: " << 4*params.n_i32 << "\n";
  std::cout << "Matrix elements: " << params.n_elem << "\n";
  std::cout << "Matrix element bytes: " << params.elem_bytes << "\n";
  std::cout << "Total work bytes: " << params.work_bytes << "\n" << std::endl;

  cudaStream_t stream;
  cublasHandle_t handle;
  cudaStreamCreate(&stream);
  cublasCreate(&handle);
  cublasSetStream(handle, stream);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::complex<double>* d_A = nullptr;
  void* work = nullptr;
  cudaMalloc((void**)(&d_A), M * N * sizeof(std::complex<double>));
  cudaMalloc(&work, params.work_bytes);
  cudaMallocHost((void**)(&ipiv), (N + 8) * sizeof(int32_t));

  cudaMemcpy(d_A, matA.data(), M * N * sizeof(std::complex<double>), cudaMemcpyHostToDevice);

  cudaEventRecord(start, stream);
  int32_t ret = device::QR::zgeqp3_ronly(stream, handle, params, d_A, M, ipiv, work);
  cudaEventRecord(stop, stream);

  cudaDeviceSynchronize();

  if (M <= 2048 && N <= 2048) {
    std::vector<std::complex<double>> matB(M * N);
    cudaMemcpy(matB.data(), d_A, M * N * sizeof(std::complex<double>), cudaMemcpyDeviceToHost);

    std::vector<int32_t> jpiv(N, 0);
    std::vector<std::complex<double>> tau(N);
    LAPACKE_zgeqp3(LAPACK_COL_MAJOR, M, N, (lapack_complex_double*)matA.data(), M, jpiv.data(), (lapack_complex_double*)tau.data());

    int32_t err_int = 0;
    for (int32_t i = 0; i < N; ++i) {
      err_int += std::abs(jpiv[i] - ipiv[i]);
      if (matA[i * (M + 1)].real() < 0.)
        cblas_zdscal(N, -1., &(matA.data())[i], M);
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
  cudaFreeHost(ipiv);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cublasDestroy(handle);
  std::cerr << cudaGetErrorString(cudaGetLastError()) << std::endl;
  return 0;
}
