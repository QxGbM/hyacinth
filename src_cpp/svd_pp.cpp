
#include <hyacin.h>
#include <cuComplex.h>
#include <tuple>
#include <algorithm>

const double f32_polar_svd_cutoff_epi = 1.e-2;
const double f64_polar_svd_cutoff_epi = 1.e-12; // for epi < cutoff, use gesvd over gesvdp

inline std::tuple<cudaDataType_t, cudaDataType_t, double, size_t, size_t> convert(hyacinPrecision_t type) {
  switch(type) {
    case HYACIN_F64: return std::make_tuple(CUDA_R_64F, CUDA_R_64F, f64_polar_svd_cutoff_epi, sizeof(double), sizeof(double));
    case HYACIN_F32: return std::make_tuple(CUDA_R_32F, CUDA_R_32F, f32_polar_svd_cutoff_epi, sizeof(float), sizeof(float));
    case HYACIN_F64_COMPLEX: return std::make_tuple(CUDA_C_64F, CUDA_R_64F, f64_polar_svd_cutoff_epi, sizeof(cuDoubleComplex), sizeof(double));
    case HYACIN_F32_COMPLEX: return std::make_tuple(CUDA_C_32F, CUDA_R_32F, f32_polar_svd_cutoff_epi, sizeof(cuComplex), sizeof(float));
    default: return std::make_tuple(cudaDataType_t(0), cudaDataType_t(0), 0., size_t(0), size_t(0));
  }
}

extern "C" void hyacinXutvk_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, hyacinPrecision_t ComputeType, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
  if (N <= 0 || K <= 0) { *dev_work_bytes = *pinned_work_bytes = uint64_t(0); return; }

  std::tuple<cudaDataType_t, cudaDataType_t, size_t, size_t, double> type = convert(ComputeType);
  cudaDataType_t type_c = std::get<0>(type), type_r = std::get<1>(type);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  size_t workspaceInBytesOnDevice, workspaceInBytesOnHost; uint64_t basis_bytes;

  if (epi < std::get<2>(type)) {
    cusolverDnXgesvd_bufferSize(handle, params, 'O', 'N', N, K,
      type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * uint64_t(algnN) * uint64_t(K);
  }
  else {
    cusolverDnXgesvdp_bufferSize(handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      type_c, nullptr, algnN, type_r, nullptr, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, &workspaceInBytesOnDevice, &workspaceInBytesOnHost);
    basis_bytes = uint64_t(std::get<3>(type)) * (uint64_t(algnN) + uint64_t(algnN) + uint64_t(algnK)) * uint64_t(K);
  }

  uint64_t s_bytes = uint64_t(std::get<4>(type)) * uint64_t(algnK);
  uint64_t trans_bytes = uint64_t(std::get<3>(type)) * uint64_t(algnN + 16384) * uint64_t(K);
  *dev_work_bytes = std::max(uint64_t(workspaceInBytesOnDevice) + basis_bytes + s_bytes, trans_bytes);
  *pinned_work_bytes = uint64_t(workspaceInBytesOnHost) + s_bytes;
}

template <class T> inline void constants(cudaDataType_t&, cudaDataType_t&, double&);
template <> inline void constants<double>(cudaDataType_t& type_c, cudaDataType_t& type_r, double& epi)
{ type_c = type_r = CUDA_R_64F; epi = f64_polar_svd_cutoff_epi; }
template <> inline void constants<float>(cudaDataType_t& type_c, cudaDataType_t& type_r, double& epi)
{ type_c = type_r = CUDA_R_32F; epi = f32_polar_svd_cutoff_epi; }
template <> inline void constants<cuDoubleComplex>(cudaDataType_t& type_c, cudaDataType_t& type_r, double& epi)
{ type_c = CUDA_C_64F; type_r = CUDA_R_64F; epi = f64_polar_svd_cutoff_epi; }
template <> inline void constants<cuComplex>(cudaDataType_t& type_c, cudaDataType_t& type_r, double& epi)
{ type_c = CUDA_C_32F; type_r = CUDA_R_32F; epi = f32_polar_svd_cutoff_epi; }

inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const double* A, int32_t lda, double* B, int32_t ldb)
{ double one = 1., zero = 0.; cublasDgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const float* A, int32_t lda, float* B, int32_t ldb)
{ float one = 1.f, zero = 0.f; cublasSgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const cuDoubleComplex* A, int32_t lda, cuDoubleComplex* B, int32_t ldb)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const cuComplex* A, int32_t lda, cuComplex* B, int32_t ldb)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, A, lda, &zero, B, ldb, B, ldb); }

inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ double one = 1., zero = 0.; cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ float one = 1.f, zero = 0.f; cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuDoubleComplex* A, int32_t lda, const cuDoubleComplex* B, int32_t ldb, cuDoubleComplex* C, int32_t ldc)
{ cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.); cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }
inline void nn_gemm(cublasHandle_t handle, int32_t M, int32_t N, int32_t K, const cuComplex* A, int32_t lda, const cuComplex* B, int32_t ldb, cuComplex* C, int32_t ldc)
{ cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f); cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &one, A, lda, B, ldb, &zero, C, ldc); }

template <class real_t, class complex_t>
inline int32_t utv_k_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char transform, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  real_t* sigma, complex_t* UA, int32_t ldu, complex_t* RJ, int32_t ldr, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(real_t) * int64_t(algnK), r_bytes = sizeof(complex_t) * int64_t(algnN) * int64_t(K);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];

  cudaDataType_t type_c, type_r; double cutoff_epi;
  constants<complex_t>(type_c, type_r, cutoff_epi);

  if (epi < cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes));

    tranpose_copy(handle, 'C', N, K, RJ, ldr, (complex_t*)R, algnN);
    cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K,
      type_c, R, algnN, type_r, S, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, nullptr);
  }
  else {
    int64_t v_bytes = sizeof(complex_t) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + r_bytes + v_bytes + s_bytes));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *V = &U[r_bytes];

    double h_err_sigma = 0.;
    tranpose_copy(handle, 'C', N, K, RJ, ldr, (complex_t*)U, algnN);
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      type_c, U, algnN, type_r, S, type_c, R, algnN, type_c, V, algnK, type_c, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, nullptr, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  if (sigma) cudaMemcpyAsync(sigma, S, sizeof(real_t) * int64_t(K), cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
  real_t *Svec = (real_t*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return s < s0; }))));

  if (0 < K) {
    tranpose_copy(handle, 'C', K, N, (complex_t*)R, algnN, RJ, ldr);
    if ((transform == 'Y' || transform == 'y')) {
      int32_t rows = int32_t((dev_work_bytes - r_bytes) / (uint64_t(K) * sizeof(complex_t))) & (~63);
      for (int32_t i = 0; i < M; i += rows) {
        int32_t m = std::min(M - i, rows), lds = std::min((m + 63) & (~63), rows);
        nn_gemm(handle, m, K, N, &UA[i], ldu, (complex_t*)R, algnN, (complex_t*)S, lds);
        tranpose_copy(handle, 'N', m, K, (complex_t*)S, lds, &UA[i], ldu);
      }
    }
  }
  return K;
}

extern "C" int32_t hyacinXutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, char transform, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  void* sigma, void* UA, int32_t ldu, void* RJ, int32_t ldr, hyacinPrecision_t ComputeType, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  if (0 < N && 0 < K) switch(ComputeType) {
    case HYACIN_F64:
      return utv_k_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (double*)sigma, (double*)UA, ldu, (double*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32:
      return utv_k_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (float*)sigma, (float*)UA, ldu, (float*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F64_COMPLEX:
      return utv_k_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (double*)sigma, (cuDoubleComplex*)UA, ldu, (cuDoubleComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32_COMPLEX:
      return utv_k_dispatcher(handle, s_handle, params, transform, epi, M, N, K, p, (float*)sigma, (cuComplex*)UA, ldu, (cuComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    default: return 0;
  }
  return 0;
}
