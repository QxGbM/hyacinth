
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
    default: return std::make_tuple(cudaDataType_t(-1), cudaDataType_t(-1), 0., size_t(0), size_t(0));
  }
}

extern "C" void hyacinXutvk_bufferSize(cusolverDnHandle_t handle, cusolverDnParams_t params, double epi, int32_t N, int32_t K, hyacinPrecision_t ComputeType, uint64_t* dev_work_bytes, uint64_t* pinned_work_bytes) {
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
  *dev_work_bytes = uint64_t(workspaceInBytesOnDevice) + basis_bytes + s_bytes;
  *pinned_work_bytes = uint64_t(workspaceInBytesOnHost) + s_bytes;
}

template <hyacinPrecision_t prec>
inline void tranpose_copy(cublasHandle_t handle, char trans, int32_t Mb, int32_t Nb, const void* A, int32_t lda, void* B, int32_t ldb) {
  if constexpr(prec == HYACIN_F64)
  { double one = 1., zero = 0.; cublasDgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, (const double*)A, lda, &zero, (double*)B, ldb, (double*)B, ldb); }
  else if constexpr(prec == HYACIN_F32)
  { float one = 1.f, zero = 0.f; cublasSgeam(handle, trans == 'C' ? CUBLAS_OP_T : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, (const float*)A, lda, &zero, (float*)B, ldb, (float*)B, ldb); }
  else if constexpr(prec == HYACIN_F64_COMPLEX) {
    cuDoubleComplex one = make_cuDoubleComplex(1., 0.), zero = make_cuDoubleComplex(0., 0.);
    cublasZgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, (const cuDoubleComplex*)A, lda, &zero, (cuDoubleComplex*)B, ldb, (cuDoubleComplex*)B, ldb);
  }
  else if constexpr(prec == HYACIN_F32_COMPLEX) {
    cuComplex one = make_cuComplex(1.f, 0.f), zero = make_cuComplex(0.f, 0.f);
    cublasCgeam(handle, trans == 'C' ? CUBLAS_OP_C : CUBLAS_OP_N, CUBLAS_OP_N, Mb, Nb, &one, (const cuComplex*)A, lda, &zero, (cuComplex*)B, ldb, (cuComplex*)B, ldb);
  }
}

template <hyacinPrecision_t prec, class complex_t>
inline void constants(cudaDataType_t& type_c, cudaDataType_t& type_r, complex_t& one, complex_t& zero, double& epi) {
  if constexpr(prec == HYACIN_F64)
  { type_c = type_r = CUDA_R_64F; one = 1.; zero = 0.; epi = f64_polar_svd_cutoff_epi; }
  else if constexpr(prec == HYACIN_F32)
  { type_c = type_r = CUDA_R_32F; one = 1.f; zero = 0.f; epi = f32_polar_svd_cutoff_epi; }
  else if constexpr(prec == HYACIN_F64_COMPLEX)
  { type_c = CUDA_C_64F; type_r = CUDA_R_64F; one = make_cuDoubleComplex(1., 0.); zero = make_cuDoubleComplex(0., 0.); epi = f64_polar_svd_cutoff_epi; }
  else if constexpr(prec == HYACIN_F32_COMPLEX)
  { type_c = CUDA_C_32F; type_r = CUDA_R_32F; one = make_cuComplex(1.f, 0.f); zero = make_cuComplex(0.f, 0.f); epi = f32_polar_svd_cutoff_epi; }
}

template <hyacinPrecision_t prec, class real_t, class complex_t>
inline int32_t utv_k_dispatcher(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  complex_t* UA, int32_t ldu, complex_t* RJ, int32_t ldr, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {
  
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t algnK = (K + 63) & (~63), algnN = (N + 63) & (~63);
  int64_t s_bytes = sizeof(real_t) * int64_t(algnK), r_bytes = sizeof(complex_t) * int64_t(algnN) * int64_t(K);
  size_t workspaceInBytesOnHost = size_t(pinned_work_bytes - uint64_t(s_bytes));
  int8_t* R = (int8_t*)dev_work, *S = &R[r_bytes], *bufferOnDevice = &S[s_bytes];
  int8_t* Sh = (int8_t*)pinned_work, *bufferOnHost = &Sh[s_bytes];

  cudaDataType_t type_c, type_r; complex_t one, zero; double cutoff_epi;
  constants<prec, complex_t>(type_c, type_r, one, zero, cutoff_epi);

  if (epi < cutoff_epi) {
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + s_bytes));

    tranpose_copy<prec>(handle, 'C', N, K, RJ, ldr, R, algnN);
    cusolverDnXgesvd(s_handle, params, 'O', 'N', N, K,
      type_c, R, algnN, type_r, S, type_c, nullptr, algnN, type_c, nullptr, algnK, type_c, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, nullptr);
  }
  else {
    int64_t v_bytes = sizeof(complex_t) * int64_t(algnK) * int64_t(K);
    size_t workspaceInBytesOnDevice = size_t(dev_work_bytes - uint64_t(r_bytes + r_bytes + v_bytes + s_bytes));
    int8_t* U = &bufferOnDevice[workspaceInBytesOnDevice], *V = &U[r_bytes];

    double h_err_sigma = 0.;
    tranpose_copy<prec>(handle, 'C', N, K, RJ, ldr, U, algnN);
    cusolverDnXgesvdp(s_handle, params, CUSOLVER_EIG_MODE_VECTOR, 1, N, K,
      type_c, U, algnN, type_r, S, type_c, R, algnN, type_c, V, algnK, type_c, bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, nullptr, &h_err_sigma);
  }
  
  cudaMemcpyAsync(Sh, S, s_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  real_t *Svec = (real_t*)Sh, s0 = epi * Svec[0];
  K = std::min(K, p + int32_t(std::distance(Svec, std::find_if(Svec, &Svec[K], [=](real_t s) { return s < s0; }))));

  if (0 < K) {
    int32_t rows = ((dev_work_bytes - r_bytes) / uint64_t(K)) & (~63);
    tranpose_copy<prec>(handle, 'C', K, N, R, algnN, RJ, ldr);

    for (int32_t i = 0; i < M; i += rows) {
      int32_t m = std::min(M - i, rows), lds = std::min((m + 63) & (~63), rows);
      cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, K, N, &one, &UA[i], type_c, ldu, R, type_c, algnN, &zero, S, type_c, lds, type_c, CUBLAS_GEMM_DEFAULT);
      tranpose_copy<prec>(handle, 'N', m, K, S, lds, &UA[i], ldu);
    }
  }
  return K;
}

extern "C" int32_t hyacinXutvk(cublasHandle_t handle, cusolverDnHandle_t s_handle, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, int32_t p,
  void* UA, int32_t ldu, void* RJ, int32_t ldr, hyacinPrecision_t ComputeType, uint64_t dev_work_bytes, void* dev_work, uint64_t pinned_work_bytes, void* pinned_work) {

  switch(ComputeType) {
    case HYACIN_F64:
      return utv_k_dispatcher<HYACIN_F64, double, double>(handle, s_handle, params, epi, M, N, K, p, (double*)UA, ldu, (double*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32:
      return utv_k_dispatcher<HYACIN_F32, float, float>(handle, s_handle, params, epi, M, N, K, p, (float*)UA, ldu, (float*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F64_COMPLEX:
      return utv_k_dispatcher<HYACIN_F64_COMPLEX, double, cuDoubleComplex>(handle, s_handle, params, epi, M, N, K, p, (cuDoubleComplex*)UA, ldu, (cuDoubleComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    case HYACIN_F32_COMPLEX:
      return utv_k_dispatcher<HYACIN_F32_COMPLEX, float, cuComplex>(handle, s_handle, params, epi, M, N, K, p, (cuComplex*)UA, ldu, (cuComplex*)RJ, ldr, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);
    default: return 0;
  }
}
