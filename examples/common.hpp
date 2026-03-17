
#include <hyacin.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string>
#include <mpi.h>

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#include <lapacke.h>
#endif

std::string replace_suffix(const std::string& str, const std::string& suffix) {
  std::string result = str;
  std::size_t pos = result.rfind('.');
  if (pos == std::string::npos) { result += '.' + suffix; }
  else { result.replace(pos + 1, std::string::npos, suffix); }
  return result;
}

template <class T>
inline void copy2d(int32_t M, int32_t N, const T* A, int32_t lda, T* B, int32_t ldb)
{ for (int32_t j = 0; j < N; ++j) { std::copy_n(&A[int64_t(j) * int64_t(lda)], M, &B[int64_t(j) * int64_t(ldb)]); } }

inline void parse_char(double& a, const std::string& s) { a = std::stod(s); }
inline void parse_char(float& a, const std::string& s) { a = std::stof(s); }
inline void parse_char(std::complex<double>& a, const std::string& s) {
  std::string::size_type l; double rl = std::stod(s, &l);
  try { double im = std::stod(s.substr(l)); a = std::complex<double>(rl, im); }
  catch (const std::invalid_argument&) { a = std::complex<double>(rl, 0.); }
}
inline void parse_char(std::complex<float>& a, const std::string& s) {
  std::string::size_type l; float rl = std::stod(s, &l);
  try { float im = std::stof(s.substr(l)); a = std::complex<float>(rl, im); }
  catch (const std::invalid_argument&) { a = std::complex<float>(rl, 0.f); }
}

template <class T>
void matrix_from_row_major_csv(int32_t M, int32_t N, int32_t mb, int32_t nb, T* A, int32_t lda, const std::string& file, MPI_Comm comm_row, MPI_Comm comm_col) {
  int32_t grid_row = 0, grid_col = 0, tile_m = 1, tile_n = 1;
  MPI_Comm_rank(comm_row, &grid_col); MPI_Comm_size(comm_row, &tile_n);
  MPI_Comm_rank(comm_col, &grid_row); MPI_Comm_size(comm_col, &tile_m);
  std::ifstream csv; std::vector<int64_t> row_bytes;
  if (grid_col == 0) {
    csv.open(replace_suffix(file, "csv"));
    if (grid_row == 0) {
      std::ifstream idx(replace_suffix(file, "cache"));
      if (idx.is_open()) { while (!idx.eof()) { int64_t i; idx >> i; row_bytes.push_back(i); } idx.close(); }
      else {
        int64_t len = 0; std::string line; row_bytes.push_back(int64_t(0));
        while(std::getline(csv, line)) row_bytes.push_back(len += 1 + line.length());
        std::ofstream idx_cache(replace_suffix(file, "cache"));
        for (int64_t i : row_bytes) idx_cache << i << std::endl;
        idx_cache.close(); csv.clear();
      }
    }
    
    if (1 < tile_m)
    { row_bytes.resize(M + 1); MPI_Bcast(&row_bytes[0], M + 1, MPI_INT64_T, 0, comm_col); }
  }

  for (int32_t i = grid_row * mb, y = 0; i < M; i = grid_row * mb + tile_m * (y += mb)) {
    int32_t ib = std::min(M - i, mb);
    std::vector<T> mat(int64_t(ib) * int64_t(N));

    if (grid_col == 0 && csv.is_open()) {
      int64_t start_byte = row_bytes[i], n_bytes = row_bytes[i + ib] - start_byte;
      std::vector<char> buf(n_bytes); char* str = &buf[0], *end = &buf[n_bytes];
      csv.seekg(start_byte, std::ios::beg); csv.read(str, n_bytes);
      auto cmp = [](char c){ return c == ' ' || c == ',' || c == '\n' || c == '(' || c == ')'; };
      while(cmp(*str)) ++str;

      for (int32_t y = 0; y < ib; ++y)
        for (int32_t x = 0; x < N; ++x) {
          std::string Ayx(str, std::distance(str, std::find_if(str, end, cmp)));
          parse_char(mat[int64_t(y) + int64_t(x) * int64_t(ib)], Ayx);
          str += Ayx.length(); while(cmp(*str)) ++str;
        }
    }

    if (1 < tile_n)
      MPI_Bcast(&mat[0], mat.size() * sizeof(T), MPI_BYTE, 0, comm_row);

    for (int32_t j = grid_col * nb, x = 0; j < N; j = grid_col * nb + tile_n * (x += nb))
      copy2d(ib, std::min(N - j, nb), &mat[int64_t(j) * int64_t(ib)], ib, &A[int64_t(y) + int64_t(x) * int64_t(lda)], lda);
  }

  if (grid_col == 0 && csv.is_open())
    csv.close();
}

inline void __f(double& a, double d, double w) { a = std::cos(w * d) / d; }
inline void __f(float& a, double d, double w) { a = float(std::cos(w * d) / d); }
inline void __f(std::complex<double>& a, double d, double w) { a = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d); }
inline void __f(std::complex<float>& a, double d, double w) { a = std::complex<float>(float(std::cos(w * d) / d), float(std::sin(w * d) / d)); }

template <class T>
void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, T* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      __f(A[i + j * lda], d, w);
    }
  }
}

inline void __lrtrsm(int32_t M, int32_t N, const double* A, int32_t lda, double* B, int32_t ldb)
{ cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, 1., A, lda, B, ldb); }
inline void __lrtrsm(int32_t M, int32_t N, const float* A, int32_t lda, float* B, int32_t ldb)
{ cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, 1.f, A, lda, B, ldb); }
inline void __lrtrsm(int32_t M, int32_t N, const std::complex<double>* A, int32_t lda, std::complex<double>* B, int32_t ldb)
{ std::complex<double> one(1., 0.); cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, A, lda, B, ldb); }
inline void __lrtrsm(int32_t M, int32_t N, const std::complex<float>* A, int32_t lda, std::complex<float>* B, int32_t ldb)
{ std::complex<float> one(1.f, 0.f); cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit, M, N, &one, A, lda, B, ldb); }

inline void __nngemm(int32_t M, int32_t N, int32_t K, const double* A, int32_t lda, const double* B, int32_t ldb, double* C, int32_t ldc)
{ cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1., A, lda, B, ldb, -1., C, ldc); }
inline void __nngemm(int32_t M, int32_t N, int32_t K, const float* A, int32_t lda, const float* B, int32_t ldb, float* C, int32_t ldc)
{ cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.f, A, lda, B, ldb, -1.f, C, ldc); }
inline void __nngemm(int32_t M, int32_t N, int32_t K, const std::complex<double>* A, int32_t lda, const std::complex<double>* B, int32_t ldb, std::complex<double>* C, int32_t ldc)
{ std::complex<double> one(1., 0.), minus_one(-1., 0.); cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one, A, lda, B, ldb, &minus_one, C, ldc); }
inline void __nngemm(int32_t M, int32_t N, int32_t K, const std::complex<float>* A, int32_t lda, const std::complex<float>* B, int32_t ldb, std::complex<float>* C, int32_t ldc)
{ std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f); cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, &one, A, lda, B, ldb, &minus_one, C, ldc); }

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<T> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(jpiv[i] - 1) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  __lrtrsm(rank, N, R, ldr, &matR[0], rank);
  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  __nngemm(M, N, rank, &matC[0], M, &matR[0], rank, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::sqrt(err / nrm);
}

template <class T>
std::pair<double, double> check_answer_svd(int32_t M, int32_t N, int32_t rank, const T* U, int32_t ldu, const T* V, int32_t ldv, const T* B, int32_t ldb) {
  if (rank <= 0 || M <= 0 || N <= 0)
    return std::make_pair(0., 0.);
  std::vector<T> matB(M * N);
  copy2d(M, N, B, ldb, &matB[0], M);

  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  __nngemm(M, N, rank, U, ldu, V, ldv, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}

template <class T>
std::pair<double, double> check_answer_svd(int32_t M, int32_t N, int32_t r1, int32_t r2, const T* U, int32_t ldu, const T* V1, int32_t ldv1, const T* V2, int32_t ldv2, const T* B, int32_t ldb) {
  if (r2 <= 0)
    return std::make_pair(0., 0.);
  std::vector<T> matV(r1 * N, T());
  __nngemm(r1, N, r2, V1, ldv1, V2, ldv2, &matV[0], r1);
  return check_answer_svd(M, N, r1, U, ldu, &matV[0], r1, B, ldb);
}

template <class T> struct __precA;
template <> struct __precA<double> { static constexpr hyacinPrecision_t value = HYACIN_F64; };
template <> struct __precA<float> { static constexpr hyacinPrecision_t value = HYACIN_F32; };
template <> struct __precA<std::complex<double>> { static constexpr hyacinPrecision_t value = HYACIN_F64_COMPLEX; };
template <> struct __precA<std::complex<float>> { static constexpr hyacinPrecision_t value = HYACIN_F32_COMPLEX; };

template <class T>
int32_t geqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpiv, T* R, int32_t ldr) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precA = __precA<T>::value, precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, precA, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *piv = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0;
  int32_t rank = hyacinXcpqrk(handle, 'R', epi, M, N, K, p, umax, precA, A, lda, (int32_t*)piv, precA, R, ldr, precC, dev_work, pinned_work, alg);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(dev_work);
  cudaFree(piv);
  cudaFreeHost(pinned_work);
  return rank;
}

template <class T>
int32_t utv_factorize_phase1(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv) {
  int32_t umax; hyacinPrecision_t precA = __precA<T>::value, precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, precA, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, p, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg);

  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

template <class T>
int32_t utv_factorize_phase1_1dr(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t gM, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv, ncclComm_t comm) {
  int32_t umax; hyacinPrecision_t precA = __precA<T>::value, precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXcpqrk_autoTune(epi, gM, 6, &umax, precA, &precC, &alg);
  hyacinXcpqrk1Drow_bufferSize(M, gM, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk1Drow(cublasH, 'J', epi, M, gM, N, K, p, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg, comm);

  uint64_t dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

template <class T>
std::pair<int32_t, int32_t> utv_factorize_phase2_1dc(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv, ncclComm_t comm, MPI_Comm mpi_comm) {
  int32_t mpi_rank, mpi_size; MPI_Comm_rank(mpi_comm, &mpi_rank); MPI_Comm_size(mpi_comm, &mpi_size);
  std::vector<int32_t> ranks_arr(mpi_size);
  MPI_Allgather(&N, 1, MPI_INT32_T, ranks_arr.data(), 1, MPI_INT32_T, mpi_comm);
  int32_t rank_max = std::reduce(ranks_arr.begin(), ranks_arr.end(), 0, [](int32_t i, int32_t j) { return std::max(i, j); });
  int32_t v_offset = std::reduce(ranks_arr.begin(), ranks_arr.begin() + mpi_rank, 0, std::plus<int32_t>());
  N = std::reduce(ranks_arr.begin() + mpi_rank, ranks_arr.end(), v_offset, std::plus<int32_t>());

  int32_t umax; hyacinPrecision_t precA = __precA<T>::value, precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXcpqrk_autoTune(epi, M, 6, &umax, precA, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);
  hyacinXAllGatherV1Dcol_bufferSize(M, rank_max, mpi_size, precA, &dev_work_bytes_new);
  dev_work_bytes = std::max(dev_work_bytes, dev_work_bytes_new);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  hyacinXAllGatherV1Dcol(cublasH, M, mpi_rank, mpi_size, ranks_arr.data(), precA, A, lda, dev_work, comm);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, p, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg);

  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return std::make_pair(rank, v_offset);
}

template <class T>
std::pair<int32_t, int32_t> utv_factorize_phase2_2d(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t gM, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv, ncclComm_t comm_row, ncclComm_t comm_col, MPI_Comm mpi_comm_row) {
  int32_t mpi_rank, mpi_size; MPI_Comm_rank(mpi_comm_row, &mpi_rank); MPI_Comm_size(mpi_comm_row, &mpi_size);
  std::vector<int32_t> ranks_arr(mpi_size);
  MPI_Allgather(&N, 1, MPI_INT32_T, ranks_arr.data(), 1, MPI_INT32_T, mpi_comm_row);
  int32_t rank_max = std::reduce(ranks_arr.begin(), ranks_arr.end(), 0, [](int32_t i, int32_t j) { return std::max(i, j); });
  int32_t v_offset = std::reduce(ranks_arr.begin(), ranks_arr.begin() + mpi_rank, 0, std::plus<int32_t>());
  N = std::reduce(ranks_arr.begin() + mpi_rank, ranks_arr.end(), v_offset, std::plus<int32_t>());

  int32_t umax; hyacinPrecision_t precA = __precA<T>::value, precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXcpqrk_autoTune(epi, gM, 6, &umax, precA, &precC, &alg);
  hyacinXcpqrk1Drow_bufferSize(M, gM, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);
  hyacinXAllGatherV1Dcol_bufferSize(M, rank_max, mpi_size, precA, &dev_work_bytes_new);
  dev_work_bytes = std::max(dev_work_bytes, dev_work_bytes_new);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  hyacinXAllGatherV1Dcol(cublasH, M, mpi_rank, mpi_size, ranks_arr.data(), precA, A, lda, dev_work, comm_row);

  int32_t p = 0; 
  int32_t rank = hyacinXcpqrk1Drow(cublasH, 'J', epi, M, gM, N, K, p, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg, comm_col);

  hyacinXutvk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXutvk(cublasH, cusolverH, params, epi, M, N, rank, p, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return std::make_pair(rank, v_offset);
}
