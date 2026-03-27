#pragma once

#include <hyacin.h>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>

using blas_int = int; // LP64 for blas
const int32_t oversampling = 10; // increase for better LRA accuracy
const int32_t u_extra = 6; // increase for better Quantization accuracy
const int32_t usd_nd_allreduce = 0; // non-zero for use float(non-deterministic) all reduce when possible, 0 for deterministic

template <class T>
inline void copy2d(int32_t M, int32_t N, const T* A, int32_t lda, T* B, int32_t ldb)
{ for (int32_t j = 0; j < N; ++j) { std::copy_n(&A[int64_t(j) * int64_t(lda)], M, &B[int64_t(j) * int64_t(ldb)]); } }

std::string replace_suffix(const std::string& str, const std::string& suffix) {
  std::string result = str;
  std::size_t pos = result.rfind('.');
  if (pos == std::string::npos) { result += '.' + suffix; }
  else { result.replace(pos + 1, std::string::npos, suffix); }
  return result;
}

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
void matrix_from_row_major_csv(int32_t M, int32_t N, int32_t mb, int32_t nb, T* A, int32_t lda, const std::string& file, int32_t grid_row = 0, int32_t grid_col = 0, int32_t tile_m = 1, int32_t tile_n = 1) {
  std::ifstream csv(replace_suffix(file, "csv")), idx(replace_suffix(file, "cache"));
  std::vector<int64_t> row_bytes;
  if (idx.is_open()) 
  { while (!idx.eof() && row_bytes.size() <= size_t(M)) { int64_t i; idx >> i; row_bytes.push_back(i); } idx.close(); }

  for (int32_t i = grid_row * mb, y = 0; i < M; i = grid_row * mb + tile_m * (y += mb)) {
    int32_t ib = std::min(M - i, mb);
    std::vector<T> mat(int64_t(ib) * int64_t(N));

    if (csv.is_open()) {
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

    for (int32_t j = grid_col * nb, x = 0; j < N; j = grid_col * nb + tile_n * (x += nb))
      copy2d(ib, std::min(N - j, nb), &mat[int64_t(j) * int64_t(ib)], ib, &A[int64_t(y) + int64_t(x) * int64_t(lda)], lda);
  }

  if (csv.is_open())
    csv.close();
}

inline void matrix_eval(double& a, double d, double w) { a = std::cos(w * d) / d; }
inline void matrix_eval(float& a, double d, double w) { a = float(std::cos(w * d) / d); }
inline void matrix_eval(std::complex<double>& a, double d, double w) { a = std::complex<double>(std::cos(w * d) / d, std::sin(w * d) / d); }
inline void matrix_eval(std::complex<float>& a, double d, double w) { a = std::complex<float>(float(std::cos(w * d) / d), float(std::sin(w * d) / d)); }

template <class T>
void make_2D_oscillatory(double w, int32_t iA, int32_t jA, int32_t M, int32_t N, T* A, int32_t lda) {
  constexpr int64_t height = 128;
  auto translate_2d = [](int64_t i) { int64_t x = i / height, y = i - height * x; return std::complex<double>(x, y); };

  for (int64_t j = 0; j < N; ++j) {
    auto vj = translate_2d(j + jA + height);
    for (int64_t i = 0; i < M; ++i) {
      auto vi = translate_2d(i + iA);
      double d = std::abs(vi + std::conj(vj));
      matrix_eval(A[i + j * lda], d, w);
    }
  }
}

extern "C" void sgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const float*, const float*, const blas_int*, const float*, const blas_int*, const float*, float*, const blas_int*);
extern "C" void dgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const double*, const double*, const blas_int*, const double*, const blas_int*, const double*, double*, const blas_int*);
extern "C" void cgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, const void*, const blas_int*, const void*, void*, const blas_int*);
extern "C" void zgemm_(const char*, const char*, const blas_int*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, const void*, const blas_int*, const void*, void*, const blas_int*);

extern "C" void strsm_(const char*, const char*, const char*, const char*, const blas_int*, const blas_int*, const float*, const float*, const blas_int*, float*, const blas_int*);
extern "C" void dtrsm_(const char*, const char*, const char*, const char*, const blas_int*, const blas_int*, const double*, const double*, const blas_int*, double*, const blas_int*);
extern "C" void ctrsm_(const char*, const char*, const char*, const char*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, void*, const blas_int*);
extern "C" void ztrsm_(const char*, const char*, const char*, const char*, const blas_int*, const blas_int*, const void*, const void*, const blas_int*, void*, const blas_int*);

inline void nngemm(blas_int M, blas_int N, blas_int K, const double* A, blas_int lda, const double* B, blas_int ldb, double* C, blas_int ldc)
{ char transa = 'N', transb = 'N'; double one = 1., minus_one = -1.; dgemm_(&transa, &transb, &M, &N, &K, &one, A, &lda, B, &ldb, &minus_one, C, &ldc); }
inline void nngemm(blas_int M, blas_int N, blas_int K, const float* A, blas_int lda, const float* B, blas_int ldb, float* C, blas_int ldc)
{ char transa = 'N', transb = 'N'; float one = 1.f, minus_one = -1.f; sgemm_(&transa, &transb, &M, &N, &K, &one, A, &lda, B, &ldb, &minus_one, C, &ldc); }
inline void nngemm(blas_int M, blas_int N, blas_int K, const std::complex<double>* A, blas_int lda, const std::complex<double>* B, blas_int ldb, std::complex<double>* C, blas_int ldc)
{ char transa = 'N', transb = 'N'; std::complex<double> one(1., 0.), minus_one(-1., 0.); zgemm_(&transa, &transb, &M, &N, &K, &one, A, &lda, B, &ldb, &minus_one, C, &ldc); }
inline void nngemm(blas_int M, blas_int N, blas_int K, const std::complex<float>* A, blas_int lda, const std::complex<float>* B, blas_int ldb, std::complex<float>* C, blas_int ldc)
{ char transa = 'N', transb = 'N'; std::complex<float> one(1.f, 0.f), minus_one(-1.f, 0.f); cgemm_(&transa, &transb, &M, &N, &K, &one, A, &lda, B, &ldb, &minus_one, C, &ldc); }

template <class T>
double check_answer_lra(int32_t rank, int32_t M, int32_t N, const T* A, int32_t lda, const int32_t* jpiv, const T* R, int32_t ldr) {
  if (rank <= 0 || M <= 0 || N <= 0)
    return std::numeric_limits<double>::quiet_NaN();

  std::vector<T> matB(int64_t(M) * int64_t(N)), matC(int64_t(M) * int64_t(rank)), matR(int64_t(N) * int64_t(rank));
  for (int32_t i = 0; i < N; ++i) {
    std::copy_n(&A[int64_t(i) * int64_t(lda)], M, &matB[int64_t(i) * int64_t(M)]);
    std::copy_n(&R[int64_t(i) * int64_t(ldr)], rank, &matR[int64_t(i) * int64_t(rank)]);
    if (i < rank)
      std::copy_n(&A[int64_t(jpiv[i] - 1) * int64_t(lda)], M, &matC[int64_t(i) * int64_t(M)]);
  }

  double nrm = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  nngemm(M, N, rank, &matC[0], M, &matR[0], rank, &matB[0], M);
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
  nngemm(M, N, rank, U, ldu, V, ldv, &matB[0], M);
  double err = std::transform_reduce(matB.begin(), matB.end(), 0., std::plus<double>(), [](auto i) { return double(std::norm(i)); });
  return std::make_pair(err, nrm);
}

template <class T>
std::pair<double, double> check_answer_svd(int32_t M, int32_t N, int32_t r1, int32_t r2, const T* U, int32_t ldu, const T* V1, int32_t ldv1, const T* V2, int32_t ldv2, const T* B, int32_t ldb) {
  if (r2 <= 0)
    return std::make_pair(0., 0.);
  std::vector<T> matV(r1 * N, T());
  nngemm(r1, N, r2, V1, ldv1, V2, ldv2, &matV[0], r1);
  return check_answer_svd(M, N, r1, U, ldu, &matV[0], r1, B, ldb);
}

inline std::string conv_to_string(double f) { char s[30]; std::sprintf(s, "%.18le", f); return std::string(s); }
inline std::string conv_to_string(float f) { char s[30]; std::sprintf(s, "%.18le", double(f)); return std::string(s); }
inline std::string conv_to_string(std::complex<double> f)
{ char s[70]; char sign = 0. <= f.imag() ? '+':'-'; std::sprintf(s, " (%.18le%c%.18lej)", f.real(), sign, std::abs(f.imag())); return std::string(s); }
inline std::string conv_to_string(std::complex<float> f)
{ char s[70]; char sign = 0. <= f.imag() ? '+':'-'; std::sprintf(s, " (%.18le%c%.18lej)", double(f.real()), sign, double(std::fabs(f.imag()))); return std::string(s); }

template <class T>
void write_matrix_to_csv(int32_t M, int32_t N, const T* A, int32_t lda, const std::string& csv) {
  std::ofstream stream(csv);
  if (stream.is_open()) {
    for (int32_t i = 0; i < M; ++i) {
      for (int32_t j = 0; j < N; ++j)
        if (j == 0) stream << conv_to_string(A[int64_t(i) + int64_t(j) * int64_t(lda)]);
          else stream << "," << conv_to_string(A[int64_t(i) + int64_t(j) * int64_t(lda)]);
      stream << std::endl;
    }
    stream.close();
  }
}

template <class T> inline hyacinPrecision_t __precA();
template <> inline hyacinPrecision_t __precA<double>() { return HYACIN_F64; };
template <> inline hyacinPrecision_t __precA<float>() { return HYACIN_F32; };
template <> inline hyacinPrecision_t __precA<std::complex<double>>() { return HYACIN_F64_COMPLEX; };
template <> inline hyacinPrecision_t __precA<std::complex<float>>() { return HYACIN_F32_COMPLEX; };

template <class T>
int32_t geqp3_ronly(cublasHandle_t handle, double epi, int32_t M, int32_t N, int32_t K, const T* A, int32_t lda, int32_t* jpiv, T* R, int32_t ldr, char algo) {
  cudaStream_t stream; cublasGetStream(handle, &stream);
  int32_t umax; hyacinPrecision_t precA = __precA<T>(), precC; hyacinAlgorithm_t alg; uint64_t dev_work_bytes, dev_work_bytes_new, pinned_work_bytes;
  hyacinXsyherk_autoTune(epi, usd_nd_allreduce, u_extra, &umax, precA, &precC, &alg);
  if (algo == 'C') alg = HYACIN_ALG_CRT; else if (algo == 'L') alg = HYACIN_ALG_LIMBS;
  hyacinXsyherk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes);
  hyacinXGinterp_bufferSize(N, K, precA, precC, &dev_work_bytes_new, &pinned_work_bytes);

  void* gram = nullptr, *piv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&gram, int64_t(N) * int64_t(N) * int64_t(hyacinXelem_bytes(precC)));
  cudaMalloc(&piv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, std::max(dev_work_bytes, dev_work_bytes_new));
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  hyacinXsyherk(handle, M, N, umax, precA, A, lda, precC, gram, N, dev_work, alg);
  int32_t rank = hyacinXGinterp(handle, epi, N, K, oversampling, precA, R, ldr, (int32_t*)piv, precC, gram, N, dev_work, pinned_work);

  cudaStreamSynchronize(stream);
  cudaMemcpy(jpiv, piv, sizeof(int32_t) * N, cudaMemcpyDefault);
  cudaFree(gram); cudaFree(piv);
  cudaFree(dev_work); cudaFreeHost(pinned_work);
  return rank;
}

template <class T>
int32_t svd_fit_transform(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv) {
  int32_t umax; hyacinPrecision_t precA = __precA<T>(), precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes, dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXsyherk_autoTune(epi, usd_nd_allreduce, u_extra, &umax, precA, &precC, &alg);
  hyacinXcpqrk_bufferSize(M, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t rank = hyacinXcpqrk(cublasH, 'J', epi, M, N, K, oversampling, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg);

  hyacinXsvdk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXsvdk(cublasH, cusolverH, params, 'Y', epi, M, N, rank, oversampling, nullptr, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

#ifndef NO_NCCL

template <class T>
int32_t svd_fit_transform_1dr(cudaStream_t stream, cublasHandle_t cublasH, cusolverDnHandle_t cusolverH, cusolverDnParams_t params, double epi, int32_t M, int32_t gM, int32_t N, int32_t K, T* A, int32_t lda, T* V, int32_t ldv, ncclComm_t comm) {
  int32_t umax; hyacinPrecision_t precA = __precA<T>(), precC; hyacinAlgorithm_t alg;
  uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXsyherk_autoTune(epi, usd_nd_allreduce, u_extra, &umax, precA, &precC, &alg);
  hyacinXcpqrk1Drow_bufferSize(M, gM, N, umax, precC, alg, &dev_work_bytes, &pinned_work_bytes);

  void* jpiv = nullptr, *dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&jpiv, int64_t(N) * sizeof(int32_t));
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);

  int32_t rank = hyacinXcpqrk1Drow(cublasH, 'J', epi, M, gM, N, K, oversampling, umax, precA, A, lda, (int32_t*)jpiv, precA, V, ldv, precC, dev_work, pinned_work, alg, comm);

  uint64_t dev_work_bytes_new, pinned_work_bytes_new;
  hyacinXsvdk_bufferSize(cusolverH, params, epi, N, rank, precA, &dev_work_bytes_new, &pinned_work_bytes_new);
  if (dev_work_bytes < dev_work_bytes_new) { cudaStreamSynchronize(stream); cudaFree(dev_work); cudaMalloc(&dev_work, dev_work_bytes = dev_work_bytes_new); }
  if (pinned_work_bytes < pinned_work_bytes_new) { cudaStreamSynchronize(stream); cudaFreeHost(pinned_work); cudaMallocHost(&pinned_work, pinned_work_bytes = pinned_work_bytes_new); }

  rank = hyacinXsvdk(cublasH, cusolverH, params, 'Y', epi, M, N, rank, oversampling, nullptr, A, lda, V, ldv, precA, dev_work_bytes, dev_work, pinned_work_bytes, pinned_work);

  cudaStreamSynchronize(stream);
  cudaFree(jpiv);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return rank;
}

template <class T>
std::pair<int32_t, int32_t> allgatherv_1dc(cudaStream_t stream, int32_t M, int32_t N, T* A, int32_t lda, ncclComm_t comm) {
  hyacinPrecision_t precA = __precA<T>(); int32_t comm_size; ncclCommCount(comm, &comm_size);
  uint64_t dev_work_bytes, pinned_work_bytes;
  hyacinXAllGatherV1Dcol_bufferSize(M, comm_size, precA, &dev_work_bytes, &pinned_work_bytes);

  void* dev_work = nullptr, *pinned_work = nullptr;
  cudaMalloc(&dev_work, dev_work_bytes);
  cudaMallocHost(&pinned_work, pinned_work_bytes);
  int32_t v_offset = hyacinXAllGatherV1Dcol(stream, M, &N, precA, A, lda, dev_work_bytes, dev_work, pinned_work, comm);

  cudaStreamSynchronize(stream);
  cudaFree(dev_work);
  cudaFreeHost(pinned_work);
  return std::make_pair(N, v_offset);
}

#ifdef BOOTSTRAP_POSIX
#include <unistd.h>
#include <fcntl.h>

void __bootstrap(int32_t& world_rank, int32_t& world_size, int32_t& local_rank, ncclUniqueId& id) {
  const char* slurm_rank = std::getenv("SLURM_PROCID"), *slurm_size = std::getenv("SLURM_NTASKS");
  const char* slurm_local_rank = std::getenv("SLURM_LOCALID");
  world_rank = std::stoi(std::string(slurm_rank ?: "0"));
  world_size = std::stoi(std::string(slurm_size ?: "1"));
  local_rank = std::stoi(std::string(slurm_local_rank ?: "0"));

  std::string slurm_job_dir(std::getenv("SLURM_SUBMIT_DIR") ?: ".");
  std::string slurm_job_id(std::getenv("SLURM_JOB_ID") ?: "0");
  std::string hyac_job_id(std::getenv("HYACIN_JOB_ID") ?: "0");
  std::string id_path = slurm_job_dir + "/" + slurm_job_id + "-" + hyac_job_id + ".id.out";
  if (world_rank == 0) {
    ncclGetUniqueId(&id); std::string tmp = id_path + ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if (fd != -1 && ::write(fd, &id, sizeof(ncclUniqueId)) == sizeof(ncclUniqueId))
    { if (::fsync(fd) != -1) if (::close(fd) != -1) ::rename(tmp.c_str(), id_path.c_str()); }
    else if (fd != -1) { ::close(fd); throw std::runtime_error("I/O error writing nccl unique ID."); }
  }
  else {
    int trials = 0; std::ifstream stream(id_path, std::ios::binary);
    while (!stream.is_open() && ++trials <= 20) { ::sleep(1); stream.open(id_path, std::ios::binary); }
    if (stream.is_open()) { stream.read((char*)&id, sizeof(ncclUniqueId)); stream.close(); }
    else { throw std::runtime_error("Timeout reading nccl unique ID."); }
  }
}

#else
#include <mpi.h>

void __bootstrap(int32_t& world_rank, int32_t& world_size, int32_t& local_rank, ncclUniqueId& id) {
  MPI_Init(nullptr, nullptr);
  MPI_Comm shmcomm; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &shmcomm);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_rank(shmcomm, &local_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_free(&shmcomm);

  if (world_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(ncclUniqueId), MPI_BYTE, 0, MPI_COMM_WORLD);
  MPI_Finalize();
}

#endif
#endif
