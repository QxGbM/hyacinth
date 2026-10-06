
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

#ifndef HYACIN_DEFAULT_GRAM_ALGORITHM
#define HYACIN_DEFAULT_GRAM_ALGORITHM 'A'
#endif
#ifndef HYACIN_DEFAULT_BATCH_K
#define HYACIN_DEFAULT_BATCH_K 65536
#endif
#ifndef HYACIN_DEFAULT_QUANTIZE_BITS_CORR
#define HYACIN_DEFAULT_QUANTIZE_BITS_CORR 5
#endif
#ifndef HYACIN_DEFAULT_GRAM_BITS_CORR
#define HYACIN_DEFAULT_GRAM_BITS_CORR -5
#endif
#ifndef HYACIN_DEFAULT_JACOBI_SVD_SWEEPS
#define HYACIN_DEFAULT_JACOBI_SVD_SWEEPS 15
#endif
#ifndef HYACIN_DEFAULT_PRECOND_OVERSAMPLING
#define HYACIN_DEFAULT_PRECOND_OVERSAMPLING 10
#endif
#ifndef HYACIN_DEFAULT_CREATE_TIMER
#define HYACIN_DEFAULT_CREATE_TIMER 1
#endif

int32_t device_sms = 0; bool device_f64_capable = false;
const std::vector<int32_t> f64_capable_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100

inline void device_params() {
  int32_t device, major, minor; cudaGetDevice(&device);
  cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device_f64_capable = (f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), 100 * major + minor));
}

bool internal::device_is_f64_capable() {
  if (device_sms == 0) { device_params(); } return device_f64_capable;
}

int32_t internal::device_num_sms() {
  if (device_sms == 0) { device_params(); } return device_sms;
}

std::pair<int32_t, int32_t> internal::gram_algorithm(char& alg, int32_t M, int32_t& u, int32_t Complex) {
  constexpr int32_t CRT_MIN_SEGMENT_K = 8192; constexpr double SBATCH_PENALTY = 1.4;
  u = std::max(0, u);
  int32_t orderA_limbs = (u <= 7) ? 1 : int32_t(uint32_t(u + 9) >> 3);
  if (alg != 'L' && alg != 'l' && 3 < orderA_limbs) {
    int32_t nrm_M = std::min(std::max(1, M), CRT_MIN_SEGMENT_K), crt_bits = (u + u + 2) + int32_t(std::ceil(std::log2(double(nrm_M)))), orderA_crt, cost_crt;
    int32_t cost_limbs = int32_t(SBATCH_PENALTY * double(orderA_limbs)) + int32_t(uint32_t(orderA_limbs * (orderA_limbs - 1)) >> 1) + (Complex ? orderA_limbs * orderA_limbs : 0);
    if (crt_bits <= U8CRT::range[22]) {
      orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], crt_bits)));
      cost_crt = int32_t(SBATCH_PENALTY * double(Complex ? (orderA_crt + orderA_crt) : orderA_crt));
    } else { orderA_crt = cost_crt = std::numeric_limits<int32_t>::max(); }
    if (alg == 'C' || alg == 'c' || cost_crt < cost_limbs)
    { alg = 'C'; std::div_t divM = std::div(M, nrm_M << (U8CRT::range[orderA_crt - 1] - crt_bits)); return std::make_pair(orderA_crt, divM.quot + int32_t(0 < divM.rem)); }
  }
  alg = 'L'; u = (u <= 7) ? 7 : ((orderA_limbs << 3) - 2);
  return std::make_pair(orderA_limbs, 1);
}

enum class segment : unsigned char { none, dist_kernel, rep_kernel, comm };
struct EventTimer {
  std::vector<std::pair<segment, cudaEvent_t>> events;
  segment lastSegment = segment::none;
};

extern "C" void hyacinCreate(hyacinHandle_t* handle) {
  auto get_env = [](const std::string& key) { const char* val = std::getenv(key.c_str()); return val ? std::string(val) : std::string(""); };
  std::string env_str;
  handle->GramMatrixAlgorithm = ((env_str = get_env("HYACIN_GRAM_ALGORITHM")) == "") ? HYACIN_DEFAULT_GRAM_ALGORITHM : env_str[0];
  handle->BatchK = ((env_str = get_env("HYACIN_BATCH_K")) == "") ? HYACIN_DEFAULT_BATCH_K : std::stoi(env_str);
  handle->QuantizeBitCorrection = ((env_str = get_env("HYACIN_QUANTIZE_BITS_CORR")) == "") ? HYACIN_DEFAULT_QUANTIZE_BITS_CORR : std::stoi(env_str);
  handle->GramBitCorrection = ((env_str = get_env("HYACIN_GRAM_BITS_CORR")) == "") ? HYACIN_DEFAULT_GRAM_BITS_CORR : std::stoi(env_str);
  handle->JacobiSVDSweeps = ((env_str = get_env("HYACIN_JACOBI_SVD_SWEEPS")) == "") ? HYACIN_DEFAULT_JACOBI_SVD_SWEEPS : std::stoi(env_str);
  handle->RankOversampling = ((env_str = get_env("HYACIN_PRECOND_OVERSAMPLING")) == "") ? HYACIN_DEFAULT_PRECOND_OVERSAMPLING : std::stoi(env_str);
  handle->Batches = 0; handle->BatchTensor = nullptr;

  cudaStreamCreateWithFlags(&handle->cudaStream, cudaStreamNonBlocking);
  cublasCreate(&handle->cublasHandle);
  cublasSetStream(handle->cublasHandle, handle->cudaStream);
  cusolverDnCreate(&handle->cusolverHandle);
  cusolverDnSetStream(handle->cusolverHandle, handle->cudaStream);
  cudaMemPoolProps props = cudaMemPoolProps(); cudaGetDevice(&props.location.id);
  props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice;
  cudaMemPoolCreate(&handle->mempool, &props);
  uint64_t threshold = std::numeric_limits<uint64_t>::max();
  cudaMemPoolSetAttribute(handle->mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
  cudaMallocHost(&handle->pinnedWorkspace, size_t(128));
#ifndef NO_NCCL
  handle->col_comm = handle->row_comm = nullptr;
#endif
  handle->timer = (((env_str = get_env("HYACIN_CREATE_TIMER")) == "") ? HYACIN_DEFAULT_CREATE_TIMER : (env_str == "1" || env_str == "true" || env_str == "True")) ? (new EventTimer()) : nullptr;
}

#ifndef NO_NCCL
extern "C" void hyacinCreate2D(hyacinHandle_t* handle, ncclComm_t col_comm, ncclComm_t row_comm) {
  hyacinCreate(handle);
  handle->col_comm = col_comm;
  handle->row_comm = row_comm;
}
#endif

extern "C" void hyacinDestroy(hyacinHandle_t* handle) {
  if (handle->cudaStream) { cudaStreamDestroy(handle->cudaStream); }
  if (handle->cublasHandle) { cublasDestroy(handle->cublasHandle); }
  if (handle->cusolverHandle) { cusolverDnDestroy(handle->cusolverHandle); }
  if (handle->mempool) { cudaMemPoolDestroy(handle->mempool); }
  if (handle->pinnedWorkspace) { cudaFreeHost(handle->pinnedWorkspace); }
  if (handle->BatchTensor) { std::free(handle->BatchTensor); }
  if (handle->timer) { delete (EventTimer*)(handle->timer); }
  std::memset(handle, 0, sizeof(hyacinHandle_t));
}

template <segment seg> inline void register_event(cudaStream_t stream, EventTimer* timer) {
  if (timer) if (timer->lastSegment != seg) {
    cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, stream);
    segment prev = timer->lastSegment; timer->events.emplace_back(prev, e); timer->lastSegment = seg;
  }
}

void Timer::register_distribute_kernel(cudaStream_t stream, void* timer) { register_event<segment::dist_kernel>(stream, (EventTimer*)timer); }
void Timer::register_replicate_kernel(cudaStream_t stream, void* timer) { register_event<segment::rep_kernel>(stream, (EventTimer*)timer); }
void Timer::register_comm(cudaStream_t stream, void* timer) { register_event<segment::comm>(stream, (EventTimer*)timer); }

extern "C" void hyacinSync_TimerSegments(const hyacinHandle_t* handle, double* eventMs, int32_t lenMs) {
  if (handle->timer == nullptr || eventMs == nullptr || lenMs <= 0) 
  { cudaStreamSynchronize(handle->cudaStream); return; }

  double d_time = 0., r_time = 0., c_time = 0.; EventTimer* t = (EventTimer*)(handle->timer);
  int32_t len = int32_t(t->events.size());
  cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, handle->cudaStream); cudaEventSynchronize(e);
  t->events.emplace_back(t->lastSegment, e);

  if (len) for (auto [seg, event] : t->events) {
    float milliseconds = 0.f;
    if (seg == segment::dist_kernel) { cudaEventElapsedTime(&milliseconds, e, event); e = event; d_time += double(milliseconds); } else
    if (seg == segment::rep_kernel) { cudaEventElapsedTime(&milliseconds, e, event); e = event; r_time += double(milliseconds); } else
    if (seg == segment::comm) { cudaEventElapsedTime(&milliseconds, e, event); e = event; c_time += double(milliseconds); } else
    { e = event; }
  }

  for (auto [seg, event] : t->events) { cudaEventDestroy(event); }
  t->events.clear(); t->lastSegment = segment::none;
  for (int32_t i = 0; i < lenMs; ++i) {
    char req = static_cast<char>(eventMs[i]);
    if (req == 'D' || req == 'd') { eventMs[i] = d_time; } else
    if (req == 'R' || req == 'r') { eventMs[i] = r_time; } else
    if (req == 'C' || req == 'c') { eventMs[i] = c_time; }
  }
}
