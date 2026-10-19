
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>

#ifndef HYACIN_DEFAULT_GRAM_ALGORITHM
#define HYACIN_DEFAULT_GRAM_ALGORITHM 'A'
#endif
#ifndef HYACIN_DEFAULT_QUANTIZE_BITS_CORR
#define HYACIN_DEFAULT_QUANTIZE_BITS_CORR 5
#endif
#ifndef HYACIN_DEFAULT_GRAM_BITS_CORR
#define HYACIN_DEFAULT_GRAM_BITS_CORR 0
#endif
#ifndef HYACIN_DEFAULT_JACOBI_SVD_SWEEPS
#define HYACIN_DEFAULT_JACOBI_SVD_SWEEPS 5
#endif
#ifndef HYACIN_DEFAULT_PRECOND_OVERSAMPLING
#define HYACIN_DEFAULT_PRECOND_OVERSAMPLING 10
#endif
#ifndef HYACIN_DEFAULT_CREATE_TIMER
#define HYACIN_DEFAULT_CREATE_TIMER 1
#endif

const std::vector<int32_t> f64_capable_sm_list({ 80, 90, 100, 107 }); // sm80,sm90,sm100,sm107
const std::vector<int32_t> limbs_u_list({ 7, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94 });

std::pair<int32_t, int32_t> internal::gram_algorithm(char& alg, int32_t M, int32_t u, int32_t Complex) {
  constexpr int32_t CRT_MIN_SEGMENT_K = 8192; constexpr double SBATCH_PENALTY = 1.4;
  u = std::max(0, u); M = std::max(1, M); int32_t crt_bits = (u + u + 2) + int32_t(std::ceil(std::log2(double(std::min(M, CRT_MIN_SEGMENT_K)))));
  if (alg == 'C' || alg == 'c') {
    if (crt_bits <= U8CRT::range[22]) {
      int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], crt_bits)));
      alg = 'C'; return std::make_pair(orderA_crt, std::min((M + 255) & (~255), CRT_MIN_SEGMENT_K << (U8CRT::range[orderA_crt - 1] - crt_bits)));
    } else { alg = 'C'; return std::make_pair(0, 0); }
  } else {
    int32_t orderA_limbs = 1 + std::distance(limbs_u_list.begin(), std::lower_bound(limbs_u_list.begin(), limbs_u_list.end(), u));
    if (alg != 'L' && alg != 'l' && 3 < orderA_limbs) {
      int32_t cost_limbs = int32_t(SBATCH_PENALTY * double(orderA_limbs)) + int32_t(uint32_t(orderA_limbs * (orderA_limbs - 1)) >> 1) + (Complex ? orderA_limbs * orderA_limbs : 0);
      if (crt_bits <= U8CRT::range[22]) {
        int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], crt_bits)));
        int32_t cost_crt = int32_t(SBATCH_PENALTY * double(Complex ? (orderA_crt + orderA_crt) : orderA_crt));
        if (cost_crt < cost_limbs)
        { alg = 'C'; return std::make_pair(orderA_crt, std::min((M + 255) & (~255), CRT_MIN_SEGMENT_K << (U8CRT::range[orderA_crt - 1] - crt_bits))); }
      }
    }
    alg = 'L'; return std::make_pair(orderA_limbs, limbs_u_list[orderA_limbs - 1]);
  }
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
  handle->QuantizeBitCorrection = ((env_str = get_env("HYACIN_QUANTIZE_BITS_CORR")) == "") ? HYACIN_DEFAULT_QUANTIZE_BITS_CORR : std::stoi(env_str);
  handle->GramBitCorrection = ((env_str = get_env("HYACIN_GRAM_BITS_CORR")) == "") ? HYACIN_DEFAULT_GRAM_BITS_CORR : std::stoi(env_str);
  handle->JacobiSVDSweeps = ((env_str = get_env("HYACIN_JACOBI_SVD_SWEEPS")) == "") ? HYACIN_DEFAULT_JACOBI_SVD_SWEEPS : std::stoi(env_str);

  cudaGetDevice(&handle->DeviceID); int32_t major, minor;
  cudaDeviceGetAttribute(&handle->DeviceSMs, cudaDevAttrMultiProcessorCount, handle->DeviceID);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, handle->DeviceID);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, handle->DeviceID);
  handle->DeviceIsF64Capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), 10 * major + minor));

  cudaStreamCreateWithFlags(&handle->cudaStream, cudaStreamNonBlocking);
  cudaMemPoolProps props = cudaMemPoolProps();
  props.location.id = handle->DeviceID; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice;
  cudaMemPoolCreate(&handle->mempool, &props);
  uint64_t threshold = std::numeric_limits<uint64_t>::max();
  cudaMemPoolSetAttribute(handle->mempool, cudaMemPoolAttrReleaseThreshold, &threshold);

  cublasCreate(&handle->cublasHandle);
  cublasSetStream(handle->cublasHandle, handle->cudaStream);
  cusolverDnCreate(&handle->cusolverHandle);
  cusolverDnSetStream(handle->cusolverHandle, handle->cudaStream);
  cudaMallocHost(&handle->pinnedWorkspace, sizeof(int32_t));
#ifndef NO_NCCL
  handle->ncclComm = nullptr;
#endif
  handle->timer = (((env_str = get_env("HYACIN_CREATE_TIMER")) == "") ? HYACIN_DEFAULT_CREATE_TIMER : (env_str == "1" || env_str == "true" || env_str == "True")) ? (new EventTimer()) : nullptr;
}

extern "C" void hyacinDestroy(hyacinHandle_t* handle) {
  if (handle->cudaStream) { cudaStreamDestroy(handle->cudaStream); }
  if (handle->cublasHandle) { cublasDestroy(handle->cublasHandle); }
  if (handle->cusolverHandle) { cusolverDnDestroy(handle->cusolverHandle); }
  if (handle->mempool) { cudaMemPoolDestroy(handle->mempool); }
  if (handle->pinnedWorkspace) { cudaFreeHost(handle->pinnedWorkspace); }
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
  if (handle->timer == nullptr) { cudaStreamSynchronize(handle->cudaStream); return; }
  EventTimer* t = (EventTimer*)(handle->timer);
  int32_t len = int32_t(t->events.size());
  cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, handle->cudaStream); cudaEventSynchronize(e);
  t->events.emplace_back(t->lastSegment, e);

  if (eventMs != nullptr && 0 < lenMs) {
    double d_time = 0., r_time = 0., c_time = 0.;
    if (len) for (auto [seg, event] : t->events) {
      float ms = 0.f;
      if (seg == segment::dist_kernel) { cudaEventElapsedTime(&ms, e, event); e = event; d_time += double(ms); } else
      if (seg == segment::rep_kernel) { cudaEventElapsedTime(&ms, e, event); e = event; r_time += double(ms); } else
      if (seg == segment::comm) { cudaEventElapsedTime(&ms, e, event); e = event; c_time += double(ms); } else { e = event; }
    }
    for (int32_t i = 0; i < lenMs; ++i) {
      char req = static_cast<char>(eventMs[i]);
      if (req == 'D' || req == 'd') { eventMs[i] = d_time; } else
      if (req == 'R' || req == 'r') { eventMs[i] = r_time; } else
      if (req == 'C' || req == 'c') { eventMs[i] = c_time; } else { eventMs[i] = 0.; }
    }
  }
  for (auto [seg, event] : t->events) { cudaEventDestroy(event); } t->events.clear(); t->lastSegment = segment::none;
}
