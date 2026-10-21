
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <vector>
#include <algorithm>
#include <stdexcept>

/* Performance parameters */
#ifndef HYACIN_DEFAULT_GRAM_ALGORITHM
#define HYACIN_DEFAULT_GRAM_ALGORITHM 'A'
#endif
#ifndef HYACIN_DEFAULT_STRIDE_BATCH_PERF_PENALTY
#define HYACIN_DEFAULT_STRIDE_BATCH_PERF_PENALTY 1.3
#endif
#ifndef HYACIN_DEFAULT_GRAM_SEGMENT_K
#define HYACIN_DEFAULT_GRAM_SEGMENT_K 8192
#endif
#ifndef HYACIN_DEFAULT_TCGEMM_SEGMENT_K
#define HYACIN_DEFAULT_TCGEMM_SEGMENT_K 8192
#endif

/* Spectral parameters */
#ifndef HYACIN_DEFAULT_GRAM_SVD_EPI_CORR
#define HYACIN_DEFAULT_GRAM_SVD_EPI_CORR 0.5
#endif
#ifndef HYACIN_DEFAULT_QUANTIZE_BITS_CORR
#define HYACIN_DEFAULT_QUANTIZE_BITS_CORR 7
#endif
#ifndef HYACIN_DEFAULT_JACOBI_SVD_SWEEPS
#define HYACIN_DEFAULT_JACOBI_SVD_SWEEPS 5
#endif

#ifndef HYACIN_DEFAULT_CREATE_TIMER
#define HYACIN_DEFAULT_CREATE_TIMER 1
#endif

const std::vector<int32_t> f64_capable_sm_list({ 80, 90, 100, 107 }); // sm80,sm90,sm100,sm107
const std::vector<int32_t> limbs_u_list({ 7, 14, 22, 30, 38, 46, 54, 62, 70, 78, 86, 94 });

std::pair<int32_t, int32_t> internal::gram_algorithm(char& alg, int32_t K, int32_t u, int32_t Complex, int32_t segmentK, double batch_penalty) {
  u = std::max(0, u); K = std::max(1, K); int32_t crt_bits = (u + u + 2) + int32_t(std::ceil(std::log2(double(0 < segmentK ? std::min(K, segmentK) : K))));
  if (alg == 'C' || alg == 'c') {
    if (crt_bits <= U8CRT::range[22]) {
      int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], crt_bits)));
      alg = 'C'; return std::make_pair(orderA_crt, std::min((K + 255) & (~255), segmentK << (U8CRT::range[orderA_crt - 1] - crt_bits)));
    } else { alg = 'C'; return std::make_pair(0, 0); }
  } else {
    int32_t orderA_limbs = 1 + std::distance(limbs_u_list.begin(), std::lower_bound(limbs_u_list.begin(), limbs_u_list.end(), u));
    if (alg != 'L' && alg != 'l' && 3 < orderA_limbs) {
      int32_t cost_limbs = int32_t(batch_penalty * double(orderA_limbs)) + int32_t(uint32_t(orderA_limbs * (orderA_limbs - 1)) >> 1) + (Complex ? orderA_limbs * orderA_limbs : 0);
      if (crt_bits <= U8CRT::range[22]) {
        int32_t orderA_crt = 1 + int32_t(std::distance(&U8CRT::range[0], std::lower_bound(&U8CRT::range[1], &U8CRT::range[23], crt_bits)));
        int32_t cost_crt = int32_t(batch_penalty * double(Complex ? (orderA_crt + orderA_crt) : orderA_crt));
        if (cost_crt < cost_limbs)
        { alg = 'C'; return std::make_pair(orderA_crt, std::min((K + 255) & (~255), segmentK << (U8CRT::range[orderA_crt - 1] - crt_bits))); }
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

constexpr uint32_t HANDLE_OWNS_CUDA_STREAM = 0x1;
constexpr uint32_t HANDLE_OWNS_CUBLAS = 0x2;
constexpr uint32_t HANDLE_OWNS_CUSOLVER = 0x4;
constexpr uint32_t HANDLE_OWNS_MEMPOOL = 0x8;
constexpr uint32_t HANDLE_OWNS_PINNED_WORK = 0x10;
constexpr uint32_t HANDLE_OWNS_TIMER = 0x20;

extern "C" void hyacinCreate(hyacinHandle_t* handle) {
  auto get_env = [](const std::string& key) { const char* val = std::getenv(key.c_str()); return val ? std::string(val) : std::string(""); };
  std::string env_str;
  handle->GramMatrixAlgorithm = ((env_str = get_env("HYACIN_GRAM_ALGORITHM")) == "") ? HYACIN_DEFAULT_GRAM_ALGORITHM : env_str[0];
  handle->StrideBatchPerfPenalty = ((env_str = get_env("HYACIN_STRIDE_BATCH_PERF_PENALTY")) == "") ? HYACIN_DEFAULT_STRIDE_BATCH_PERF_PENALTY : std::stod(env_str);
  handle->GramSegmentK = ((env_str = get_env("HYACIN_GRAM_SEGMENT_K")) == "") ? HYACIN_DEFAULT_GRAM_SEGMENT_K : std::stoi(env_str);
  handle->TCGemmSegmentK = ((env_str = get_env("HYACIN_TCGEMM_SEGMENT_K")) == "") ? HYACIN_DEFAULT_TCGEMM_SEGMENT_K : std::stoi(env_str);

  handle->GramSVDEpiCorrection = ((env_str = get_env("HYACIN_GRAM_SVD_EPI_CORR")) == "") ? HYACIN_DEFAULT_GRAM_SVD_EPI_CORR : std::stod(env_str);
  handle->QuantizeBitCorrection = ((env_str = get_env("HYACIN_QUANTIZE_BITS_CORR")) == "") ? HYACIN_DEFAULT_QUANTIZE_BITS_CORR : std::stoi(env_str);
  handle->JacobiSVDSweeps = ((env_str = get_env("HYACIN_JACOBI_SVD_SWEEPS")) == "") ? HYACIN_DEFAULT_JACOBI_SVD_SWEEPS : std::stoi(env_str);

  cudaGetDevice(&handle->DeviceID); int32_t major, minor;
  cudaDeviceGetAttribute(&handle->DeviceSMs, cudaDevAttrMultiProcessorCount, handle->DeviceID);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, handle->DeviceID);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, handle->DeviceID);
  handle->DeviceIsF64Capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), 10 * major + minor));

  if (!(handle->HandleOwnedResource & HANDLE_OWNS_CUDA_STREAM))
  { cudaStreamCreateWithFlags(&handle->cudaStream, cudaStreamNonBlocking); handle->HandleOwnedResource |= HANDLE_OWNS_CUDA_STREAM; }
  if (!(handle->HandleOwnedResource & HANDLE_OWNS_CUBLAS))
  { cublasCreate(&handle->cublasHandle); handle->HandleOwnedResource |= HANDLE_OWNS_CUBLAS; }
  if (!(handle->HandleOwnedResource & HANDLE_OWNS_CUSOLVER))
  { cusolverDnCreate(&handle->cusolverHandle); handle->HandleOwnedResource |= HANDLE_OWNS_CUSOLVER; }
  if (!(handle->HandleOwnedResource & HANDLE_OWNS_MEMPOOL)) {
    cudaMemPoolProps props{}; props.location.id = handle->DeviceID; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice;
    cudaMemPoolCreate(&handle->mempool, &props); handle->HandleOwnedResource |= HANDLE_OWNS_MEMPOOL;
  }
  if (!(handle->HandleOwnedResource & HANDLE_OWNS_PINNED_WORK))
  { cudaMallocHost(&handle->pinnedWorkspace, sizeof(int32_t)); handle->HandleOwnedResource |= HANDLE_OWNS_PINNED_WORK; }

  uint64_t threshold = std::numeric_limits<uint64_t>::max();
  cudaMemPoolSetAttribute(handle->mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
  cublasSetStream(handle->cublasHandle, handle->cudaStream);
  cusolverDnSetStream(handle->cusolverHandle, handle->cudaStream);

  int32_t create_timer = ((env_str = get_env("HYACIN_CREATE_TIMER")) == "") ? HYACIN_DEFAULT_CREATE_TIMER : int32_t(env_str == "1" || env_str == "true" || env_str == "True");
  if (!(handle->HandleOwnedResource & HANDLE_OWNS_TIMER) && create_timer)
  { handle->timer = (new EventTimer()); handle->HandleOwnedResource |= HANDLE_OWNS_TIMER; }
}

extern "C" void hyacinDestroy(hyacinHandle_t* handle) {
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUDA_STREAM) && (handle->cudaStream != nullptr)) { cudaStreamDestroy(handle->cudaStream); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUBLAS) && (handle->cublasHandle != nullptr)) { cublasDestroy(handle->cublasHandle); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUSOLVER) && (handle->cusolverHandle != nullptr)) { cusolverDnDestroy(handle->cusolverHandle); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_MEMPOOL) && (handle->mempool != nullptr)) { cudaMemPoolDestroy(handle->mempool); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_PINNED_WORK) && (handle->pinnedWorkspace != nullptr)) { cudaFreeHost(handle->pinnedWorkspace); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_TIMER) && (handle->timer != nullptr)) { delete (EventTimer*)(handle->timer); }
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
