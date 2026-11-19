
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <vector>
#include <algorithm>
#include <string>
#include <cassert>

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
#ifndef HYACIN_DEFAULT_QUANTIZE_BITS_CORR
#define HYACIN_DEFAULT_QUANTIZE_BITS_CORR 7
#endif
#ifndef HYACIN_DEFAULT_QRCP_TRUNCATE_RANK
#define HYACIN_DEFAULT_QRCP_TRUNCATE_RANK 1
#endif
#ifndef HYACIN_DEFAULT_JACOBI_SVD_SWEEPS
#define HYACIN_DEFAULT_JACOBI_SVD_SWEEPS 3
#endif

#ifndef HYACIN_DEFAULT_CREATE_TIMER
#define HYACIN_DEFAULT_CREATE_TIMER 0
#endif

const std::vector<int32_t> f64_capable_sm_list({ 80, 90, 100, 107 }); // sm80,sm90,sm100,sm107
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

template <class T> static inline bool handle_is_null(const T& handle) {
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&handle);
  return std::all_of(bytes, &bytes[sizeof(T)], [](unsigned char byte) { constexpr unsigned char zero = 0; return byte == zero; });
}

extern "C" void hyacinCreate(hyacinHandle_t* handle) {
  CUDA_CHECK(cudaGetDevice(&handle->DeviceID)); int32_t major, minor;
  CUDA_CHECK(cudaDeviceGetAttribute(&handle->DeviceSMs, cudaDevAttrMultiProcessorCount, handle->DeviceID));
  CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, handle->DeviceID));
  CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, handle->DeviceID));
  handle->DeviceIsF64Capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), 10 * major + minor));

  auto get_env = [](const std::string& key) { const char* val = std::getenv(key.c_str()); return val ? std::string(val) : std::string(""); };
  std::string env_str;
  handle->GramMatrixAlgorithm = ((env_str = get_env("HYACIN_GRAM_ALGORITHM")) == "") ? HYACIN_DEFAULT_GRAM_ALGORITHM : env_str[0];
  handle->StrideBatchPerfPenalty = ((env_str = get_env("HYACIN_STRIDE_BATCH_PERF_PENALTY")) == "") ? HYACIN_DEFAULT_STRIDE_BATCH_PERF_PENALTY : std::stod(env_str);
  handle->GramSegmentK = ((env_str = get_env("HYACIN_GRAM_SEGMENT_K")) == "") ? HYACIN_DEFAULT_GRAM_SEGMENT_K : std::stoi(env_str);
  handle->TCGemmTargetK = ((env_str = get_env("HYACIN_TCGEMM_TARGET_K")) == "") ? HYACIN_DEFAULT_TCGEMM_SEGMENT_K : std::stoi(env_str);

  handle->QuantizeBitCorrection = ((env_str = get_env("HYACIN_QUANTIZE_BITS_CORR")) == "") ? HYACIN_DEFAULT_QUANTIZE_BITS_CORR : std::stoi(env_str);
  handle->QRCPTruncateRank = ((env_str = get_env("HYACIN_QRCP_TRUNCATE_RANK")) == "") ? HYACIN_DEFAULT_QRCP_TRUNCATE_RANK : std::stoi(env_str);
  handle->JacobiSVDSweeps = ((env_str = get_env("HYACIN_JACOBI_SVD_SWEEPS")) == "") ? HYACIN_DEFAULT_JACOBI_SVD_SWEEPS : std::stoi(env_str);

  if (handle_is_null(handle->cudaStream))
  { CUDA_CHECK(cudaStreamCreateWithFlags(&handle->cudaStream, cudaStreamDefault)); handle->HandleOwnedResource |= HANDLE_OWNS_CUDA_STREAM; }
  if (handle_is_null(handle->cublasHandle))
  { CUBLAS_CHECK(cublasCreate(&handle->cublasHandle)); CUBLAS_CHECK(cublasSetStream(handle->cublasHandle, handle->cudaStream)); handle->HandleOwnedResource |= HANDLE_OWNS_CUBLAS; }
  if (handle_is_null(handle->cusolverHandle))
  { CUSOLVER_CHECK(cusolverDnCreate(&handle->cusolverHandle)); CUSOLVER_CHECK(cusolverDnSetStream(handle->cusolverHandle, handle->cudaStream)); handle->HandleOwnedResource |= HANDLE_OWNS_CUSOLVER; }
  if (handle_is_null(handle->mempool)) {
    cudaMemPoolProps props{}; props.location.id = handle->DeviceID; props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice;
    CUDA_CHECK(cudaMemPoolCreate(&handle->mempool, &props)); uint64_t threshold = std::numeric_limits<uint64_t>::max();
    CUDA_CHECK(cudaMemPoolSetAttribute(handle->mempool, cudaMemPoolAttrReleaseThreshold, &threshold)); handle->HandleOwnedResource |= HANDLE_OWNS_MEMPOOL;
  }
  if (handle->pinnedWorkspace == nullptr)
  { CUDA_CHECK(cudaMallocHost(&handle->pinnedWorkspace, sizeof(int32_t))); handle->HandleOwnedResource |= HANDLE_OWNS_PINNED_WORK; }

  int32_t create_timer = ((env_str = get_env("HYACIN_CREATE_TIMER")) == "") ? HYACIN_DEFAULT_CREATE_TIMER : int32_t(env_str == "1" || env_str == "true" || env_str == "True");
  if (create_timer && (handle->timer == nullptr))
  { handle->timer = (new EventTimer()); handle->HandleOwnedResource |= HANDLE_OWNS_TIMER; }
  env_str = get_env("HYACIN_VERBOSE"); if (env_str == "1" || env_str == "true" || env_str == "True") { handle->logger = stdout; }
}

extern "C" void hyacinDestroy(hyacinHandle_t* handle) {
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUDA_STREAM) && (!handle_is_null(handle->cudaStream))) { CUDA_CHECK(cudaStreamDestroy(handle->cudaStream)); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUBLAS) && (!handle_is_null(handle->cublasHandle))) { CUBLAS_CHECK(cublasDestroy(handle->cublasHandle)); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_CUSOLVER) && (!handle_is_null(handle->cusolverHandle))) { CUSOLVER_CHECK(cusolverDnDestroy(handle->cusolverHandle)); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_MEMPOOL) && (!handle_is_null(handle->mempool))) { CUDA_CHECK(cudaMemPoolDestroy(handle->mempool)); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_PINNED_WORK) && (handle->pinnedWorkspace != nullptr)) { CUDA_CHECK(cudaFreeHost(handle->pinnedWorkspace)); }
  if ((handle->HandleOwnedResource & HANDLE_OWNS_TIMER) && (handle->timer != nullptr)) { delete (EventTimer*)(handle->timer); }
  std::memset(handle, 0, sizeof(hyacinHandle_t));
}

template <segment seg> static inline void register_event(cudaStream_t stream, EventTimer* timer) {
  if (timer) if (timer->lastSegment != seg) {
    cudaEvent_t e; CUDA_CHECK(cudaEventCreate(&e));
    CUDA_CHECK(cudaEventRecord(e, stream));
    segment prev = timer->lastSegment; timer->events.emplace_back(prev, e); timer->lastSegment = seg;
  }
}

void Timer::register_distribute_kernel(cudaStream_t stream, void* timer) { register_event<segment::dist_kernel>(stream, (EventTimer*)timer); }
void Timer::register_replicate_kernel(cudaStream_t stream, void* timer) { register_event<segment::rep_kernel>(stream, (EventTimer*)timer); }
void Timer::register_comm(cudaStream_t stream, void* timer) { register_event<segment::comm>(stream, (EventTimer*)timer); }

extern "C" void hyacinSync_TimerSegments(const hyacinHandle_t* handle, double* eventMs, int32_t lenMs) {
  if (handle->timer == nullptr) { CUDA_CHECK(cudaStreamSynchronize(handle->cudaStream)); return; }
  EventTimer* t = (EventTimer*)(handle->timer);
  int32_t len = int32_t(t->events.size());
  cudaEvent_t e; CUDA_CHECK(cudaEventCreate(&e));
  CUDA_CHECK(cudaEventRecord(e, handle->cudaStream));
  CUDA_CHECK(cudaEventSynchronize(e));
  t->events.emplace_back(t->lastSegment, e);

  if (eventMs != nullptr && 0 < lenMs) {
    double d_time = 0., r_time = 0., c_time = 0.;
    if (len) for (auto [seg, event] : t->events) {
      float ms = 0.f;
      if (seg == segment::dist_kernel) { CUDA_CHECK(cudaEventElapsedTime(&ms, e, event)); e = event; d_time += double(ms); } else
      if (seg == segment::rep_kernel) { CUDA_CHECK(cudaEventElapsedTime(&ms, e, event)); e = event; r_time += double(ms); } else
      if (seg == segment::comm) { CUDA_CHECK(cudaEventElapsedTime(&ms, e, event)); e = event; c_time += double(ms); } else { e = event; }
    }
    for (int32_t i = 0; i < lenMs; ++i) {
      char req = static_cast<char>(eventMs[i]);
      if (req == 'D' || req == 'd') { eventMs[i] = d_time; } else
      if (req == 'R' || req == 'r') { eventMs[i] = r_time; } else
      if (req == 'C' || req == 'c') { eventMs[i] = c_time; } else { eventMs[i] = 0.; }
    }
  }
  for (auto [seg, event] : t->events) { CUDA_CHECK(cudaEventDestroy(event)); } t->events.clear(); t->lastSegment = segment::none;
}
