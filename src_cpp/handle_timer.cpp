
#include <hyacin.h>
#include <internal.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

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

Batch::BatchArgs::BatchArgs(cudaStream_t stream, cudaMemPool_t mempool, char algo, int32_t u_ceil, int32_t K, int32_t N, int32_t Complex, int32_t elemBytes) : tensor(), batchMaxK((std::max(0, K) + 255) & (~255)) {
  int32_t u_floor = Complex;
  while (u_floor <= u_ceil) {
    char algi = algo; int32_t ui = u_floor;
    int32_t orderA = internal::int8::gram_algorithm(algi, batchMaxK, ui); u_floor = 1 + ui;
    if (algi == 'L') { tensor.insert(std::make_pair(ui, std::make_tuple(orderA, 0, 0, algi))); }
  }

  int32_t orderA = internal::int8::gram_algorithm(algo, batchMaxK, u_ceil);
  tensor.insert(std::make_pair(u_ceil, std::make_tuple(orderA, 0, 0, algo)));
  int32_t order = 0; int32_t panelsLimb = Complex ? 3 : 1;
  for (auto& [key, value] : tensor)
  { std::get<1>(value) = order; order += (std::get<3>(value) == 'C') ? elemBytes : (panelsLimb * std::get<0>(value)); }

  uint64_t bytes = uint64_t(N) * uint64_t(batchMaxK) * uint64_t(order);
  if (cudaSuccess != cudaMallocFromPoolAsync((void**)&data, bytes, mempool, stream))
    throw std::runtime_error("Batch Workspace allocation failed");
}

std::tuple<int32_t, int32_t, int32_t, int8_t*, char> Batch::BatchArgs::processA(int32_t M, int32_t N, int32_t uc, char alg, char& op) {
  if (uc < 0 || M <= 0) { op = 'S'; return std::make_tuple(uc, 0, 0, data, alg); } else {
    auto iter = tensor.lower_bound(uc);
    if (iter == tensor.end() || batchMaxK < M) { op = 'E'; return std::make_tuple(uc, 0, 0, data, alg); } else {
      int32_t rows = std::get<2>(iter->second);
      if (M <= batchMaxK - rows) { op = 'Z'; std::get<2>(iter->second) = rows + M; }
        else { op = 'F'; std::get<2>(iter->second) = M; }
      int64_t prefix_elem = int64_t(N) * int64_t(batchMaxK) * int64_t(std::get<1>(iter->second));
      return std::make_tuple(iter->first, std::get<0>(iter->second), rows, &data[prefix_elem], std::get<3>(iter->second));
    }
  }
}

void Batch::BatchArgs::flush(int32_t N, std::vector<std::tuple<int32_t, int32_t, int32_t, int8_t*, char>>& list) {
  int64_t stride = int64_t(N) * int64_t(batchMaxK);
  for (auto& [key, value] : tensor) if (std::get<2>(value)) {
    int64_t prefix_elem = stride * int64_t(std::get<1>(value));
    list.emplace_back(key, std::get<0>(value), std::get<2>(value), &data[prefix_elem], std::get<3>(value)); std::get<2>(value) = 0;
  }
}

void Batch::BatchArgs::free_data(cudaStream_t stream) {
  if (data) { cudaFreeAsync(data, stream); }
}

extern "C" void hyacinXherkBatchDestroy(hyacinHandle_t handle, void* param) {
  if (param) { reinterpret_cast<Batch::BatchArgs*>(param)->free_data(handle.cudaStream); delete reinterpret_cast<Batch::BatchArgs*>(param); }
}

enum class segment { none, dist_kernel, rep_kernel, comm };
struct EventTimer {
  std::vector<std::pair<segment, cudaEvent_t>> events;
  segment lastSegment = segment::none;
};

extern "C" void hyacinCreate(hyacinHandle_t* handle, int32_t create_timer) {
  cudaStreamCreateWithFlags(&handle->cudaStream, cudaStreamNonBlocking);
  cublasCreate(&handle->cublasHandle);
  cublasSetStream(handle->cublasHandle, handle->cudaStream);
  cusolverDnCreate(&handle->cusolverHandle);
  cusolverDnSetStream(handle->cusolverHandle, handle->cudaStream);
  cusolverDnCreateParams(&handle->cusolverParams);
  cudaMemPoolProps props = cudaMemPoolProps(); cudaGetDevice(&props.location.id);
  props.allocType = cudaMemAllocationTypePinned; props.location.type = cudaMemLocationTypeDevice;
  cudaMemPoolCreate(&handle->mempool, &props);
  uint64_t threshold = std::numeric_limits<uint64_t>::max();
  cudaMemPoolSetAttribute(handle->mempool, cudaMemPoolAttrReleaseThreshold, &threshold);
  cudaMallocHost(&handle->pinnedWorkspace, size_t(128));
#ifndef NO_NCCL
  handle->col_comm = handle->row_comm = nullptr;
#endif
  handle->timer = create_timer ? (new EventTimer()) : nullptr;
}

#ifndef NO_NCCL
extern "C" void hyacinCreate2D(hyacinHandle_t* handle, ncclComm_t col_comm, ncclComm_t row_comm, int32_t create_timer) {
  hyacinCreate(handle, create_timer);
  handle->col_comm = col_comm;
  handle->row_comm = row_comm;
}
#endif

extern "C" void hyacinDestroy(hyacinHandle_t handle) {
  cudaStreamDestroy(handle.cudaStream);
  cublasDestroy(handle.cublasHandle);
  cusolverDnDestroy(handle.cusolverHandle);
  cusolverDnDestroyParams(handle.cusolverParams);
  cudaMemPoolDestroy(handle.mempool);
  cudaFreeHost(handle.pinnedWorkspace);
  if (handle.timer) { delete (EventTimer*)(handle.timer); }
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

extern "C" void hyacinSync_TimerSegments(hyacinHandle_t handle, double* eventMs, int32_t lenMs) {
  if (handle.timer == nullptr || eventMs == nullptr || lenMs <= 0) 
  { cudaStreamSynchronize(handle.cudaStream); return; }

  double d_time = 0., r_time = 0., c_time = 0.;
  int32_t len = int32_t(((EventTimer*)handle.timer)->events.size());
  cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, handle.cudaStream); cudaEventSynchronize(e);
  ((EventTimer*)handle.timer)->events.emplace_back(((EventTimer*)handle.timer)->lastSegment, e);

  if (len) for (auto [seg, event] : ((EventTimer*)handle.timer)->events) {
    float milliseconds = 0.f;
    if (seg == segment::dist_kernel) { cudaEventElapsedTime(&milliseconds, e, event); e = event; d_time += double(milliseconds); } else
    if (seg == segment::rep_kernel) { cudaEventElapsedTime(&milliseconds, e, event); e = event; r_time += double(milliseconds); } else
    if (seg == segment::comm) { cudaEventElapsedTime(&milliseconds, e, event); e = event; c_time += double(milliseconds); } else
    { e = event; }
  }

  for (auto [seg, event] : ((EventTimer*)handle.timer)->events) { cudaEventDestroy(event); }
  ((EventTimer*)handle.timer)->events.clear(); ((EventTimer*)handle.timer)->lastSegment = segment::none;
  for (int32_t i = 0; i < lenMs; ++i) {
    char req = static_cast<char>(eventMs[i]);
    if (req == 'D' || req == 'd') { eventMs[i] = d_time; } else
    if (req == 'R' || req == 'r') { eventMs[i] = r_time; } else
    if (req == 'C' || req == 'c') { eventMs[i] = c_time; }
  }
}
