
#include <hyacin.h>
#include <internal.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>

int32_t device_sms = 0, device_f64_capable = 0;
const std::vector<int32_t> f64_capable_sm_list({ 800, 900, 1000 }); // sm80,sm90,sm100
inline void device_params() {
  int32_t device, major, minor; cudaGetDevice(&device);
  cudaDeviceGetAttribute(&device_sms, cudaDevAttrMultiProcessorCount, device);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  device_f64_capable = int32_t(f64_capable_sm_list.end() != std::find(f64_capable_sm_list.begin(), f64_capable_sm_list.end(), 100 * major + minor));
}

int32_t internal::device_is_f64_capable() {
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

enum class segment { none, kernel, comm };
struct EventTimer {
  std::vector<cudaEvent_t> events;
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

void Timer::register_kernel(cudaStream_t stream, void* timer) {
  if (timer)
    if (((EventTimer*)timer)->lastSegment != segment::kernel) {
      cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, stream);
      ((EventTimer*)timer)->lastSegment = segment::kernel;
      ((EventTimer*)timer)->events.emplace_back(e);
    }
}

void Timer::register_comm(cudaStream_t stream, void* timer) {
  if (timer)
    if (((EventTimer*)timer)->lastSegment != segment::comm) {
      cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, stream);
      ((EventTimer*)timer)->lastSegment = segment::comm;
      ((EventTimer*)timer)->events.emplace_back(e);
    }
}

extern "C" void hyacinSync_TimerSegments(hyacinHandle_t handle, double* kernelMs, double* commMs) {
  if (handle.timer == nullptr) 
  { cudaStreamSynchronize(handle.cudaStream); *kernelMs = *commMs = 0.; return; }

  double k_time = 0., c_time = 0.;
  int32_t len = int32_t(((EventTimer*)handle.timer)->events.size());
  cudaEvent_t e; cudaEventCreate(&e); cudaEventRecord(e, handle.cudaStream); cudaEventSynchronize(e);
  ((EventTimer*)handle.timer)->events.emplace_back(e);

  if (len) {
    segment seg = ((EventTimer*)handle.timer)->lastSegment;
    for (int32_t i = len - 1; 0 <= i; --i) {
      float milliseconds = 0.f;
      cudaEventElapsedTime(&milliseconds, ((EventTimer*)handle.timer)->events[i], ((EventTimer*)handle.timer)->events[i + 1]);
      if (seg == segment::kernel) { k_time += double(milliseconds); seg = segment::comm; }
        else if (seg == segment::comm) { c_time += double(milliseconds); seg = segment::kernel; }
    }
  }

  for (cudaEvent_t e : ((EventTimer*)handle.timer)->events)
    cudaEventDestroy(e);
  ((EventTimer*)handle.timer)->events.clear(); ((EventTimer*)handle.timer)->lastSegment = segment::none;
  if (kernelMs) { *kernelMs += k_time; } if (commMs) { *commMs += c_time; }
}
