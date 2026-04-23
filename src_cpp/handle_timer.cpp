
#include <hyacin.h>
#include <internal.hpp>
#include <vector>

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
  cudaMallocHost(&handle->pinnedWorkspace, size_t(8192));
  handle->timer = create_timer ? (new EventTimer()) : nullptr;
}

extern "C" void hyacinDestroy(hyacinHandle_t handle) {
  cudaStreamDestroy(handle.cudaStream);
  cublasDestroy(handle.cublasHandle);
  cusolverDnDestroy(handle.cusolverHandle);
  cusolverDnDestroyParams(handle.cusolverParams);
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

