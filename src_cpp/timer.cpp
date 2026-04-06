
#include <hyacin.h>
#include <internal.hpp>
#include <vector>

#ifdef BUILTIN_TIMER
std::vector<cudaEvent_t> events;
enum class segment { none, kernel, comm };
segment last_seg = segment::none;

void Timer::register_kernel(cudaStream_t stream) {
  if (last_seg != segment::kernel) {
    last_seg = segment::kernel; events.emplace_back(); cudaEvent_t* end = &events[events.size() - 1];
    cudaEventCreate(end); cudaEventRecord(*end, stream);
  }
}

void Timer::register_comm(cudaStream_t stream) {
  if (last_seg != segment::comm) {
    last_seg = segment::comm; events.emplace_back(); cudaEvent_t* end = &events[events.size() - 1];
    cudaEventCreate(end); cudaEventRecord(*end, stream);
  }
}

extern "C" void hyacinSync_TimerSegments(cudaStream_t stream, double* kernel_time, double* comm_time) {
  double k_time = 0., c_time = 0.;
  int32_t len = int32_t(events.size()); segment seg = last_seg;
  if (len) {
    events.emplace_back(); cudaEvent_t* end = &events[len];
    cudaEventCreate(end); cudaEventRecord(*end, stream); cudaEventSynchronize(*end);
    for (int32_t i = len - 1; 0 <= i; --i) {
      float milliseconds = 0.f;
      cudaEventElapsedTime(&milliseconds, events[i], events[i + 1]);
      if (seg == segment::kernel) { k_time += double(milliseconds); seg = segment::comm; }
        else if (seg == segment::comm) { c_time += double(milliseconds); seg = segment::kernel; }
    }
  }

  for (cudaEvent_t e : events)
    cudaEventDestroy(e);
  events.clear(); last_seg = segment::none;
  if (kernel_time) { *kernel_time += k_time; } if (comm_time) { *comm_time += c_time; }
}

#else

extern "C" void hyacinSync_TimerSegments(cudaStream_t stream, double* kernel_time, double* comm_time) {
  cudaStreamSynchronize(stream);
  if (kernel_time) { *kernel_time = 0.; } if (comm_time) { *comm_time = 0.; }
}

#endif

