
#include <hyacin.h>
#include <internal.hpp>
#include <vector>

#ifdef BUILTIN_TIMER
std::vector<cudaEvent_t> events;
int32_t last_seg = 0;

void Timer::register_kernel(cudaStream_t stream) {
  if (last_seg != 1) {
    last_seg = 1; events.emplace_back(); cudaEvent_t* end = &events[events.size() - 1];
    cudaEventCreate(end); cudaEventRecord(*end, stream);
  }
}

void Timer::register_comm(cudaStream_t stream) {
  if (last_seg != 2) {
    last_seg = 2; events.emplace_back(); cudaEvent_t* end = &events[events.size() - 1];
    cudaEventCreate(end); cudaEventRecord(*end, stream);
  }
}

extern "C" void hyacinSync_TimerSegments(cudaStream_t stream, double* kernel_time, double* comm_time) {
  double k_time = 0., c_time = 0.;
  int32_t len = int32_t(events.size()), seg = last_seg;
  if (len) {
    events.emplace_back(); cudaEvent_t* end = &events[len];
    cudaEventCreate(end); cudaEventRecord(*end, stream); cudaEventSynchronize(*end);
    for (int32_t i = len - 1; 0 <= i; --i) {
      float milliseconds = 0.f;
      cudaEventElapsedTime(&milliseconds, events[i], events[i + 1]);
      if (seg == 1) { k_time += double(milliseconds); seg = 2; }
        else if (seg == 2) { c_time += double(milliseconds); seg = 1; }
    }
  }

  for (cudaEvent_t e : events)
    cudaEventDestroy(e);
  events.clear(); last_seg = 0;
  if (kernel_time) { *kernel_time = k_time; } if (comm_time) { *comm_time = c_time; }
}

#else

extern "C" void hyacinSync_TimerSegments(cudaStream_t stream, double* kernel_time, double* comm_time) {
  cudaStreamSynchronize(stream);
  if (kernel_time) { *kernel_time = 0.; } if (comm_time) { *comm_time = 0.; }
}

#endif

