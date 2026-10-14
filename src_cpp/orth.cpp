
#include <hyacin.h>
#include <internal.hpp>
#include <stdexcept>

extern "C" void hyacinXorth(const hyacinHandle_t* handle, int32_t M, int32_t N, int32_t K, hyacinPrecision_t Atype, const void* A, int32_t lda, void* Q, int32_t ldq, void* R, int32_t ldr) {
  if ((M <= 0) || (N <= 0)) { return; }
  Timer::register_distribute_kernel(handle->cudaStream, handle->timer);
  switch (Atype) {
    case HYACIN_F64: return;
    case HYACIN_F32: return;
    case HYACIN_F16: return;
    case HYACIN_F64_COMPLEX: return;
    case HYACIN_F32_COMPLEX: return;
    case HYACIN_F16_COMPLEX: return;
    default: return;
  }
}
