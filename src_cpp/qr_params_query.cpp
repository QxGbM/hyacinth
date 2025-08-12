
#include <hyacinth.hpp>
#include <internal.hpp>
#include <limits>

void device::QR::dgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N) {
  epi = -std::log2(std::max(std::abs(epi), std::numeric_limits<double>::epsilon()));

  params->M = M;
  params->N = N;
  params->algnM = (M + 63) & (~63);
  params->algnN = (N + 63) & (~63);
  params->orderA = std::max(1, 1 + int32_t(std::ceil(epi / device::Config::exp_base)));
  params->orderC = std::max(1, std::min(params->orderA * 2, 1 + int32_t(std::ceil((2 * epi) / device::Config::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 4 : (params->acc_bits <= 53 ? 8 : 16);
  params->iter_k = 1024 << (14 - 2 * device::Config::exp_base);
  params->use_fp64_over_32 = 1;

  params->n_i8 = uint64_t(params->algnM) * uint64_t(N) * uint64_t(params->orderA);
  params->n_i8 = (params->n_i8 + uint64_t(255)) & (~uint64_t(255));
  params->n_i32 = uint64_t(params->algnN) * uint64_t(N) * uint64_t(params->orderC);
  params->n_elem = uint64_t(params->algnN) * uint64_t(N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::sgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N) {
  epi = -std::log2f(std::max(std::abs(epi), std::numeric_limits<float>::epsilon()));

  params->M = M;
  params->N = N;
  params->algnM = (M + 63) & (~63);
  params->algnN = (N + 63) & (~63);
  params->orderA = std::max(1, 1 + int32_t(std::ceil(epi / device::Config::exp_base)));
  params->orderC = std::max(1, std::min(params->orderA * 2, 1 + int32_t(std::ceil((2 * epi) / device::Config::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 4 : 8;
  params->iter_k = 1024 << (14 - 2 * device::Config::exp_base);
  params->use_fp64_over_32 = 1;

  params->n_i8 = uint64_t(params->algnM) * uint64_t(N) * uint64_t(params->orderA);
  params->n_i8 = (params->n_i8 + uint64_t(255)) & (~uint64_t(255));
  params->n_i32 = uint64_t(params->algnN) * uint64_t(N) * uint64_t(params->orderC);
  params->n_elem = uint64_t(params->algnN) * uint64_t(N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::zgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N) {
  epi = -std::log2(std::max(std::abs(epi), std::numeric_limits<double>::epsilon()));

  params->M = M;
  params->N = N;
  params->algnM = (M + 63) & (~63);
  params->algnN = (N + 63) & (~63);
  params->orderA = std::max(1, 1 + int32_t(std::ceil(epi / device::Config::exp_base)));
  params->orderC = std::max(1, std::min(params->orderA * 2, 1 + int32_t(std::ceil((2 * epi) / device::Config::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 8 : (params->acc_bits <= 53 ? 16 : 32);
  params->iter_k = 1024 << (14 - 2 * device::Config::exp_base);
  params->use_fp64_over_32 = 1;

  params->n_i8 = 2 * uint64_t(params->algnM) * uint64_t(N) * uint64_t(params->orderA);
  params->n_i8 = (params->n_i8 + uint64_t(255)) & (~uint64_t(255));
  params->n_i32 = 2 * uint64_t(params->algnN) * uint64_t(N) * uint64_t(params->orderC);
  params->n_elem = uint64_t(params->algnN) * uint64_t(N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::cgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N) {
  epi = -std::log2f(std::max(std::abs(epi), std::numeric_limits<float>::epsilon()));

  params->M = M;
  params->N = N;
  params->algnM = (M + 63) & (~63);
  params->algnN = (N + 63) & (~63);
  params->orderA = std::max(1, 1 + int32_t(std::ceil(epi / device::Config::exp_base)));
  params->orderC = std::max(1, std::min(params->orderA * 2, 1 + int32_t(std::ceil((2 * epi) / device::Config::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 8 : 16;
  params->iter_k = 1024 << (14 - 2 * device::Config::exp_base);
  params->use_fp64_over_32 = 1;

  params->n_i8 = 2 * uint64_t(params->algnM) * uint64_t(N) * uint64_t(params->orderA);
  params->n_i8 = (params->n_i8 + uint64_t(255)) & (~uint64_t(255));
  params->n_i32 = 2 * uint64_t(params->algnN) * uint64_t(N) * uint64_t(params->orderC);
  params->n_elem = uint64_t(params->algnN) * uint64_t(N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::set_double_double_as_fp128(geqp3_params* params) {
  params->use_fp64_over_32 = 1;
}

void device::QR::set_quad_float_as_fp128(geqp3_params* params) {
  params->use_fp64_over_32 = 0;
}
