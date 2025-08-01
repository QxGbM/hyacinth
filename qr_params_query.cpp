
#include <hyacinth.hpp>
#include <internal.hpp>

void device::QR::dgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N) {
  epi = -std::log2(epi);

  params->M = M;
  params->N = N;
  params->algnM = (M + 15) & (~15);
  params->algnN = (N + 15) & (~15);
  params->order = std::max(1, std::min(internal::int8::order_max * 4, 1 + int32_t(std::ceil(epi / internal::int8::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 4 : (params->acc_bits <= 53 ? 8 : 16);
  params->use_fp64_over_32 = 1;

  params->n_i8 = size_t(params->algnM) * size_t(params->algnN) * size_t(params->order);
  params->n_i32 = 2 * size_t(params->algnN) * size_t(params->algnN) * size_t(params->order) + ((N + 63) & (~63));
  params->n_elem = size_t(params->algnN) * (N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::sgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N) {
  epi = -std::log2f(epi);

  params->M = M;
  params->N = N;
  params->algnM = (M + 15) & (~15);
  params->algnN = (N + 15) & (~15);
  params->order = std::max(1, std::min(internal::int8::order_max * 4, 1 + int32_t(std::ceil(epi / internal::int8::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 4 : 8;
  params->use_fp64_over_32 = 1;

  params->n_i8 = size_t(params->algnM) * size_t(params->algnN) * size_t(params->order);
  params->n_i32 = 2 * size_t(params->algnN) * size_t(params->algnN) * size_t(params->order) + ((N + 63) & (~63));
  params->n_elem = size_t(params->algnN) * (N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::zgeqp3_ronly_params_query(geqp3_params* params, double epi, int32_t M, int32_t N) {
  epi = -std::log2(epi);

  params->M = M;
  params->N = N;
  params->algnM = (M + 15) & (~15);
  params->algnN = (N + 15) & (~15);
  params->order = std::max(1, std::min(internal::int8::order_max * 4, 1 + int32_t(std::ceil(epi / internal::int8::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 8 : (params->acc_bits <= 53 ? 16 : 32);
  params->use_fp64_over_32 = 1;

  params->n_i8 = 2 * size_t(params->algnM) * size_t(params->algnN) * size_t(params->order);
  params->n_i32 = 4 * size_t(params->algnN) * size_t(params->algnN) * size_t(params->order) + ((N + 63) & (~63));
  params->n_elem = size_t(params->algnN) * (N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::cgeqp3_ronly_params_query(geqp3_params* params, float epi, int32_t M, int32_t N) {
  epi = -std::log2f(epi);

  params->M = M;
  params->N = N;
  params->algnM = (M + 15) & (~15);
  params->algnN = (N + 15) & (~15);
  params->order = std::max(1, std::min(internal::int8::order_max * 4, 1 + int32_t(std::ceil(epi / internal::int8::exp_base))));
  params->acc_bits = std::max(1, int32_t(std::ceil(2 * epi)));
  params->elem_bytes = params->acc_bits <= 24 ? 8 : 16;
  params->use_fp64_over_32 = 1;

  params->n_i8 = 2 * size_t(params->algnM) * size_t(params->algnN) * size_t(params->order);
  params->n_i32 = 4 * size_t(params->algnN) * size_t(params->algnN) * size_t(params->order) + ((N + 63) & (~63));
  params->n_elem = size_t(params->algnN) * (N + 1);
  params->work_bytes = params->n_i8 + params->n_i32 * 4 + params->n_elem * params->elem_bytes;
}

void device::QR::set_double_double_as_fp128(geqp3_params* params) {
  params->use_fp64_over_32 = 1;
}

void device::QR::set_quad_float_as_fp128(geqp3_params* params) {
  params->use_fp64_over_32 = 0;
}
