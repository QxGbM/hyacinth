
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <charconv>
#include <stdexcept>

Batch::BatchArgs::BatchArgs(char algo, int32_t u_ceil, int32_t u_floor, int32_t min_uinc, int32_t K, int32_t Complex, int32_t elemBytes, int32_t& order) :
  tensor(), batchMaxK((K + 255) & (~255)) {
  min_uinc = std::max(1, min_uinc);
  while (u_floor <= u_ceil) {
    char algi = algo; int32_t ui = u_floor;
    int32_t orderA = internal::int8::gram_algorithm(algi, batchMaxK, ui);
    if (algi == 'L') { tensor.insert(std::make_pair(ui, std::make_tuple(orderA, 0, 0, algi))); }
    u_floor = std::max(1 + ui, u_floor + min_uinc);
  }

  int32_t orderA = internal::int8::gram_algorithm(algo, batchMaxK, u_ceil);
  tensor.insert(std::make_pair(u_ceil, std::make_tuple(orderA, 0, 0, algo)));
  order = 0; int32_t panelsLimb = Complex ? 3 : 1;
  if (1 < int32_t(tensor.size()))
    for (auto& [key, value] : tensor)
    { std::get<1>(value) = order; order += (std::get<3>(value) == 'C') ? elemBytes : (panelsLimb * std::get<0>(value)); }
}

std::tuple<int32_t, int32_t, int64_t, int32_t, char> Batch::BatchArgs::processA(int32_t M, int32_t N, int32_t uc, char alg, char& op) {
  if (uc < 0 || M <= 0) { op = 'S'; return std::make_tuple(uc, 0, int64_t(0), 0, alg); } else {
    auto iter = tensor.lower_bound(uc);
    if (iter == tensor.end() || batchMaxK < M) { op = 'E'; return std::make_tuple(uc, 0, int64_t(0), 0, alg); } else {
      int32_t rows = std::get<2>(iter->second);
      if (M <= batchMaxK - rows) { op = 'Z'; std::get<2>(iter->second) = rows + M; }
        else { op = 'F'; std::get<2>(iter->second) = M; }
      int64_t prefix_elem = int64_t(N) * int64_t(batchMaxK) * int64_t(std::get<1>(iter->second));
      return std::make_tuple(iter->first, std::get<0>(iter->second), prefix_elem, rows, std::get<3>(iter->second));
    }
  }
}

void Batch::BatchArgs::flush(int32_t N, std::vector<std::tuple<int32_t, int32_t, int64_t, int32_t, char>>& list) {
  int64_t stride = int64_t(N) * int64_t(batchMaxK);
  for (auto& [key, value] : tensor) if (std::get<2>(value)) {
    int64_t prefix_elem = stride * int64_t(std::get<1>(value));
    list.emplace_back(key, std::get<0>(value), prefix_elem, std::get<2>(value), std::get<3>(value)); std::get<2>(value) = 0;
  }
}
