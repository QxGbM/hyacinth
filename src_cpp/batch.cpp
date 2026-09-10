
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <charconv>
#include <stdexcept>

Batch::BatchArgs::BatchArgs(int32_t K, int32_t Complex, int32_t elemBytes, const std::string& str, int32_t& order) :
  tensor(), batchMaxK((K + 255) & (~255)), Complex(Complex) {
  std::vector<char> alg; std::vector<int32_t> u;
  const char* p = str.data(), *end = p + str.size();
  while (p != end && std::isdigit(static_cast<unsigned char>(*p))) { ++p; }
  while (p != end) {
    while (p != end && !std::isdigit(static_cast<unsigned char>(*p))) { ++p; }
    if (p != end) {
      const char key = p[-1], *iBegin = p;
      while (p != end && std::isdigit(static_cast<unsigned char>(*p))) { ++p; }
      int32_t value; auto [ptr, ec] = std::from_chars(iBegin, p, value);
      if (std::isalpha(static_cast<unsigned char>(key)) && ec == std::errc{} && ptr == p)
      { alg.emplace_back(key); u.emplace_back(value); }
    }
  }

  for (int32_t i = 0; i < int32_t(alg.size()); ++i) {
    char algi = alg[i]; int32_t ui = u[i];
    int32_t order = internal::int8::gram_algorithm(algi, batchMaxK, ui);
    tensor.insert(std::make_pair(ui, std::make_tuple(order, 0, 0, algi)));
  }

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
