
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <charconv>
#include <stdexcept>

Batch::BatchArgs::BatchArgs(int32_t K, const std::string& str, int32_t& order, int32_t& count_crt) : tensor(), batchMaxK((K + 255) & (~255)) {
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
    tensor.insert(std::make_pair(ui, std::make_tuple(order, 0, 0, 0, algi)));
  }

  order = count_crt = 0;
  if (1 < int32_t(tensor.size()))
    for (auto& [key, value] : tensor) {
      std::get<1>(value) = order; std::get<2>(value) = count_crt; 
      order += std::get<0>(value); count_crt += int32_t(std::get<4>(value) == 'C');
    }
}

std::tuple<int32_t, int32_t, int64_t, int64_t, int32_t, char> Batch::BatchArgs::processA(int32_t M, int32_t N, int32_t uc, char alg, char& op) {
  if (uc <= 0 || M <= 0) { op = 'S'; return std::make_tuple(uc, 0, int64_t(0), int64_t(0), 0, alg); } else {
    auto iter = tensor.lower_bound(uc);
    if (iter == tensor.end() || batchMaxK < M) { op = 'E'; return std::make_tuple(uc, 0, int64_t(0), int64_t(0), 0, alg); } else {
      int64_t N64 = int64_t(N), stride = N64 * int64_t(batchMaxK);
      int64_t prefix_elem = stride * int64_t(std::get<1>(iter->second)), prefix_sum = N64 * int64_t(std::get<2>(iter->second)) * int64_t(2);
      int32_t order = std::get<0>(iter->second), rows = std::get<3>(iter->second);
      if (M <= batchMaxK - rows) { op = 'Z'; std::get<3>(iter->second) = rows + M; }
        else { op = 'F'; std::get<3>(iter->second) = M; }
      return std::make_tuple(iter->first, order, prefix_elem, prefix_sum, rows, std::get<4>(iter->second));
    }
  }
}

void Batch::BatchArgs::flush(int32_t N, std::vector<std::tuple<int32_t, int32_t, int64_t, int64_t, int32_t, char>>& list) {
  int64_t N64 = int64_t(N), stride = N64 * int64_t(batchMaxK);
  for (auto& [key, value] : tensor) if (std::get<3>(value)) {
    int64_t prefix_elem = stride * int64_t(std::get<1>(value)), prefix_sum = N64 * int64_t(std::get<2>(value));
    list.emplace_back(key, std::get<0>(value), prefix_elem, prefix_sum, std::get<3>(value), std::get<4>(value)); std::get<3>(value) = 0;
  }
}

extern "C" void hyacinXherkBatchCreate(void** param, const char config[], int32_t* batchK, int32_t* panels, int32_t* CRTcounts) {
  std::string str(config); *param = new Batch::BatchArgs(*batchK, str, *panels, *CRTcounts);
}

extern "C" void hyacinXherkBatchDestroy(void* param) {
  if (param) { delete reinterpret_cast<Batch::BatchArgs*>(param); }
}
