
#include <hyacin.h>
#include <internal.hpp>
#include <crt_constants.hpp>
#include <charconv>
#include <stdexcept>

Batch::BatchArgs::BatchArgs(int32_t& K, const std::string& str, int32_t& order, int32_t& count_crt) : batchMaxK((K + 255) & (~255)), tensor() {
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

  K = batchMaxK; order = count_crt = 0;
  if (1 < int32_t(tensor.size()))
    for (auto& [key, value] : tensor) {
      std::get<1>(value) = order; std::get<2>(value) = count_crt; 
      order += std::get<0>(value); count_crt += int32_t(std::get<4>(value) == 'C');
    }
}

void Batch::BatchArgs::processA(int32_t M, int32_t uc, char alg, char& op, std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, char>& param) {
  if (uc <= 0 || M <= 0) { op = 'S'; param = std::make_tuple(0, 0, 0, 0, 0, 'S'); } else {
    auto iter = tensor.lower_bound(uc);
    if (iter == tensor.end() || batchMaxK < M) {
      op = 'E'; int32_t order = internal::int8::gram_algorithm(alg, M, uc);
      param = std::make_tuple(uc, order, 0, 0, 0, alg);
    } else {
      param = std::tuple_cat(std::tuple{iter->first}, iter->second); int32_t rows = std::get<3>(iter->second);
      if (M <= batchMaxK - rows) { op = 'Z'; std::get<3>(iter->second) = rows + M; }
        else { op = 'F'; std::get<3>(iter->second) = M; }
    }
  }
}

void Batch::BatchArgs::flush(std::vector<std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, char>>& list) {
  for (auto& [key, value] : tensor) { if (std::get<3>(value)) { list.emplace_back(std::tuple_cat(std::tuple{key}, value)); std::get<3>(value) = 0; }}
}

extern "C" void hyacinXherkBatchCreate(void** param, const char config[], int32_t* batchK, int32_t* panels, int32_t* CRTcounts) {
  std::string str(config); *param = new Batch::BatchArgs(*batchK, str, *panels, *CRTcounts);
}

extern "C" void hyacinXherkBatchDestroy(void* param) {
  if (param) { delete reinterpret_cast<Batch::BatchArgs*>(param); }
}
