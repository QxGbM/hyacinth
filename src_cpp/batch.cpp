
#include <internal.hpp>
#include <crt_constants.hpp>
#include <tuple>
#include <map>
#include <vector>
#include <string>
#include <charconv>
#include <stdexcept>

class BatchArgs {
private:
  int32_t batchMaxK;
  std::map<int32_t, std::tuple<int32_t, int32_t, int32_t, char>> tensor;
  // tuple is [Tensor Order, Tensor Panel Prefix, Tensor Rows, Algorithm 'L' or 'C']

public:
  BatchArgs(int32_t K, const char alg[], const int32_t u[], int32_t Nbatches) : batchMaxK(K), tensor() {
    for (int32_t i = 0; i < Nbatches; ++i) {
      char algi = alg[i]; int32_t ui = u[i];
      int32_t order = internal::int8::gram_algorithm(algi, batchMaxK, ui);
      tensor.insert(std::make_pair(ui, std::make_tuple(order, 0, 0, algi)));
    }

    if (1 < int32_t(tensor.size()))
      for (auto iter = tensor.begin(); std::next(iter) != tensor.end(); iter = std::next(iter))
        std::get<1>(std::next(iter)->second) = std::get<0>(iter->second) + std::get<1>(iter->second);
  }

  BatchArgs(int32_t K, const std::string& str) : batchMaxK(K), tensor() {
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

    if (1 < int32_t(tensor.size()))
      for (auto iter = tensor.begin(); std::next(iter) != tensor.end(); iter = std::next(iter))
        std::get<1>(std::next(iter)->second) = std::get<0>(iter->second) + std::get<1>(iter->second);
  }

  int32_t total_order() const {
    if (tensor.size()) {
      auto last = std::prev(tensor.end());
      return std::get<0>(last->second) + std::get<1>(last->second);
    } else { return 0; }
  }

  // op: 'E'=eager eval; 'Z'=lazy batch; 'F'=lazy+flush; 'S' = skip
  void processA(int32_t M, int32_t uc, char alg, char& op, std::tuple<int32_t, int32_t, int32_t, char>& param) {
    if (uc <= 0) { op = 'S'; param = std::make_tuple(0, 0, 0, 'S'); } else {
      auto iter = tensor.lower_bound(uc);
      if (iter == tensor.end() || batchMaxK < M) {
        op = 'E'; int32_t order = internal::int8::gram_algorithm(alg, M, uc);
        param = std::make_tuple(order, 0, 0, alg);
      } else {
        param = iter->second; int32_t rows = std::get<2>(param);
        if (M <= batchMaxK - rows) { op = 'Z'; std::get<2>(iter->second) = rows + M; }
          else { op = 'F'; std::get<2>(iter->second) = M; }
      }
    }
  }

  void flush(std::vector<std::tuple<int32_t, int32_t, int32_t, char>>& list) {
    for (auto& [key, value] : tensor) { if (std::get<2>(value)) { list.emplace_back(value); std::get<2>(value) = 0; }}
  }

};
