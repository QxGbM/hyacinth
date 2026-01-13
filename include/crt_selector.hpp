
#include <crt_constants.hpp>

namespace CRT {
  constexpr uint64_t modular(int32_t iter) { return 0 <= iter && iter < 3 ? Common::mo[iter] : uint64_t(0); }
  constexpr uint64_t rem_e32(int32_t iter) { return 0 <= iter && iter < 3 ? Common::rem_e32[iter] : uint64_t(0); }
  constexpr uint64_t rem_e63(int32_t iter) { return 0 <= iter && iter < 3 ? Common::rem_e63[iter] : uint64_t(0); }

  constexpr int32_t active_moduli(int32_t n_moduli, int32_t iter) {
    n_moduli = n_moduli - (iter * 8);
    return 8 < n_moduli ? 8 : (n_moduli < 0 ? 0 : n_moduli);
  }

  constexpr int32_t order_pd(int32_t n_moduli) {
    switch(n_moduli) {
      case 2: return Moduli2::order_pd;
      case 3: return Moduli3::order_pd;
      case 4: return Moduli4::order_pd;
      case 5: return Moduli5::order_pd;
      case 6: return Moduli6::order_pd;
      case 7: return Moduli7::order_pd;
      case 8: return Moduli8::order_pd;
      case 9: return Moduli9::order_pd;
      case 10: return Moduli10::order_pd;
      case 11: return Moduli11::order_pd;
      case 12: return Moduli12::order_pd;
      case 13: return Moduli13::order_pd;
      case 14: return Moduli14::order_pd;
      case 15: return Moduli15::order_pd;
      case 16: return Moduli16::order_pd;
      case 17: return Moduli17::order_pd;
      case 18: return Moduli18::order_pd;
      case 19: return Moduli19::order_pd;
      case 20: return Moduli20::order_pd;
      case 21: return Moduli21::order_pd;
      case 22: return Moduli22::order_pd;
      case 23: return Moduli23::order_pd;
      default: return 0;
    }
  }

  constexpr uint64_t modular_inv(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return 0 <= iter && iter < 1 ? Moduli2::minv[iter] : uint64_t(0);
      case 3: return 0 <= iter && iter < 1 ? Moduli3::minv[iter] : uint64_t(0);
      case 4: return 0 <= iter && iter < 1 ? Moduli4::minv[iter] : uint64_t(0);
      case 5: return 0 <= iter && iter < 1 ? Moduli5::minv[iter] : uint64_t(0);
      case 6: return 0 <= iter && iter < 1 ? Moduli6::minv[iter] : uint64_t(0);
      case 7: return 0 <= iter && iter < 1 ? Moduli7::minv[iter] : uint64_t(0);
      case 8: return 0 <= iter && iter < 1 ? Moduli8::minv[iter] : uint64_t(0);
      case 9: return 0 <= iter && iter < 2 ? Moduli9::minv[iter] : uint64_t(0);
      case 10: return 0 <= iter && iter < 2 ? Moduli10::minv[iter] : uint64_t(0);
      case 11: return 0 <= iter && iter < 2 ? Moduli11::minv[iter] : uint64_t(0);
      case 12: return 0 <= iter && iter < 2 ? Moduli12::minv[iter] : uint64_t(0);
      case 13: return 0 <= iter && iter < 2 ? Moduli13::minv[iter] : uint64_t(0);
      case 14: return 0 <= iter && iter < 2 ? Moduli14::minv[iter] : uint64_t(0);
      case 15: return 0 <= iter && iter < 2 ? Moduli15::minv[iter] : uint64_t(0);
      case 16: return 0 <= iter && iter < 2 ? Moduli16::minv[iter] : uint64_t(0);
      case 17: return 0 <= iter && iter < 3 ? Moduli17::minv[iter] : uint64_t(0);
      case 18: return 0 <= iter && iter < 3 ? Moduli18::minv[iter] : uint64_t(0);
      case 19: return 0 <= iter && iter < 3 ? Moduli19::minv[iter] : uint64_t(0);
      case 20: return 0 <= iter && iter < 3 ? Moduli20::minv[iter] : uint64_t(0);
      case 21: return 0 <= iter && iter < 3 ? Moduli21::minv[iter] : uint64_t(0);
      case 22: return 0 <= iter && iter < 3 ? Moduli22::minv[iter] : uint64_t(0);
      case 23: return 0 <= iter && iter < 3 ? Moduli23::minv[iter] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

  constexpr const int32_t* p_div(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return iter == 0 ? &Moduli2::pd1[0] : nullptr;
      case 3: return iter == 0 ? &Moduli3::pd1[0] : nullptr;
      case 4: return iter == 0 ? &Moduli4::pd1[0] : nullptr;
      case 5: return iter == 0 ? &Moduli5::pd1[0] : nullptr;
      case 6: return iter == 0 ? &Moduli6::pd1[0] : nullptr;
      case 7: return iter == 0 ? &Moduli7::pd1[0] : nullptr;
      case 8: return iter == 0 ? &Moduli8::pd1[0] : nullptr;
      case 9: return iter == 0 ? &Moduli9::pd1[0] : (iter == 1 ? &Moduli9::pd2[0] : nullptr);
      case 10: return iter == 0 ? &Moduli10::pd1[0] : (iter == 1 ? &Moduli10::pd2[0] : nullptr);
      case 11: return iter == 0 ? &Moduli11::pd1[0] : (iter == 1 ? &Moduli11::pd2[0] : nullptr);
      case 12: return iter == 0 ? &Moduli12::pd1[0] : (iter == 1 ? &Moduli12::pd2[0] : nullptr);
      case 13: return iter == 0 ? &Moduli13::pd1[0] : (iter == 1 ? &Moduli13::pd2[0] : nullptr);
      case 14: return iter == 0 ? &Moduli14::pd1[0] : (iter == 1 ? &Moduli14::pd2[0] : nullptr);
      case 15: return iter == 0 ? &Moduli15::pd1[0] : (iter == 1 ? &Moduli15::pd2[0] : nullptr);
      case 16: return iter == 0 ? &Moduli16::pd1[0] : (iter == 1 ? &Moduli16::pd2[0] : nullptr);
      case 17: return iter == 0 ? &Moduli17::pd1[0] : (iter == 1 ? &Moduli17::pd2[0] : (iter == 2 ? &Moduli17::pd3[0] : nullptr));
      case 18: return iter == 0 ? &Moduli18::pd1[0] : (iter == 1 ? &Moduli18::pd2[0] : (iter == 2 ? &Moduli18::pd3[0] : nullptr));
      case 19: return iter == 0 ? &Moduli19::pd1[0] : (iter == 1 ? &Moduli19::pd2[0] : (iter == 2 ? &Moduli19::pd3[0] : nullptr));
      case 20: return iter == 0 ? &Moduli20::pd1[0] : (iter == 1 ? &Moduli20::pd2[0] : (iter == 2 ? &Moduli20::pd3[0] : nullptr));
      case 21: return iter == 0 ? &Moduli21::pd1[0] : (iter == 1 ? &Moduli21::pd2[0] : (iter == 2 ? &Moduli21::pd3[0] : nullptr));
      case 22: return iter == 0 ? &Moduli22::pd1[0] : (iter == 1 ? &Moduli22::pd2[0] : (iter == 2 ? &Moduli22::pd3[0] : nullptr));
      case 23: return iter == 0 ? &Moduli23::pd1[0] : (iter == 1 ? &Moduli23::pd2[0] : (iter == 2 ? &Moduli23::pd3[0] : nullptr));
      default: return nullptr;
    }
  }

  constexpr uint64_t domain_p(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return 0 <= iter && iter < 1 ? Moduli2::p[iter] : uint64_t(0);
      case 3: return 0 <= iter && iter < 1 ? Moduli3::p[iter] : uint64_t(0);
      case 4: return 0 <= iter && iter < 1 ? Moduli4::p[iter] : uint64_t(0);
      case 5: return 0 <= iter && iter < 1 ? Moduli5::p[iter] : uint64_t(0);
      case 6: return 0 <= iter && iter < 1 ? Moduli6::p[iter] : uint64_t(0);
      case 7: return 0 <= iter && iter < 1 ? Moduli7::p[iter] : uint64_t(0);
      case 8: return 0 <= iter && iter < 2 ? Moduli8::p[iter] : uint64_t(0);
      case 9: return 0 <= iter && iter < 2 ? Moduli9::p[iter] : uint64_t(0);
      case 10: return 0 <= iter && iter < 2 ? Moduli10::p[iter] : uint64_t(0);
      case 11: return 0 <= iter && iter < 2 ? Moduli11::p[iter] : uint64_t(0);
      case 12: return 0 <= iter && iter < 2 ? Moduli12::p[iter] : uint64_t(0);
      case 13: return 0 <= iter && iter < 2 ? Moduli13::p[iter] : uint64_t(0);
      case 14: return 0 <= iter && iter < 2 ? Moduli14::p[iter] : uint64_t(0);
      case 15: return 0 <= iter && iter < 2 ? Moduli15::p[iter] : uint64_t(0);
      case 16: return 0 <= iter && iter < 2 ? Moduli16::p[iter] : uint64_t(0);
      case 17: return 0 <= iter && iter < 3 ? Moduli17::p[iter] : uint64_t(0);
      case 18: return 0 <= iter && iter < 3 ? Moduli18::p[iter] : uint64_t(0);
      case 19: return 0 <= iter && iter < 3 ? Moduli19::p[iter] : uint64_t(0);
      case 20: return 0 <= iter && iter < 3 ? Moduli20::p[iter] : uint64_t(0);
      case 21: return 0 <= iter && iter < 3 ? Moduli21::p[iter] : uint64_t(0);
      case 22: return 0 <= iter && iter < 3 ? Moduli22::p[iter] : uint64_t(0);
      case 23: return 0 <= iter && iter < 3 ? Moduli23::p[iter] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

};
