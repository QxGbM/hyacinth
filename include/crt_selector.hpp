
#include <crt_constants.hpp>

namespace CRT {
  constexpr uint64_t modular(int32_t iter) { return iter < 4 ? Common::mo[iter] : uint64_t(0); }
  constexpr uint64_t rem_e32(int32_t iter) { return iter < 4 ? Common::rem_e32[iter] : uint64_t(0); }

  constexpr int32_t active_moduli(int32_t n_moduli, int32_t iter) {
    n_moduli = n_moduli - (iter * 4);
    return 4 < n_moduli ? 4 : n_moduli;
  }

  constexpr int32_t order_p(int32_t n_moduli) {
    switch(n_moduli) {
      case 2: return Moduli2::order_p;
      case 3: return Moduli3::order_p;
      case 4: return Moduli4::order_p;
      case 5: return Moduli5::order_p;
      case 6: return Moduli6::order_p;
      case 7: return Moduli7::order_p;
      case 8: return Moduli8::order_p;
      case 9: return Moduli9::order_p;
      case 10: return Moduli10::order_p;
      case 11: return Moduli11::order_p;
      case 12: return Moduli12::order_p;
      case 13: return Moduli13::order_p;
      case 14: return Moduli14::order_p;
      default: return 0;
    }
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
      default: return 0;
    }
  }

  constexpr uint64_t modular_inv(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return iter < 1 ? Moduli2::minv[iter] : uint64_t(0);
      case 3: return iter < 1 ? Moduli3::minv[iter] : uint64_t(0);
      case 4: return iter < 1 ? Moduli4::minv[iter] : uint64_t(0);
      case 5: return iter < 2 ? Moduli5::minv[iter] : uint64_t(0);
      case 6: return iter < 2 ? Moduli6::minv[iter] : uint64_t(0);
      case 7: return iter < 2 ? Moduli7::minv[iter] : uint64_t(0);
      case 8: return iter < 2 ? Moduli8::minv[iter] : uint64_t(0);
      case 9: return iter < 3 ? Moduli9::minv[iter] : uint64_t(0);
      case 10: return iter < 3 ? Moduli10::minv[iter] : uint64_t(0);
      case 11: return iter < 3 ? Moduli11::minv[iter] : uint64_t(0);
      case 12: return iter < 3 ? Moduli12::minv[iter] : uint64_t(0);
      case 13: return iter < 4 ? Moduli13::minv[iter] : uint64_t(0);
      case 14: return iter < 4 ? Moduli14::minv[iter] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

  constexpr const int32_t* p_div(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return iter == uint32_t(0) ? &Moduli2::pd1[0] : nullptr;
      case 3: return iter == uint32_t(0) ? &Moduli3::pd1[0] : nullptr;
      case 4: return iter == uint32_t(0) ? &Moduli4::pd1[0] : nullptr;
      case 5: return iter == uint32_t(0) ? &Moduli5::pd1[0] : (iter == 1 ? &Moduli5::pd2[0] : nullptr);
      case 6: return iter == uint32_t(0) ? &Moduli6::pd1[0] : (iter == 1 ? &Moduli6::pd2[0] : nullptr);
      case 7: return iter == uint32_t(0) ? &Moduli7::pd1[0] : (iter == 1 ? &Moduli7::pd2[0] : nullptr);
      case 8: return iter == uint32_t(0) ? &Moduli8::pd1[0] : (iter == 1 ? &Moduli8::pd2[0] : nullptr);
      case 9: return iter == uint32_t(0) ? &Moduli9::pd1[0] : (iter == 1 ? &Moduli9::pd2[0] : (iter == 2 ? &Moduli9::pd3[0] : nullptr));
      case 10: return iter == uint32_t(0) ? &Moduli10::pd1[0] : (iter == 1 ? &Moduli10::pd2[0] : (iter == 2 ? &Moduli10::pd3[0] : nullptr));
      case 11: return iter == uint32_t(0) ? &Moduli11::pd1[0] : (iter == 1 ? &Moduli11::pd2[0] : (iter == 2 ? &Moduli11::pd3[0] : nullptr));
      case 12: return iter == uint32_t(0) ? &Moduli12::pd1[0] : (iter == 1 ? &Moduli12::pd2[0] : (iter == 2 ? &Moduli12::pd3[0] : nullptr));
      case 13: return iter == uint32_t(0) ? &Moduli13::pd1[0] : (iter == 1 ? &Moduli13::pd2[0] : (iter == 2 ? &Moduli13::pd3[0] : (iter == 3 ? &Moduli13::pd4[0] : nullptr)));
      case 14: return iter == uint32_t(0) ? &Moduli14::pd1[0] : (iter == 1 ? &Moduli14::pd2[0] : (iter == 2 ? &Moduli14::pd3[0] : (iter == 3 ? &Moduli14::pd4[0] : nullptr)));
      default: return nullptr;
    }
  }

  constexpr uint64_t domain_p(int32_t n_moduli, int32_t iter) {
    switch(n_moduli) {
      case 2: return iter < 1 ? Moduli2::p[iter] : uint64_t(0);
      case 3: return iter < 1 ? Moduli3::p[iter] : uint64_t(0);
      case 4: return iter < 1 ? Moduli4::p[iter] : uint64_t(0);
      case 5: return iter < 2 ? Moduli5::p[iter] : uint64_t(0);
      case 6: return iter < 2 ? Moduli6::p[iter] : uint64_t(0);
      case 7: return iter < 2 ? Moduli7::p[iter] : uint64_t(0);
      case 8: return iter < 2 ? Moduli8::p[iter] : uint64_t(0);
      case 9: return iter < 3 ? Moduli9::p[iter] : uint64_t(0);
      case 10: return iter < 3 ? Moduli10::p[iter] : uint64_t(0);
      case 11: return iter < 3 ? Moduli11::p[iter] : uint64_t(0);
      case 12: return iter < 3 ? Moduli12::p[iter] : uint64_t(0);
      case 13: return iter < 4 ? Moduli13::p[iter] : uint64_t(0);
      case 14: return iter < 4 ? Moduli14::p[iter] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

};
