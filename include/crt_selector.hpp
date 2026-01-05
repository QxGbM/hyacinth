
#include <crt_constants.hpp>

namespace CRT {
  constexpr uint64_t modular(uint32_t frag) { return frag < uint32_t(4) ? Common::mo[frag] : uint64_t(0); }
  constexpr uint64_t rem_e32(uint32_t frag) { return frag < uint32_t(4) ? Common::rem_e32[frag] : uint64_t(0); }

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

  constexpr uint64_t modular_inv(int32_t n_moduli, uint32_t frag) {
    switch(n_moduli) {
      case 2: return frag < uint32_t(1) ? Moduli2::minv[frag] : uint64_t(0);
      case 3: return frag < uint32_t(1) ? Moduli3::minv[frag] : uint64_t(0);
      case 4: return frag < uint32_t(1) ? Moduli4::minv[frag] : uint64_t(0);
      case 5: return frag < uint32_t(2) ? Moduli5::minv[frag] : uint64_t(0);
      case 6: return frag < uint32_t(2) ? Moduli6::minv[frag] : uint64_t(0);
      case 7: return frag < uint32_t(2) ? Moduli7::minv[frag] : uint64_t(0);
      case 8: return frag < uint32_t(2) ? Moduli8::minv[frag] : uint64_t(0);
      case 9: return frag < uint32_t(3) ? Moduli9::minv[frag] : uint64_t(0);
      case 10: return frag < uint32_t(3) ? Moduli10::minv[frag] : uint64_t(0);
      case 11: return frag < uint32_t(3) ? Moduli11::minv[frag] : uint64_t(0);
      case 12: return frag < uint32_t(3) ? Moduli12::minv[frag] : uint64_t(0);
      case 13: return frag < uint32_t(4) ? Moduli13::minv[frag] : uint64_t(0);
      case 14: return frag < uint32_t(4) ? Moduli14::minv[frag] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

  constexpr const int32_t* p_div(int32_t n_moduli, uint32_t frag) {
    switch(n_moduli) {
      case 2: return frag == uint32_t(0) ? &Moduli2::pd1[0] : nullptr;
      case 3: return frag == uint32_t(0) ? &Moduli3::pd1[0] : nullptr;
      case 4: return frag == uint32_t(0) ? &Moduli4::pd1[0] : nullptr;
      case 5: return frag == uint32_t(0) ? &Moduli5::pd1[0] : (frag == uint32_t(1) ? &Moduli5::pd2[0] : nullptr);
      case 6: return frag == uint32_t(0) ? &Moduli6::pd1[0] : (frag == uint32_t(1) ? &Moduli6::pd2[0] : nullptr);
      case 7: return frag == uint32_t(0) ? &Moduli7::pd1[0] : (frag == uint32_t(1) ? &Moduli7::pd2[0] : nullptr);
      case 8: return frag == uint32_t(0) ? &Moduli8::pd1[0] : (frag == uint32_t(1) ? &Moduli8::pd2[0] : nullptr);
      case 9: return frag == uint32_t(0) ? &Moduli9::pd1[0] : (frag == uint32_t(1) ? &Moduli9::pd2[0] : (frag == uint32_t(2) ? &Moduli9::pd3[0] : nullptr));
      case 10: return frag == uint32_t(0) ? &Moduli10::pd1[0] : (frag == uint32_t(1) ? &Moduli10::pd2[0] : (frag == uint32_t(2) ? &Moduli10::pd3[0] : nullptr));
      case 11: return frag == uint32_t(0) ? &Moduli11::pd1[0] : (frag == uint32_t(1) ? &Moduli11::pd2[0] : (frag == uint32_t(2) ? &Moduli11::pd3[0] : nullptr));
      case 12: return frag == uint32_t(0) ? &Moduli12::pd1[0] : (frag == uint32_t(1) ? &Moduli12::pd2[0] : (frag == uint32_t(2) ? &Moduli12::pd3[0] : nullptr));
      case 13: return frag == uint32_t(0) ? &Moduli13::pd1[0] : (frag == uint32_t(1) ? &Moduli13::pd2[0] : (frag == uint32_t(2) ? &Moduli13::pd3[0] : (frag == uint32_t(3) ? &Moduli13::pd4[0] : nullptr)));
      case 14: return frag == uint32_t(0) ? &Moduli14::pd1[0] : (frag == uint32_t(1) ? &Moduli14::pd2[0] : (frag == uint32_t(2) ? &Moduli14::pd3[0] : (frag == uint32_t(3) ? &Moduli14::pd4[0] : nullptr)));
      default: return nullptr;
    }
  }

  constexpr uint64_t domain_p(int32_t n_moduli, uint32_t frag) {
    switch(n_moduli) {
      case 2: return frag < uint32_t(1) ? Moduli2::p[frag] : uint64_t(0);
      case 3: return frag < uint32_t(1) ? Moduli3::p[frag] : uint64_t(0);
      case 4: return frag < uint32_t(1) ? Moduli4::p[frag] : uint64_t(0);
      case 5: return frag < uint32_t(2) ? Moduli5::p[frag] : uint64_t(0);
      case 6: return frag < uint32_t(2) ? Moduli6::p[frag] : uint64_t(0);
      case 7: return frag < uint32_t(2) ? Moduli7::p[frag] : uint64_t(0);
      case 8: return frag < uint32_t(2) ? Moduli8::p[frag] : uint64_t(0);
      case 9: return frag < uint32_t(3) ? Moduli9::p[frag] : uint64_t(0);
      case 10: return frag < uint32_t(3) ? Moduli10::p[frag] : uint64_t(0);
      case 11: return frag < uint32_t(3) ? Moduli11::p[frag] : uint64_t(0);
      case 12: return frag < uint32_t(3) ? Moduli12::p[frag] : uint64_t(0);
      case 13: return frag < uint32_t(4) ? Moduli13::p[frag] : uint64_t(0);
      case 14: return frag < uint32_t(4) ? Moduli14::p[frag] : uint64_t(0);
      default: return uint64_t(0);
    }
  }

};
