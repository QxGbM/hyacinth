
from math import gcd
from itertools import combinations

moduli_all = [256, 253, 251, 249, 247, 241, 239, 235, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_all), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli[{i}]={a}, moduli[{j}]={b}) = {g}")
    all_ok = False

def extract_limbs(x, limbs, sft, limbs_pad):
  while x:
    limbs.append(x & ((1 << sft) - 1))
    x >>= sft
    limbs_pad = limbs_pad - 1
  while 0 < limbs_pad:
    limbs.append(0)
    limbs_pad = limbs_pad - 1
  return limbs

def pd_print(pd, sft, limbs_pad = 6, chunk_size = 18):
  limbs = []
  for x in pd:
    extract_limbs(x, limbs, sft, limbs_pad)
  str = ",\n      ".join(", ".join(f"{l}" for l in limbs[i:i+chunk_size]) for i in range(0, len(limbs), chunk_size))
  print(f"    static constexpr int32_t pd[{len(limbs)}] =" + " {\n      " + str + " };")

if all_ok:
  rem_e32 = [(1 << 32) % m for m in moduli_all]
  rem_e63 = [(1 << 63) % m for m in moduli_all]
  padded_len = (len(moduli_all) + 7) & (~7)

  print("#pragma once\n#include <cstdint>\nnamespace U8CRT {\n")
  print("  template <int32_t Moduli> struct Constants;")
  print(f"  constexpr uint16_t mo[{padded_len}] =" + " { " + ", ".join(f"{m}" for m in moduli_all) + " };")
  print(f"  constexpr uint16_t rem_e32[{padded_len}] =" + " { " + ", ".join(f"{m}" for m in rem_e32) + " };")
  print(f"  constexpr uint16_t rem_e63[{padded_len}] =" + " { " + ", ".join(f"{m}" for m in rem_e63) + " };\n")

  for n in range(2, len(moduli_all)+1):
    moduli = moduli_all[:n:]
    P = 1
    for m in moduli:
      P *= m

    p_limbs = []
    extract_limbs(P, p_limbs, 63, 3)
    P_div = [P // m for m in moduli]
    inv = [pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)]
    rem_e32 = [(i * (-(1 << 32))) % m for i, m in zip(inv, moduli)]
    padded_len = (n + 7) & (~7)

    print(f"  template<> struct Constants<{n}>" + " {")
    print(f"    static constexpr uint16_t minv[{padded_len}] =" + " { " + ", ".join(f"{m}" for m in inv) + " };")
    print(f"    static constexpr uint16_t rem_e32[{padded_len}] =" + " { " + ", ".join(f"{m}" for m in rem_e32) + " };")
    print(f"    static constexpr int64_t p[{len(p_limbs)}] =" + " { " + ", ".join(f"{l}ll" for l in p_limbs) + " };")
    pd_print(P_div, 31)
    print("  };\n")
  print("};\n")
