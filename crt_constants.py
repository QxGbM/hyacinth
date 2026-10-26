
from math import gcd
from itertools import combinations
import numpy as np

moduli_real = [256, 253, 251, 249, 247, 241, 239, 235, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163, 157]
moduli_complex = [241, 233, 229, 221, 205, 197, 193, 181, 173, 157, 149, 137, 113, 109, 101, 97, 89, 73, 61, 53, 256, 253, 251, 249]
moduli_complex_imag = [64, 89, 107, 21, 32, 14, 81, 19, 80, 28, 44, 37, 15, 33, 10, 22, 34, 27, 11, 23, 0, 0, 0, 0]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_real), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli_real[{i}]={a}, moduli_real[{j}]={b}) = {g}")
    all_ok = False

for (i, a), (j, b) in combinations(enumerate(moduli_complex), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli_complex[{i}]={a}, moduli_complex[{j}]={b}) = {g}")
    all_ok = False

for i, m in zip(moduli_complex_imag, moduli_complex):
  if (0 < i and (1 + i**2) % m != 0):
    print(f"{i}^2 mod {m} != -1")
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
  rem_e32 = [(1 << 32) % m for m in moduli_real]
  rem_e63 = [(1 << 63) % m for m in moduli_real]
  ranges = np.trunc(np.cumsum(np.log2(moduli_real))).astype(np.int32)

  print("#pragma once\n#include <cstdint>\nnamespace R8CRT {\n")
  print("  template <int32_t Moduli> struct Constants;")
  print(f"  constexpr uint16_t range[{len(moduli_real)}] =" + " { " + ", ".join(f"{m}" for m in ranges) + " };")
  print(f"  constexpr uint16_t mo[{len(moduli_real)}] =" + " { " + ", ".join(f"{m}" for m in moduli_real) + " };")
  print(f"  constexpr uint16_t rem_e32[{len(moduli_real)}] =" + " { " + ", ".join(f"{m}" for m in rem_e32) + " };")
  print(f"  constexpr uint16_t rem_e63[{len(moduli_real)}] =" + " { " + ", ".join(f"{m}" for m in rem_e63) + " };\n")

  for n in range(2, len(moduli_real)+1):
    moduli = moduli_real[:n:]
    P = 1
    for m in moduli:
      P *= m

    p_limbs = []
    extract_limbs(P, p_limbs, 63, 3)
    P_div = [P // m for m in moduli]
    inv = [pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)]

    print(f"  template<> struct Constants<{n}>" + " {")
    print(f"    static constexpr uint16_t minv[{n}] =" + " { " + ", ".join(f"{m}" for m in inv) + " };")
    print(f"    static constexpr int64_t p[{len(p_limbs)}] =" + " { " + ", ".join(f"{l}ll" for l in p_limbs) + " };")
    pd_print(P_div, 31)
    print("  };\n")
  print("};\n")

  rem_e32 = [(1 << 32) % m for m in moduli_complex]
  rem_e63 = [(1 << 63) % m for m in moduli_complex]
  ranges = np.trunc(np.cumsum(np.log2(moduli_complex))).astype(np.int32)

  print("namespace C8CRT {\n")
  print("  template <int32_t Moduli> struct Constants;")
  print(f"  constexpr uint16_t range[{len(moduli_complex)}] =" + " { " + ", ".join(f"{m}" for m in ranges) + " };")
  print(f"  constexpr uint16_t mo[{len(moduli_complex)}] =" + " { " + ", ".join(f"{m}" for m in moduli_complex) + " };")
  print(f"  constexpr uint16_t im[{len(moduli_complex)}] =" + " { " + ", ".join(f"{m}" for m in moduli_complex_imag) + " };")
  print(f"  constexpr uint16_t rem_e32[{len(moduli_complex)}] =" + " { " + ", ".join(f"{m}" for m in rem_e32) + " };")
  print(f"  constexpr uint16_t rem_e63[{len(moduli_complex)}] =" + " { " + ", ".join(f"{m}" for m in rem_e63) + " };\n")

  for n in range(2, len(moduli_complex)+1):
    moduli = moduli_complex[:n:]
    P = 1
    for m in moduli:
      P *= m

    p_limbs = []
    extract_limbs(P, p_limbs, 63, 3)
    P_div = [P // m for m in moduli]
    inv = [pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)]

    print(f"  template<> struct Constants<{n}>" + " {")
    print(f"    static constexpr uint16_t minv[{n}] =" + " { " + ", ".join(f"{m}" for m in inv) + " };")
    print(f"    static constexpr int64_t p[{len(p_limbs)}] =" + " { " + ", ".join(f"{l}ll" for l in p_limbs) + " };")
    pd_print(P_div, 31)
    print("  };\n")
  print("};\n")
