
from math import gcd
from itertools import combinations

moduli_all = [256, 253, 251, 249, 247, 241, 239, 235, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_all), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli[{i}]={a}, moduli[{j}]={b}) = {g}")
    all_ok = False

def pack_u8x8(a):
  b = []
  n = len(a)
  for i in range(0, n, 8):
    v = 0
    for j in range(0, 8):
      if i + j < n:
        v |= (a[i + j] & 255) << (8 * j)
    b.append(v)
  return b

def extract_limbs(x, limbs, sft):
  while x:
    limbs.append(x & ((1 << sft) - 1))
    x >>= sft
  return limbs

def pd_print(suffix, pd):
  limbs = []
  for x in pd:
    extract_limbs(x, limbs, 31)
  print(f"  constexpr int32_t pd{suffix}[{len(limbs)}] =" + " { " + ", ".join(f"{l}" for l in limbs) + " };")

if all_ok:
  moduli_simd = pack_u8x8(moduli_all)
  rem_e32 = pack_u8x8([(1 << 32) % m for m in moduli_all])
  rem_e63 = pack_u8x8([(1 << 63) % m for m in moduli_all])

  print("#pragma once\n#include <cstdint>\n")
  print("namespace CRT::Common {")
  print(f"  constexpr uint64_t mo[{len(moduli_simd)}] =" + " { " + ", ".join(f"{m}llu" for m in moduli_simd) + " };")
  print(f"  constexpr uint64_t rem_e32[{len(rem_e32)}] =" + " { " + ", ".join(f"{m}llu" for m in rem_e32) + " };")
  print(f"  constexpr uint64_t rem_e63[{len(rem_e63)}] =" + " { " + ", ".join(f"{m}llu" for m in rem_e63) + " };")
  print("};\n")

  for n in range(2, len(moduli_all)+1):
    moduli = moduli_all[:n:]
    P = 1
    for m in moduli:
      P *= m

    p_limbs = []
    extract_limbs(P, p_limbs, 63)
    P_div = [P // m for m in moduli]
    inv = [pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)]
    rem_e32 = [(i * (-(1 << 32))) % m for i, m in zip(inv, moduli)]

    inv = pack_u8x8(inv)
    rem_e32 = pack_u8x8(rem_e32)

    print(f"namespace CRT::Moduli{n}" + " {")
    print(f"  constexpr uint64_t minv[{len(inv)}] =" + " { " + ", ".join(f"{m}llu" for m in inv) + " };")
    print(f"  constexpr uint64_t rem_e32[{len(rem_e32)}] =" + " { " + ", ".join(f"{m}llu" for m in rem_e32) + " };")
    print(f"  constexpr int64_t p[{len(p_limbs)}] =" + " { " + ", ".join(f"{l}ll" for l in p_limbs) + " };")
    for i in range(0, n, 8):
      pd_print((i//8)+1, P_div[i:min(i+8, n):])
    print("};\n")
