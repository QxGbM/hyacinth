
from math import gcd
from itertools import combinations

moduli_all = [65280, 65279, 65273, 65269, 65267, 65261, 65257, 65251, 65249, 65243, 65239, 65237, 65231, 65227]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_all), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli[{i}]={a}, moduli[{j}]={b}) = {g}")
    all_ok = False

def pack_u16x4(a):
  b = []
  n = len(a)
  for i in range(0, n, 4):
    v = 0
    if i < n:
      v |= a[i]
    if i + 1 < n:
      v |= a[i + 1] << 16
    if i + 2 < n:
      v |= a[i + 2] << 32
    if i + 3 < n:
      v |= a[i + 3] << 48
    b.append(v)
  return b

def extract_limbs(x, limbs, sft):
  while x:
    limbs.append(x & ((1 << sft) - 1))
    x >>= sft
  return limbs

def p_limbs_print(x, order):
  limbs = []
  extract_limbs(x, limbs, 63)
  print(f"  constexpr uint64_t p[{order}] =" + " { " + ", ".join(f"{l}llu" for l in limbs) + " };")

def pd_print(suffix, pd):
  limbs = []
  for x in pd:
    extract_limbs(x, limbs, 31)
  print(f"  constexpr int32_t pd{suffix}[{len(limbs)}] =" + " { " + ", ".join(f"{l}" for l in limbs) + " };")

if all_ok:
  moduli_simd = pack_u16x4(moduli_all)
  rem_e32 = pack_u16x4([(1 << 32) % m for m in moduli_all])

  print("#pragma once\n#include <cstdint>\n")
  print("namespace CRT::Common {")
  print(f"  constexpr uint64_t mo[{len(moduli_simd)}] =" + " { " + ", ".join(f"{m}llu" for m in moduli_simd) + " };")
  print(f"  constexpr uint64_t rem_e32[{len(rem_e32)}] =" + " { " + ", ".join(f"{m}llu" for m in rem_e32) + " };")
  print("};\n")

  for n in range(2, len(moduli_all)+1):
    order_p = ((n * 16) + 63) // 63
    order_pd = ((n * 16) + 14) // 31
    print(f"namespace CRT::Moduli{n}" + " {")
    print(f"  const int32_t order_p = {order_p}, order_pd = {order_pd};")
    moduli = moduli_all[:n:]
    P = 1
    for m in moduli:
      P *= m

    P_div = [P // m for m in moduli]
    inv = pack_u16x4([pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)])

    print(f"  constexpr uint64_t minv[{len(inv)}] =" + " { " + ", ".join(f"{m}llu" for m in inv) + " };")
    p_limbs_print(P, order_p)
    for i in range(0, n, 4):
      pd_print((i//4)+1, P_div[i:min(i+4, n):])
    print("};\n")
