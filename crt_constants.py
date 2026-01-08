
from math import gcd
from itertools import combinations

moduli_all = [256, 253, 251, 249, 247, 241, 239, 235, 233, 229, 227, 223, 217, 211, 199, 197, 193, 191, 181, 179, 173, 167, 163]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_all), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli[{i}]={a}, moduli[{j}]={b}) = {g}")
    all_ok = False

def pack_u8x4(a):
  b = []
  n = len(a)
  for i in range(0, n, 4):
    v = 0
    if i < n:
      v |= (a[i] & 255)
    if i + 1 < n:
      v |= (a[i + 1] & 255) << 8
    if i + 2 < n:
      v |= (a[i + 2] & 255) << 16
    if i + 3 < n:
      v |= (a[i + 3] & 255) << 24
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
  moduli_simd = pack_u8x4(moduli_all)
  rem_e32 = pack_u8x4([(1 << 32) % m for m in moduli_all])

  print("#pragma once\n#include <cstdint>\n")
  print("namespace CRT::Common {")
  print(f"  constexpr uint32_t mo[{len(moduli_simd)}] =" + " { " + ", ".join(f"{m}u" for m in moduli_simd) + " };")
  print(f"  constexpr uint32_t rem_e32[{len(rem_e32)}] =" + " { " + ", ".join(f"{m}u" for m in rem_e32) + " };")
  print("};\n")

  for n in range(2, len(moduli_all)+1):
    order_pd = (((n - 1) * 8) + 30) // 31
    moduli = moduli_all[:n:]
    P = 1
    for m in moduli:
      P *= m

    p_limbs = []
    extract_limbs(P, p_limbs, 63)
    P_div = [P // m for m in moduli]
    inv = pack_u8x4([pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)])

    print(f"namespace CRT::Moduli{n}" + " {")
    print(f"  const int32_t order_p = {len(p_limbs)}, order_pd = {order_pd};")
    print(f"  constexpr uint32_t minv[{len(inv)}] =" + " { " + ", ".join(f"{m}u" for m in inv) + " };")
    print(f"  constexpr uint64_t p[{len(p_limbs)}] =" + " { " + ", ".join(f"{l}llu" for l in p_limbs) + " };")
    for i in range(0, n, 4):
      pd_print((i//4)+1, P_div[i:min(i+4, n):])
    print("};\n")
