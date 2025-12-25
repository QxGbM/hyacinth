
from math import gcd
from itertools import combinations

moduli_all = [65280, 65279, 65273, 65269, 65267, 65261, 65257, 65251, 65249, 65243, 65239, 65237, 65231, 65227]
all_ok = True

for (i, a), (j, b) in combinations(enumerate(moduli_all), 2):
  g = gcd(a, b)
  if g != 1:
    print(f"gcd(moduli[{i}]={a}, moduli[{j}]={b}) = {g}")
    all_ok = False

def p_limbs_print(x, order):
  limbs = []
  while x:
    limbs.append(x & ((1 << 63) - 1))
    x >>= 63
  print(f"  constexpr uint64_t p[{order}] =" + " { " + ", ".join(f"{l}llu" for l in limbs) + " };")

def pd_limbs_print(x, i, order):
  limbs = []
  while x:
    limbs.append(x & ((1 << 31) - 1))
    x >>= 31
  print(f"  constexpr int32_t pd{i}[{order}] =" + " { " + ", ".join(f"{l}" for l in limbs) + " };")

if all_ok:
  print("#pragma once\n#include <cstdint>\n")
  for n in range(2, len(moduli_all)+1):
    order_p = ((n * 16) + 62) // 63
    order_pd = ((n * 16) + 14) // 31
    print(f"namespace CRT::Moduli{n}" + " {")
    print(f"  const int32_t order_p = {order_p}, order_pd = {order_pd};")
    moduli = moduli_all[:n:]
    P = 1
    for m in moduli:
      P *= m

    P_div = [P // m for m in moduli]
    inv = [pow(Pd % m, -1, m) for Pd, m in zip(P_div, moduli)]

    print(f"  constexpr uint16_t mo[{n}] =" + " { " + ", ".join(f"{m}" for m in moduli) + " };")
    print(f"  constexpr uint16_t minv[{n}] =" + " { " + ", ".join(f"{m}" for m in inv) + " };")

    for i, v in enumerate(P_div):
      pd_limbs_print(v, i+1, order_pd)
    p_limbs_print(P, order_p)
    print("};\n")
