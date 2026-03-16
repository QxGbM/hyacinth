
import argparse
import numpy as np

parser = argparse.ArgumentParser(description=("Generate an MxN test matrix"))
parser.add_argument("M", type=int, nargs='?', default=2048, help="Number of rows (default: 2048)")
parser.add_argument("N", type=int, nargs='?', default=2048, help="Number of columns (default: 2048)")
parser.add_argument("file", type=str, nargs='?', default="matrix.csv", help="Output CSV file name (default: matrix.csv)")
parser.add_argument("data", type=str, nargs='?', default="D", help="Output CSV format (default: D)")
parser.add_argument("omega", type=float, nargs='?', default=1., help="Kernel omega (default: 1.0)")
args = parser.parse_args()

# indices
i = np.arange(args.M)
j = np.arange(args.N)

# grid coordinates
xi = i // 128
yi = i - 128 * xi

xj = j // 128
yj = j - 128 * xj

# broadcast to matrix form
diff_x = xi[:, None] - (-1 - xj[None, :])
diff_y = yi[:, None] - yj[None, :]

# distance
r = np.sqrt(diff_x * diff_x + diff_y * diff_y)

# Helmholtz kernel
if args.data == "Z" or args.data == "C":
  A = np.exp(1j * args.omega * r) / r
  np.savetxt(args.file, A, delimiter=',', newline='\n')

if args.data == "D" or args.data == "S":
  A = np.cos(args.omega * r) / r
  np.savetxt(args.file, A, delimiter=',', newline='\n')