
import argparse
import sys
import numpy as np

def load_matrix(filename: str) -> np.ndarray:
  """
  Load a CSV matrix that may contain either real or complex entries.

  Supported formats include:
    real:    %.18e,%.18e,...
    complex: %.18e+%.18ej or %.18e-%.18ej
  """
  try:
    # complex128 can also represent purely real values.
    m = np.loadtxt(filename, delimiter=',', dtype=np.complex128)
  except Exception as e:
    raise ValueError(f"Failed to load '{filename}': {e}")

  # Ensure at least 2D for consistent matrix handling.
  m = np.atleast_2d(m)
  return m


def relative_error(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
  """
  Compute element-wise relative error:
    |test - ref| / |ref|

  For entries where |ref| == 0, use absolute error instead:
    |test - ref|
  """
  abs_ref = np.abs(ref)
  abs_diff = np.abs(test - ref)

  err = np.empty_like(abs_diff, dtype=np.float64)

  zero_mask = (abs_ref == 0.0)
  nonzero_mask = ~zero_mask

  err[nonzero_mask] = abs_diff[nonzero_mask] / abs_ref[nonzero_mask]
  err[zero_mask] = abs_diff[zero_mask]

  return err

def main() -> int:
  parser = argparse.ArgumentParser(description="Bit-wise comparisons of matrices.")
  parser.add_argument("ref", type=str, nargs='?', default="ref.csv", help="Reference CSV file name (default: ref.csv)")
  parser.add_argument("test", type=str, nargs='?', default="test.csv", help="Test CSV file name (default: test.csv)")
  args = parser.parse_args()

  try:
    ref_m = load_matrix(args.ref)
    test_m = load_matrix(args.test)
  except ValueError as e:
    print(e, file=sys.stderr)
    return 1

  # Check if dims agree
  if ref_m.shape != test_m.shape:
    print(f"Dimension mismatch: ref shape = {ref_m.shape}, " + f"test shape = {test_m.shape}", file=sys.stderr)
    return 1

  # Compare element-wise relative error, print max, avg
  err = relative_error(ref_m, test_m)

  max_err = np.max(err)
  avg_err = np.mean(err)

  print(f"shape: {ref_m.shape}")
  print(f"max relative error: {max_err:.18e}")
  print(f"avg relative error: {avg_err:.18e}")
  return 0

if __name__ == "__main__":
  sys.exit(main())
