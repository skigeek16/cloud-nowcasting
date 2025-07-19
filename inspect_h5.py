# inspect_h5.py
import h5py
import sys

def print_h5_structure(name, obj):
    """Prints the name and type of objects in the HDF5 file."""
    print(name)

if len(sys.argv) < 2:
    print("Usage: python3 inspect_h5.py <path_to_your_h5_file>")
    sys.exit(1)

file_path = sys.argv[1]
with h5py.File(file_path, 'r') as f:
    print(f"--- Structure of {file_path} ---")
    f.visititems(print_h5_structure)
    print("---------------------------------")