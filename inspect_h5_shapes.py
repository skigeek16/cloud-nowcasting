# inspect_h5_shapes.py
import h5py
import sys
import numpy as np

def print_h5_structure(name, obj):
    """Prints the name and shape of datasets in the HDF5 file."""
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}  |  Shape: {obj.shape}  |  Data Type: {obj.dtype}")

if len(sys.argv) < 2:
    print("Usage: python3 inspect_h5_shapes.py <FOLDER>")
    sys.exit(1)

file_path = sys.argv[1]
with h5py.File(file_path, 'r') as f:
    print(f"--- Structure of {file_path} ---")
    f.visititems(print_h5_structure)
    print("---------------------------------")