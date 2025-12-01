"""
Check npz file contents (params.npz or da3_output.npz).
"""
import argparse
import numpy as np
from pathlib import Path


def check_npz(npz_path: str):
    """Read and display contents of an npz file."""
    path = Path(npz_path)
    
    if not path.exists():
        print(f"File not found: {path}")
        return
    
    print(f"Reading: {path}")
    print("=" * 60)
    
    data = np.load(path)
    print(f"Keys: {list(data.keys())}")
    
    for key in data.keys():
        val = data[key]
        print(f"\n{key}:")
        if hasattr(val, 'shape'):
            print(f"  Shape: {val.shape}")
            print(f"  Dtype: {val.dtype}")
            
            # For small arrays, print the full value
            if val.size <= 20:
                print(f"  Value: {val}")
            else:
                print(f"  Min: {val.min():.6f}, Max: {val.max():.6f}, Mean: {val.mean():.6f}")
                if val.ndim == 1:
                    print(f"  First 5: {val[:5]}")
                elif val.ndim == 2:
                    print(f"  First row: {val[0]}")
        else:
            print(f"  Value: {val}")


def main():
    parser = argparse.ArgumentParser(description="Check npz file contents")
    parser.add_argument("npz_path", type=str, help="Path to npz file")
    args = parser.parse_args()
    
    check_npz(args.npz_path)


if __name__ == "__main__":
    main()
