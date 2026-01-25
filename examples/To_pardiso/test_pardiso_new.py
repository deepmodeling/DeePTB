"""
Test script for the new to_pardiso_new method with JSON format.

This script demonstrates:
1. Using to_pardiso_new to export data
2. Comparing old vs new file formats
3. Verifying JSON structure
"""

import os
import sys
import json
import numpy as np

# Add DeePTB to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.getcwd())))
from dptb.postprocess.unified.system import TBSystem

def test_to_pardiso_new():
    """Test the new to_pardiso_new method."""

    # Setup paths (relative to this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = script_dir
    model_path = os.path.join(root_dir, "nnsk.iter_ovp0.000.pth")
    struct_path = os.path.join(root_dir, "min.vasp")

    # Initialize TBSystem
    print("="*70)
    print("Initializing TBSystem...")
    print("="*70)
    tbsys = TBSystem(data=struct_path, calculator=model_path)
    print(f"System Info: {tbsys.atoms}")
    print()

    # Export using NEW method (integrated into to_pardiso)
    output_dir_new = os.path.join(root_dir, "output_new")
    print("="*70)
    print("Exporting with to_pardiso (JSON + legacy format)...")
    print("="*70)
    # Using the new method explicitely
    tbsys.to_pardiso_new(output_dir=output_dir_new)
    print()

    # Verify JSON structure
    print("="*70)
    print("Verifying JSON structure...")
    print("="*70)
    json_path = os.path.join(output_dir_new, "structure.json")

    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"JSON keys: {list(data.keys())}")
    print(f"  - natoms: {data['structure']['nsites']}")
    print(f"  - norbits: {data['basis_info']['total_orbitals']}")
    print(f"  - spinful: {data['basis_info'].get('spinful', 'Not found')}")
    print(f"  - Formula: {data['structure']['chemical_formula']}")
    print(f"  - basis: {data['basis_info']['basis']}")
    print()

    # Compare file sizes
    print("="*70)
    print("File comparison:")
    print("="*70)

    files_new = os.listdir(output_dir_new)
    print(f"Exported files ({len(files_new)}):")
    for fname in sorted(files_new):
        fpath = os.path.join(output_dir_new, fname)
        size = os.path.getsize(fpath)
        print(f"  - {fname}: {size:,} bytes")
        
    print()
    print("="*70)
    print("Running New Julia Backend...")
    print("="*70)
    
    import subprocess
    julia_script = os.path.abspath(os.path.join(root_dir, "../../dptb/postprocess/pardiso/main.jl"))
    config_path = os.path.join(root_dir, "band.json")
    julia_out_dir = os.path.join(output_dir_new, "results") # New Output Paths
    
    cmd = [
        "julia",
        julia_script,
        "--input_dir", output_dir_new,
        "--output_dir", julia_out_dir,
        "--config", config_path
    ]
    
    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print("Julia backend run successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Julia execution failed with code {e.returncode}")
    except FileNotFoundError:
        print("Julia executable not found. Skipping execution.")


if __name__ == "__main__":
    test_to_pardiso_new()
