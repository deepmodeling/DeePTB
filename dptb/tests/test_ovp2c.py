
import pytest
import os
import torch
import numpy as np

op2c = pytest.importorskip("op2c")

from dptb.data.build import build_dataset
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.postprocess.ovp2c import compute_overlap
from pathlib import Path

# Path to the data directory contains info.json and orbital file
DATA_ROOT = os.path.join(Path(os.path.abspath(__file__)).parent, "data/e3_band/data/")
ORB_FILE = "Si_gga_7au_100Ry_2s2p1d.orb"

def test_ovp2c_silicon():
    # 1. Setup Data Loading
    common_options = {
        "basis": {
            "Si": "2s2p1d"
        },
        "device": "cpu",
        "dtype": "float64", # Use double precision for comparison
        "overlap": True,
        "seed": 42
    }
    
    # r_max roughly 7.0 bohr = 3.7 Angstrom. 
    # Providing a generous cutoff to ensure all pairs are found.
    data_options = {
        "r_max": 7.5, 
        "train": {
            "root": DATA_ROOT,
            "prefix": "Si64"
        }
    }

    # Load dataset
    # Note: build_dataset might return a concatenation of all subfolders if multiple exist?
    # Here only Si64.0 should be found if it follows patterns.
    # Actually build_dataset might look for a specific structure.
    dataset = build_dataset(**data_options, **data_options["train"], **common_options)
    assert len(dataset) > 0, "Dataset should not be empty"
    
    data = dataset[0] # Get first frame (AtomicData object)
    
    # Store reference overlap (loaded from overlaps.h5)
    # The build_dataset handles loading and converting blocks to features.
    # So AtomicDataDict.EDGE_OVERLAP_KEY should contain the reference values.
    # We convert to dict for easier handling in compute_overlap if needed, 
    # although compute_overlap expects AtomicDataDict.
    
    # Ensure data is converted to AtomicDataDict format for compute_overlap
    if isinstance(data, AtomicData):
        data_dict = AtomicData.to_AtomicDataDict(data)
    else:
        data_dict = data
    
    # Check if overlap exists
    assert AtomicDataDict.EDGE_OVERLAP_KEY in data_dict
    ref_overlap = data_dict[AtomicDataDict.EDGE_OVERLAP_KEY].clone()
    
    # 2. Setup OrbitalMapper
    # idp needs to know types and basis.
    # The dataset loader likely populated atom types.
    # We need a fresh IDP for compute_overlap mostly for type info?
    # compute_overlap uses passed idp.
    idp = OrbitalMapper(basis=common_options['basis'], method="e3tb")
    
    # 3. Run compute_overlap
    # orb_names should be a list of filenames corresponding to atom types.
    # Dataset types are usually mapped.
    # For Si (single element), it should be just one file.
    orb_names = [ORB_FILE] 
    
    # We assume the atom type index 0 corresponds to Si.
    # Check data atom types
    # types = data_dict[AtomicDataDict.ATOM_TYPES_KEY]
    # Unique types should be [0].
    
    # Call the function
    # It modifies data_dict in-place (replaces EDGE_OVERLAP_KEY)
    compute_overlap(
        data=data_dict,
        idp=idp,
        orb_dir=DATA_ROOT,
        orb_names=orb_names
    )
    
    computed_overlap = data_dict[AtomicDataDict.EDGE_OVERLAP_KEY]
    
    # 4. Compare
    # Tolerance might need adjustment depending on op2c vs h5 precision
    # print max difference
    diff = torch.abs(computed_overlap - ref_overlap)
    max_diff = torch.max(diff).item()
    print(f"Max difference: {max_diff}")
    
    # Using a relatively loose tolerance first to check general correctness
    # If op2c is exact reference, it should be very small.
    # But h5 might be stored in float32? data_options says float64?
    # common_options dtype=float64.
    assert torch.allclose(computed_overlap, ref_overlap, atol=1e-7, rtol=1e-7)

if __name__ == "__main__":
    test_ovp2c_silicon()
