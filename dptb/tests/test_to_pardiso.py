import os
import shutil
import pytest
import torch
import h5py
import json
import ast
import numpy as np
from pathlib import Path
from dptb.postprocess.unified.system import TBSystem

ROOT_DIR = os.path.join(Path(os.path.abspath(__file__)).parent, "data")
TEST_DATA_DIR = os.path.join(ROOT_DIR, "test_to_pardiso")
MODEL_PATH = os.path.join(TEST_DATA_DIR, "nnsk.iter_ovp0.000.pth")
STRU_PATH = os.path.join(TEST_DATA_DIR, "min.vasp")
OUTPUT_DIR = os.path.join(TEST_DATA_DIR, "output")


def test_to_pardiso():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    
    tbsys = TBSystem(data=STRU_PATH, calculator=MODEL_PATH)
    tbsys.to_pardiso(output_dir=OUTPUT_DIR)
        
    # Verify Files and Content
    # 1. Atomic Numbers
    file_path = os.path.join(OUTPUT_DIR, "atomic_numbers.dat")
    assert os.path.exists(file_path), "Missing atomic_numbers.dat"
    atomic_numbers_saved = np.loadtxt(file_path, dtype=int)
    if atomic_numbers_saved.ndim == 0:
        atomic_numbers_saved = np.array([atomic_numbers_saved])
    assert np.array_equal(atomic_numbers_saved, tbsys.atoms.get_atomic_numbers()), "Atomic numbers mismatch"

    # 2. Positions
    file_path = os.path.join(OUTPUT_DIR, "positions.dat")
    assert os.path.exists(file_path), "Missing positions.dat"
    positions_saved = np.loadtxt(file_path)
    if positions_saved.ndim == 1:
        positions_saved = positions_saved.reshape(-1, 3) 
    assert np.allclose(positions_saved, tbsys.atoms.get_positions(), atol=1e-6), "Positions mismatch"

    # 3. Cell
    file_path = os.path.join(OUTPUT_DIR, "cell.dat")
    assert os.path.exists(file_path), "Missing cell.dat"
    assert np.allclose(np.loadtxt(file_path), np.array(tbsys.atoms.get_cell()), atol=1e-6), "Cell mismatch"

    # 4. Basis
    file_path = os.path.join(OUTPUT_DIR, "basis.dat")
    assert os.path.exists(file_path), "Missing basis.dat"
    with open(file_path, 'r') as f:
        basis_dict = f.read().strip()
    expected_basis_dict = {}
    for elem, orbitals in tbsys.calculator.model.idp.basis.items():
        counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0, 'g': 0}
        for o in orbitals:
            for orb_type in "spdfg":
                if orb_type in o:
                    counts[orb_type] += 1
                    break 
        compressed = ""
        for orb_type in "spdfg":
            if counts[orb_type] > 0:
                compressed += f"{counts[orb_type]}{orb_type}"
        expected_basis_dict[elem] = compressed
    assert str(expected_basis_dict) == basis_dict, "Basis mismatch between basis.dat and model config"
       

    # 5. Hamiltonian HDF5
    file_path = os.path.join(OUTPUT_DIR, "predicted_hamiltonians.h5")
    assert os.path.exists(file_path), "Missing predicted_hamiltonians.h5"
    with h5py.File(file_path, 'r') as f:
        assert "0" in f, "Hamiltonian HDF5 missing group '0'"
        assert len(f["0"].keys()) > 0, "Hamiltonian group '0' is empty"

    # 6. Overlap HDF5
    if tbsys.calculator.overlap:
        file_path = os.path.join(OUTPUT_DIR, "predicted_overlaps.h5")
        assert os.path.exists(file_path), "Missing predicted_overlaps.h5"
        with h5py.File(file_path, 'r') as f:
            assert "0" in f, "Overlap HDF5 missing group '0'"
            assert len(f["0"].keys()) > 0, "Overlap group '0' is empty"
             
    # Cleanup
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)


def test_save_h5():
    """Test the _save_h5 method including list wrapping logic."""
    OUTPUT_DIR_H5 = os.path.join(TEST_DATA_DIR, "output_h5")
    if os.path.exists(OUTPUT_DIR_H5):
        shutil.rmtree(OUTPUT_DIR_H5)
    os.makedirs(OUTPUT_DIR_H5, exist_ok=True)

    tbsys = TBSystem(data=STRU_PATH, calculator=MODEL_PATH)
    h_dict = {
        "0_0_0_0_0": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        "0_1_0_0_0": torch.tensor([[0.5, 0.5], [0.5, 0.5]], dtype=torch.float64)
    }
    
    # Verify single dict export
    tbsys._save_h5(h_dict,  "test.h5", OUTPUT_DIR_H5)
    assert os.path.exists(os.path.join(OUTPUT_DIR_H5, "test.h5")), "HDF5 file not created for single dict"
    with h5py.File(os.path.join(OUTPUT_DIR_H5, "test.h5"), 'r') as f:
        assert "0" in f, "Group '0' missing in HDF5"
        grp0 = f["0"]
        assert "0_0_0_0_0" in grp0, "Key '0_0_0_0_0' missing in group"
        assert "0_1_0_0_0" in grp0, "Key '0_1_0_0_0' missing in group"
        assert np.array_equal(grp0["0_0_0_0_0"][:], h_dict["0_0_0_0_0"].numpy())

    # Test Data: List of Dictionaries
    h_list = [h_dict, h_dict]
    tbsys._save_h5(h_list, "test_list.h5", OUTPUT_DIR_H5)
    assert os.path.exists(os.path.join(OUTPUT_DIR_H5, "test_list.h5")), "HDF5 file not created for list"
    with h5py.File(os.path.join(OUTPUT_DIR_H5, "test_list.h5"), 'r') as f:
        assert "0" in f, "Group '0' missing in list HDF5"
        assert "1" in f, "Group '1' missing in list HDF5"
        assert "0_0_0_0_0" in f["1"], "Key missing in group '1'"

    # Cleanup
    if os.path.exists(OUTPUT_DIR_H5):
        shutil.rmtree(OUTPUT_DIR_H5)


def test_symmetrize_hamiltonian():
    """Test the _symmetrize_hamiltonian method with synthetic and real data."""
    tbsys = TBSystem(data=STRU_PATH, calculator=MODEL_PATH)
    
    # Get initial raw data
    hr_raw, sr_raw = tbsys.calculator.get_hr(tbsys.data)
    
    # Verify HR Symmetrization
    missing_conjugates_hr = []
    initial_keys_hr = set(hr_raw.keys())
    
    for key in initial_keys_hr:
        parts = key.split('_')
        src, dst, rx, ry, rz = map(int, parts)
        rev_key = f"{dst}_{src}_{-rx}_{-ry}_{-rz}"
        if rev_key not in initial_keys_hr:
            missing_conjugates_hr.append((key, rev_key))
            # Just pick up to 5 examples to keep test focused
            if len(missing_conjugates_hr) >= 5:
                break
                
    hr_sym = tbsys._symmetrize_hamiltonian(hr_raw.copy())

    for key, rev_key in missing_conjugates_hr:
        assert rev_key not in initial_keys_hr, "Logic error: key should have been missing initially"
        assert rev_key in hr_sym, f"Expected {rev_key} to be added after symmetrization"
        val_original = hr_raw[key]
        val_added = hr_sym[rev_key]
        if isinstance(val_original, torch.Tensor):
            assert torch.allclose(val_added, val_original.t().conj(), atol=1e-6), f"Value mismatch for {rev_key}"
        else:
            assert np.allclose(val_added, val_original.T.conj(), atol=1e-6), f"Value mismatch for {rev_key}"
             
    # Verify SR Symmetrization
    if sr_raw is not None:
        missing_conjugates_sr = []
        initial_keys_sr = set(sr_raw.keys())
        
        for key in initial_keys_sr:
            parts = key.split('_')
            src, dst, rx, ry, rz = map(int, parts)
            rev_key = f"{dst}_{src}_{-rx}_{-ry}_{-rz}"
            if rev_key not in initial_keys_sr:
                missing_conjugates_sr.append((key, rev_key))
                if len(missing_conjugates_sr) >= 5:
                    break
                    
        sr_sym = tbsys._symmetrize_hamiltonian(sr_raw.copy())
        
        for src_key, required_rev_key in missing_conjugates_sr:
            assert required_rev_key in sr_sym, f"Expected {required_rev_key} to be added to SR"
            val_original = sr_raw[src_key]
            val_added = sr_sym[required_rev_key]
            if isinstance(val_original, torch.Tensor):
                assert torch.allclose(val_added, val_original.t().conj(), atol=1e-6), f"Value mismatch for SR {required_rev_key}"
            else:
                assert np.allclose(val_added, val_original.T.conj(), atol=1e-6), f"Value mismatch for SR {required_rev_key}"
