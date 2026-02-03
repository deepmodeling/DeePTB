"""
Consistency tests for feature_to_block optimization.

This test module verifies that the optimized feature_to_block function
produces identical outputs to the original implementation, ensuring
correctness after performance optimization.

The optimization replaced nested Python loops with vectorized scatter
operations and added device-side caching to avoid redundant transfers.
"""

import pytest
import os
import torch
import ase
import re
import time
from pathlib import Path
from dptb.nn.nnsk import NNSK
from dptb.data.transforms import OrbitalMapper
from dptb.data.build import build_dataset
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
from dptb.data import _keys
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.data.interfaces.ham_to_feature import block_to_feature, feature_to_block
from dptb.utils.constants import anglrMId

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")


def feature_to_block_reference(data, idp, overlap: bool = False):
    """
    Reference implementation of feature_to_block from commit 6c38085.

    This is the original implementation before optimization, used as
    ground truth for consistency testing.

    Args:
        data: AtomicData dictionary containing node/edge features
        idp: OrbitalMapper instance with basis information
        overlap: If True, process overlap features instead of Hamiltonian

    Returns:
        Dictionary mapping block indices to block matrices
    """
    idp.get_orbital_maps()
    idp.get_orbpair_maps()

    has_block = False
    if not overlap:
        if data.get(_keys.NODE_FEATURES_KEY, None) is not None:
            node_features = data[_keys.NODE_FEATURES_KEY]
            edge_features = data[_keys.EDGE_FEATURES_KEY]
            has_block = True
            blocks = {}
    else:
        if data.get(_keys.NODE_OVERLAP_KEY, None) is not None:
            node_features = data[_keys.NODE_OVERLAP_KEY]
            edge_features = data[_keys.EDGE_OVERLAP_KEY]
            has_block = True
            blocks = {}
        else:
            raise KeyError("Overlap features not found in data.")

    if has_block:
        # get node blocks from node_features
        for atom, onsite in enumerate(node_features):
            symbol = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom].reshape(-1))]
            basis_list = idp.basis[symbol]
            block = torch.zeros((idp.norbs[symbol], idp.norbs[symbol]), device=node_features.device, dtype=node_features.dtype)

            for index, basis_i in enumerate(basis_list):
                f_basis_i = idp.basis_to_full_basis[symbol].get(basis_i)
                slice_i = idp.orbital_maps[symbol][basis_i]
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                for basis_j in basis_list[index:]:
                    f_basis_j = idp.basis_to_full_basis[symbol].get(basis_j)
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol][basis_j]
                    pair_ij = f_basis_i + "-" + f_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = onsite[feature_slice].reshape(2*li+1, 2*lj+1)
                    block[slice_i, slice_j] = block_ij
                    if slice_i != slice_j:
                        block[slice_j, slice_i] = block_ij.T

            block_index = '_'.join(map(str, map(int, [atom, atom] + list([0, 0, 0]))))
            blocks[block_index] = block

        # get edge blocks from edge_features
        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_cell_shift = data[_keys.EDGE_CELL_SHIFT_KEY]
        for edge, hopping in enumerate(edge_features):
            atom_i, atom_j, R_shift = edge_index[0][edge], edge_index[1][edge], edge_cell_shift[edge]
            symbol_i = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_i].reshape(-1))]
            symbol_j = ase.data.chemical_symbols[idp.untransform(data[_keys.ATOM_TYPE_KEY][atom_j].reshape(-1))]
            block = torch.zeros((idp.norbs[symbol_i], idp.norbs[symbol_j]), device=edge_features.device, dtype=edge_features.dtype)

            for index, f_basis_i in enumerate(idp.full_basis):
                basis_i = idp.full_basis_to_basis[symbol_i].get(f_basis_i)
                if basis_i is None:
                    continue
                li = anglrMId[re.findall(r"[a-zA-Z]+", basis_i)[0]]
                slice_i = idp.orbital_maps[symbol_i][basis_i]
                for f_basis_j in idp.full_basis[index:]:
                    basis_j = idp.full_basis_to_basis[symbol_j].get(f_basis_j)
                    if basis_j is None:
                        continue
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", basis_j)[0]]
                    slice_j = idp.orbital_maps[symbol_j][basis_j]
                    pair_ij = f_basis_i + "-" + f_basis_j
                    feature_slice = idp.orbpair_maps[pair_ij]
                    block_ij = hopping[feature_slice].reshape(2*li+1, 2*lj+1)
                    if f_basis_i == f_basis_j:
                        block[slice_i, slice_j] = 0.5 * block_ij
                    else:
                        block[slice_i, slice_j] = block_ij

            block_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(R_shift))))
            if atom_i < atom_j:
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[block_index] += block
            elif atom_i == atom_j:
                r_index = '_'.join(map(str, map(int, [atom_i, atom_j] + list(-R_shift))))
                if blocks.get(r_index, None) is None:
                    blocks[block_index] = block
                else:
                    blocks[r_index] += block.T
            else:
                block_index = '_'.join(map(str, map(int, [atom_j, atom_i] + list(-R_shift))))
                if blocks.get(block_index, None) is None:
                    blocks[block_index] = block.T
                else:
                    blocks[block_index] += block.T

    return blocks


class TestFeatureToBlockConsistency:
    """
    Test suite to verify consistency between optimized and reference
    implementations of feature_to_block.
    """

    # Common configuration for hBN test system
    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": False,
        "seed": 3982377700
    }

    model_options = {
        "nnsk": {
            "onsite": {
                "method": "none"
            },
            "hopping": {
                "method": "powerlaw",
                "rs": 2.6,
                "w": 0.35
            },
            "freeze": False,
            "std": 0.1,
            "push": None
        }
    }

    data_options = {
        "r_max": 2.6,
        "er_max": 2.6,
        "oer_max": 1.6,
        "train": {
            "root": f"{rootdir}/hBN/dataset",
            "prefix": "kpath",
            "get_eigenvalues": True
        }
    }

    # Build test dataset
    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)
    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=False)
    batch = next(iter(train_loader))
    batch = AtomicData.to_AtomicDataDict(batch)

    # Create OrbitalMapper instances
    idp = OrbitalMapper(basis=common_options['basis'], method="sktb")

    def test_consistency_hamiltonian_hBN(self):
        """
        Test that optimized and reference implementations produce identical
        Hamiltonian blocks for hBN system.
        """
        # Create NNSK model and generate features
        hamiltonian = SKHamiltonian(idp_sk=self.idp, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"], transform=False)

        with torch.no_grad():
            data = nnsk(self.batch.copy())
            data = hamiltonian(data)

            # Run both implementations
            blocks_optimized = feature_to_block(data, nnsk.idp, overlap=False)
            blocks_reference = feature_to_block_reference(data, nnsk.idp, overlap=False)

        # Verify same block indices
        assert set(blocks_optimized.keys()) == set(blocks_reference.keys()), \
            f"Block indices mismatch: optimized has {set(blocks_optimized.keys())}, reference has {set(blocks_reference.keys())}"

        # Compare each block
        for block_idx in blocks_optimized.keys():
            opt_block = blocks_optimized[block_idx]
            ref_block = blocks_reference[block_idx]

            # Check shapes match
            assert opt_block.shape == ref_block.shape, \
                f"Shape mismatch for block {block_idx}: {opt_block.shape} vs {ref_block.shape}"

            # Check values match within tolerance
            assert torch.allclose(opt_block, ref_block, atol=1e-6, rtol=1e-5), \
                f"Values mismatch for block {block_idx}: max diff = {torch.max(torch.abs(opt_block - ref_block))}"


    def test_performance_improvement(self):
        """
        Benchmark test to verify that optimized implementation is faster
        than the reference implementation.
        """
        # Create NNSK model and generate features
        hamiltonian = SKHamiltonian(idp_sk=self.idp, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"], transform=False)

        with torch.no_grad():
            data = nnsk(self.batch.copy())
            data = hamiltonian(data)

            # Warm-up runs
            for _ in range(3):
                _ = feature_to_block(data, nnsk.idp, overlap=False)
                _ = feature_to_block_reference(data, nnsk.idp, overlap=False)

            # Benchmark optimized implementation
            num_runs = 10
            start_time = time.time()
            for _ in range(num_runs):
                _ = feature_to_block(data, nnsk.idp, overlap=False)
            optimized_time = (time.time() - start_time) / num_runs

            # Benchmark reference implementation
            start_time = time.time()
            for _ in range(num_runs):
                _ = feature_to_block_reference(data, nnsk.idp, overlap=False)
            reference_time = (time.time() - start_time) / num_runs

        speedup = reference_time / optimized_time
        print(f"\nPerformance comparison:")
        print(f"  Optimized: {optimized_time*1000:.3f} ms")
        print(f"  Reference: {reference_time*1000:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        # Assert that optimized version is faster (or at least not significantly slower)
        # Allow for some variance in timing measurements
        assert optimized_time <= reference_time * 1.1, \
            f"Optimized implementation is slower: {optimized_time:.4f}s vs {reference_time:.4f}s"

    def test_round_trip_consistency(self):
        """
        Test that feature_to_block -> block_to_feature round-trip preserves features.
        This verifies that both the optimized feature_to_block and block_to_feature
        work correctly together.
        """
        # Create NNSK model and generate features
        hamiltonian = SKHamiltonian(idp_sk=self.idp, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"], transform=False)

        with torch.no_grad():
            data = nnsk(self.batch.copy())
            data = hamiltonian(data)

            # Save original features
            original_node_features = data[_keys.NODE_FEATURES_KEY].clone()
            original_edge_features = data[_keys.EDGE_FEATURES_KEY].clone()

            # Round trip: features -> blocks -> features
            blocks = feature_to_block(data, nnsk.idp, overlap=False)
            block_to_feature(data, nnsk.idp, blocks=blocks)

            # Verify features are preserved
            assert torch.allclose(data[_keys.NODE_FEATURES_KEY], original_node_features, atol=1e-6, rtol=1e-5), \
                "Node features not preserved in round-trip"
            assert torch.allclose(data[_keys.EDGE_FEATURES_KEY], original_edge_features, atol=1e-6, rtol=1e-5), \
                "Edge features not preserved in round-trip"
