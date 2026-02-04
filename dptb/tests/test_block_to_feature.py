import pytest
import os
import torch
import ase
import re
import time
from dptb.nn.nnsk import NNSK
from dptb.data.transforms import OrbitalMapper
from dptb.data.build import build_dataset
from pathlib import Path
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
from dptb.data import _keys
import numpy as np
from dptb.nn.hamiltonian import  SKHamiltonian
from dptb.data.interfaces.ham_to_feature import block_to_feature, feature_to_block
from dptb.utils.constants import anglrMId
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.tests.tstools import compare_tensors_as_sets_float, compare_tensors_as_sets

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestBlock2Feature:
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
        "push": None}
    }
    data_options = {
        "r_max": 2.6,
        "er_max": 2.6,
        "oer_max":1.6,
        "train": {
            "root": f"{rootdir}/hBN/dataset",
            "prefix": "kpath",
            "get_eigenvalues": True
        }
    }

    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)
    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)

    batch = next(iter(train_loader))
    batch = AtomicData.to_AtomicDataDict(batch)
    idp_sk = OrbitalMapper(basis=common_options['basis'], method="sktb")
    idp = OrbitalMapper(basis=common_options['basis'], method="e3tb")

    sk2irs = {
            's-s': torch.tensor([[1.]]),
            's-p': torch.tensor([[1.]]),
            's-d': torch.tensor([[1.]]),
            'p-s': torch.tensor([[1.]]),
            'p-p': torch.tensor([
                [3**0.5/3,2/3*3**0.5],[6**0.5/3,-6**0.5/3]
            ]),
            'p-d':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],[(3/5)**0.5,-2/5**0.5]
            ]),
            'd-s':torch.tensor([[1.]]),
            'd-p':torch.tensor([
                [(2/5)**0.5,(6/5)**0.5],
                [(3/5)**0.5,-2/5**0.5]
            ]),
            'd-d':torch.tensor([
                [5**0.5/5, 2*5**0.5/5, 2*5**0.5/5],
                [2*(1/14)**0.5,2*(1/14)**0.5,-4*(1/14)**0.5],
                [3*(2/35)**0.5,-4*(2/35)**0.5,(2/35)**0.5]
                ])
        }

    def test_transform_onsiteblocks_none(self):
        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"],transform=False)
        data = nnsk(self.batch)
        data = hamiltonian(data)

        with torch.no_grad():
            block = feature_to_block(data, nnsk.idp)
            block_to_feature(data, nnsk.idp, blocks=block)
        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape == torch.Size([2, 13])

        expected_onsite = torch.tensor([[-18.4200038910,   0.0000000000,   0.0000000000,   0.0000000000,
                                          -7.2373123169,  -0.0000000000,  -0.0000000000,  -0.0000000000,
                                          -7.2373123169,  -0.0000000000,  -0.0000000000,  -0.0000000000,
                                          -7.2373123169],
                                        [ -9.3830089569,   0.0000000000,   0.0000000000,   0.0000000000,
                                          -3.7138016224,  -0.0000000000,  -0.0000000000,  -0.0000000000,
                                          -3.7138016224,  -0.0000000000,  -0.0000000000,  -0.0000000000,
                                          -3.7138016224]])
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY], expected_onsite)

    def test_transform_hoppingblocks(self):
        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"],transform=False)
        nnsk.hopping_param.data = torch.tensor([[[-0.0299384445, -0.0187778082],
         [ 0.1915897578,  0.0690195262],
         [-0.2321701497, -0.1196410209],
         [ 0.0197028164, -0.1177332327]],

        [[ 0.0550494418, -0.0191540867],
         [-0.1395172030,  0.0475118719],
         [-0.0351739973,  0.0052711815],
         [ 0.0192712545, -0.1666133553]],

        [[ 0.0550494418, -0.0191540867],
         [ 0.0586687513,  0.0158295482],
         [-0.0351739973,  0.0052711815],
         [ 0.0192712545, -0.1666133553]],

        [[ 0.1311892122, -0.0209838580],
         [ 0.0781731308,  0.0989692509],
         [ 0.0414713360, -0.1508950591],
         [ 0.2036036998,  0.0131590459]]])
        data = nnsk(self.batch)
        data = hamiltonian(data)

        expected_edge_index = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1]])
        expected_edge_cell_shift = torch.tensor([[-1.,  0.,  0.],
        [-1.,  0.,  0.],
        [ 0.,  1.,  0.],
        [ 0.,  1.,  0.],
        [ 1.,  1.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  0.],
        [ 0., -1.,  0.],
        [-1.,  0.,  0.],
        [ 1., -0., -0.],
        [ 1., -0., -0.],
        [-0., -1., -0.],
        [-0., -1., -0.],
        [-1., -1., -0.],
        [-0., -0., -0.],
        [-1., -1., -0.],
        [-0.,  1., -0.],
        [ 1., -0., -0.]])

        exp_val = torch.cat((expected_edge_index.T, expected_edge_cell_shift), axis=1)
        tar_val = torch.cat((data[AtomicDataDict.EDGE_INDEX_KEY].T, data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]), axis=1)
        exp_val = exp_val.int()
        tar_val = tar_val.int()

        assert compare_tensors_as_sets(exp_val, tar_val)

        with torch.no_grad():
            block = feature_to_block(data, nnsk.idp)
            block_to_feature(data, nnsk.idp, blocks=block)
        
        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape == torch.Size([18, 13])
        

        expected_selected_hopblock = torch.tensor([[ 5.3185172379e-02, -4.6635824091e-09,  1.3500485174e-09,
                                                     3.0885510147e-02,  8.2756355405e-02,  4.3990724937e-16,
                                                     1.0063905265e-08,  4.3990724937e-16,  8.2756355405e-02,
                                                    -2.9133742085e-09,  1.0063905265e-08, -2.9133742085e-09,
                                                     1.6106124967e-02],
                                                   [ 6.2371429056e-02, -6.6437192261e-02,  2.9040618799e-09,
                                                     2.9040618799e-09, -3.9765007794e-02,  2.7151161319e-09,
                                                     2.7151161319e-09,  2.7151161319e-09,  2.2349609062e-02,
                                                    -1.1868149201e-16,  2.7151161319e-09, -1.1868149201e-16,
                                                     2.2349609062e-02],
                                                   [ 5.3185172379e-02, -0.0000000000e+00,  1.3500485174e-09,
                                                    -3.0885510147e-02,  8.2756355405e-02,  0.0000000000e+00,
                                                     0.0000000000e+00,  0.0000000000e+00,  8.2756355405e-02,
                                                     2.9133742085e-09,  0.0000000000e+00,  2.9133742085e-09,
                                                     1.6106124967e-02],
                                                   [ 6.2371429056e-02,  3.3218599856e-02,  2.9040618799e-09,
                                                    -5.7536296546e-02,  6.8209525198e-03, -1.3575581770e-09,
                                                     2.6896420866e-02, -1.3575582880e-09,  2.2349609062e-02,
                                                     2.3513595515e-09,  2.6896420866e-02,  2.3513595515e-09,
                                                    -2.4236353114e-02],
                                                   [ 6.2371429056e-02, -1.5878447890e-01, -6.9406898007e-09,
                                                     1.1987895121e-08, -3.9765007794e-02, -2.7151161319e-09,
                                                     4.6895234362e-09, -2.7151161319e-09,  2.2349609062e-02,
                                                     2.0498557313e-16,  4.6895234362e-09,  2.0498557313e-16,
                                                     2.2349609062e-02],
                                                   [-1.0692023672e-02,  5.7914875448e-02,  2.9231701504e-09,
                                                     3.3437173814e-02, -5.7711567730e-02, -3.2524076765e-09,
                                                    -3.7203211337e-02, -3.2524074545e-09,  6.7262742668e-03,
                                                    -1.8777785993e-09, -3.7203207612e-02, -1.8777785993e-09,
                                                    -1.4753011055e-02]])
        
        # assert compare_tensors_as_sets_float(data[AtomicDataDict.EDGE_FEATURES_KEY][[0,3,9,5,12,15]], expected_selected_hopblock)
        # assert torch.all(torch.abs(data[AtomicDataDict.EDGE_FEATURES_KEY][[0,3,9,5,12,15]] - expected_selected_hopblock) < 1e-6)

        testind = [0,3,9,5,12,15]
        tarind_list = []
        for i in range(len(testind)):
            ind = testind[i]
            bond = exp_val.tolist()[ind]
            assert bond in tar_val.tolist()
            tarind = tar_val.tolist().index(bond)
            tarind_list.append(tarind)
        assert torch.all(torch.abs(data[AtomicDataDict.EDGE_FEATURES_KEY][tarind_list] - expected_selected_hopblock) < 1e-4)


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

    This ensures the performance optimization (commits 904a27d and d97a847)
    produces identical results to the original implementation.
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

