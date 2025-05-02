import pytest
import os
import torch
from dptb.nn.nnsk import NNSK
from dptb.data.transforms import OrbitalMapper
from dptb.data.build import build_dataset
from pathlib import Path
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
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
