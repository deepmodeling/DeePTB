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
from dptb.utils.constants import anglrMId, orbitalId
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.tests.tstools import compare_tensors_as_sets_float

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestSKHamiltonian:
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
    
    def test_init(self):
        hamiltonian = SKHamiltonian(basis=self.common_options['basis'], idp_sk=self.idp_sk, onsite=True)
        assert hamiltonian.idp_sk == self.idp_sk
        hamiltonian = SKHamiltonian(basis=self.common_options['basis'], onsite=True)
        assert hamiltonian.idp_sk == self.idp_sk
        
        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True)
        assert hamiltonian.strain is False
        assert hamiltonian.onsite
        assert hamiltonian.idp_sk == self.idp_sk
        assert hamiltonian.idp == self.idp

        assert hamiltonian.edge_field == AtomicDataDict.EDGE_FEATURES_KEY
        assert hamiltonian.node_field == AtomicDataDict.NODE_FEATURES_KEY


    def test_initialize_basis(self):
        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True)

        # 这部分的检查看起来非常没有意义，因为这部分测试代码是直接从_initialize_CG_basis中复制过来的。
        # 但是这部分测试是为了保证_initialize_CG_basis 不被修改，或者在修改的时候能够保证正确性。
        for l1 in orbitalId.keys():
            for l2 in orbitalId.keys():
                pairtype = orbitalId[l1]+"-"+orbitalId[l2]
                basis_ref = []
                for im in range(0, min(l1,l2)+1):
                    mat = torch.zeros((2*l1+1, 2*l2+1))
                    if im == 0:
                        mat[l1,l2] = 1.
                    else:
                        mat[l1+im,l2+im] = 1.
                        mat[l1-im, l2-im] = 1.
                    basis_ref.append(mat)
            
                basis_ref = torch.stack(basis_ref, dim=-1)

                
                basis = hamiltonian._initialize_basis(pairtype)

                # print(basis, basis_ref)

                assert torch.allclose(basis, basis_ref)

        assert hamiltonian.skbasis.keys() == self.idp_sk.orbpairtype_maps.keys()
        # for pairtype in self.idp_sk.orbpairtype_maps.keys():
        #     assert torch.allclose(hamiltonian.cgbasis[pairtype], hamiltonian._initialize_CG_basis(pairtype))

    def test_onsiteblocks_none(self):
        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"],transform=False)
        data = nnsk(self.batch)
        data = hamiltonian(data)
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

    def test_hoppingblocks(self):
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
        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape == torch.Size([18, 13])
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

        assert compare_tensors_as_sets_float(exp_val, tar_val)


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

    def test_onsite_stain(self):
        model_options = self.model_options
        model_options["nnsk"]["onsite"] =  {"method": "strain", "rs":2.6, "w":0.35}

        hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True,strain=True)
        nnsk = NNSK(**self.common_options, **self.model_options["nnsk"],transform=False)
        nnsk.strain_param.data = torch.tensor([[[-0.1577102989,  0.0360933691],
         [-0.1353305429, -0.0207103472],
         [-0.0248758439, -0.1232000738],
         [ 0.0625670776, -0.1223127022]],

        [[-0.1118659005,  0.0378381126],
         [-0.0780378580, -0.0873878151],
         [-0.0732800886,  0.0514300428],
         [ 0.0397648588,  0.0643456504]],

        [[-0.1445292681, -0.0807765052],
         [ 0.1197529435, -0.1834534705],
         [ 0.0420080312,  0.1129035875],
         [ 0.0426359400, -0.1136116311]],

        [[-0.0388229564, -0.0334187299],
         [ 0.0952333137, -0.0462365486],
         [-0.0607928596, -0.0362483226],
         [-0.1507207304, -0.0508698337]]])
        
        data = nnsk(self.batch)
        data = hamiltonian(data)

        assert data[AtomicDataDict.NODE_FEATURES_KEY].shape == torch.Size([2, 13])
        expected_strainonsite =torch.tensor([[-1.8916072845e+01, -7.4505805969e-09, -1.8260744028e-08,
          7.4505805969e-09, -7.0913019180e+00,  1.1102230246e-16,
          2.9103830457e-11,  1.1102230246e-16, -7.0902109146e+00,
          2.2030988145e-16,  3.7543941289e-09,  2.2030988145e-16,
         -7.0913019180e+00],
        [-9.7643690109e+00,  0.0000000000e+00,  1.1720334925e-08,
         -1.4901161194e-08, -3.7709136009e+00, -4.4408920985e-16,
          2.6077032089e-08, -4.4408920985e-16, -3.5776705742e+00,
          0.0000000000e+00,  2.6077032089e-08,  0.0000000000e+00,
         -3.7709136009e+00]])
        
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY], expected_strainonsite, atol=1e-6, rtol=1e-4)


