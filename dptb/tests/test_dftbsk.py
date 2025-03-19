import pytest
import os
import torch
from dptb.nn.dftbsk import DFTBSK
from dptb.nn.dftb.sk_param import SKParam
from dptb.data.transforms import OrbitalMapper
from dptb.data.build import build_dataset
from pathlib import Path
from dptb.data import AtomicDataset, DataLoader, AtomicDataDict, AtomicData
import numpy as np
from dptb.tests.tstools import compare_tensors_as_sets_float

rootdir = os.path.join(Path(os.path.abspath(__file__)).parent, "data")

class TestDFTBSK:
    skdatapath = f"{rootdir}/../../../examples/hBN_dftb/slakos"

    common_options = {
        "basis": {
            "B": ["2s", "2p"],
            "N": ["2s", "2p"]
        },
        "device": "cpu",
        "dtype": "float32",
        "overlap": True,
        "seed": 3982377700
    }
    model_options = {
    "dftbsk": {
            "skdata": skdatapath
        }
    }
    data_options = {
        "r_max": 2.6,
        "er_max": 2.6,
        "oer_max":1.6,
        "train": {
            "root": f"{rootdir}/hBN/dataset",
            "prefix": "kpath",
            "get_eigenvalues": False
        }
    }
    train_datasets = build_dataset(**data_options, **data_options["train"], **common_options)
    train_loader = DataLoader(dataset=train_datasets, batch_size=1, shuffle=True)

    batch = next(iter(train_loader))
    batch = AtomicData.to_AtomicDataDict(batch)
    
    def test_init_dftbsk(self):
        model = DFTBSK(**self.common_options, **self.model_options['dftbsk'])
        skparams = SKParam(basis=self.common_options["basis"], skdata=self.skdatapath)
        skdict = skparams.skdict
        assert model.hopping_param.requires_grad == False
        assert model.overlap_param.requires_grad == False
        assert model.onsite_param.requires_grad == False
        assert model.distance_param.requires_grad == False

        assert torch.allclose(skdict['Hopping'], model.hopping_param)
        assert torch.allclose(skdict['Overlap'], model.overlap_param)
        assert torch.allclose(skdict['OnsiteE'], model.onsite_param)
        assert torch.allclose(skdict['Distance'], model.distance_param)
        assert model.onsite_fn.functype == 'dftb'
        assert model.hopping_fn.functype == 'dftb'

    def test_forward_dftbsk(self):
        model = DFTBSK(**self.common_options, **self.model_options['dftbsk'], transform=False)
        data = model(self.batch)
        assert torch.allclose(data[AtomicDataDict.NODE_FEATURES_KEY],torch.tensor([[-1.8268676758e+01, -7.1081967354e+00],
                                                                                   [-9.2467069626e+00, -3.5892550945e+00]]))
        expected_edge_feature = torch.tensor([[-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740054, -9.2785778, -7.2594633,  3.2225938],
        [-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740068, -9.2785797, -7.2594643,  3.2225947],
        [-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740068, -9.2785797, -7.2594643,  3.2225947],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515],
        [-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740054,  6.6869082, -7.2594633,  3.2225938],
        [-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740068,  6.6869092, -7.2594643,  3.2225947],
        [-0.4897783,  0.7541569,  1.0319425, -0.1561696],
        [-7.7740068,  6.6869092, -7.2594643,  3.2225947],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515],
        [-1.1053317, -1.4127309,  1.7213905, -0.3220515]])

        assert compare_tensors_as_sets_float(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_edge_feature, precision=5)
        # assert torch.allclose(data[AtomicDataDict.EDGE_FEATURES_KEY], expected_edge_feature)

        expected_edge_overlap = torch.tensor([[ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880,  0.3365866,  0.3249941, -0.1442579],
        [ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880,  0.3365866,  0.3249941, -0.1442579],
        [ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880,  0.3365866,  0.3249941, -0.1442579],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881],
        [ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880, -0.2903073,  0.3249941, -0.1442579],
        [ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880, -0.2903073,  0.3249941, -0.1442579],
        [ 0.0115951, -0.0208762, -0.0355638,  0.0046229],
        [ 0.2665880, -0.2903073,  0.3249941, -0.1442579],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881],
        [ 0.0399277,  0.0585683, -0.0838758,  0.0126881]])

        assert compare_tensors_as_sets_float(data[AtomicDataDict.EDGE_OVERLAP_KEY], expected_edge_overlap, precision=5)
        # assert torch.allclose(data[AtomicDataDict.EDGE_OVERLAP_KEY], expected_edge_overlap)

        assert AtomicDataDict.NODE_SOC_SWITCH_KEY in data
        assert not data[AtomicDataDict.NODE_SOC_SWITCH_KEY].all()




