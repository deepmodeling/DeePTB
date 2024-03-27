import pytest
from dptb.nnops.loss import HamilLossAnalysis
from dptb.data import AtomicData
from ase.io import read
import torch

@pytest.fixture(scope='session', autouse=True)
def root_directory(request):
    return str(request.config.rootdir)

@pytest.mark.order(1)
def test_hamilloss_analysis_wo_decompose(root_directory):
    la = HamilLossAnalysis(basis={"B":"1s1p", "N": "1s1p"}, decompose=False)
    data = AtomicData.from_ase(
        atoms=read(root_directory+"/dptb/tests/data/hBN/hBN.vasp"),
        r_max=4.0
        ).to_dict()
    data = la.idp(data)

    data["edge_features"] = torch.zeros(data["edge_index"].shape[1], 13)
    data["node_features"] = torch.zeros(data["atom_types"].shape[0], 13)

    ref_data = data.copy()
    ref_data["edge_features"] = torch.ones(data["edge_index"].shape[1], 13)
    ref_data["node_features"] = torch.ones(data["atom_types"].shape[0], 13)

    result = la(data, ref_data)
    assert torch.abs(result["mae"] - 1.0) < 1e-6
    assert torch.abs(result["rmse"] - 1.0) < 1e-6
    assert result["onsite"]["B"]["n_element"] == 13
    assert result["onsite"]["N"]["n_element"] == 13
    assert result["hopping"]["B-N"]["n_element"] == 156
    assert result["hopping"]["B-B"]["n_element"] == 78
    assert result["hopping"]["N-N"]["n_element"] == 78

@pytest.mark.order(2)
def test_hamilloss_analysis_w_decompose(root_directory):
    la = HamilLossAnalysis(basis={"B":"1s1p", "N": "1s1p"}, decompose=True)
    data = AtomicData.from_ase(
        atoms=read(root_directory+"/dptb/tests/data/hBN/hBN.vasp"),
        r_max=4.0
        ).to_dict()
    data = la.idp(data)

    data["edge_features"] = torch.zeros(data["edge_index"].shape[1], 13)
    data["node_features"] = torch.zeros(data["atom_types"].shape[0], 13)

    ref_data = data.copy()
    ref_data["edge_features"] = torch.ones(data["edge_index"].shape[1], 13)
    ref_data["node_features"] = torch.ones(data["atom_types"].shape[0], 13)

    result = la(data, ref_data)
    assert torch.abs(result["mae"] - 0.7673) < 1e-4
    assert torch.abs(result["rmse"] - 1.0) < 1e-6
    assert result["onsite"]["B"]["n_element"] == 13
    assert result["onsite"]["N"]["n_element"] == 13
    assert result["hopping"]["B-N"]["n_element"] == 156
    assert result["hopping"]["B-B"]["n_element"] == 78
    assert result["hopping"]["N-N"]["n_element"] == 78