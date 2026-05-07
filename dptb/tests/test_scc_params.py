from pathlib import Path

import torch
import pytest

from dptb.data.transforms import OrbitalMapper
from dptb.nn.dftb.scc_params import SCCParams
from dptb.nn.dftb.sk_param import SKParam
from dptb.nn.nnsk import NNSK
from dptb.utils.constants import Harte2eV
from dptb.nn.sktb.HubbardUDB import Hubbard_U_dict


ROOTDIR = Path(__file__).resolve().parent / "data"


def test_scc_params_manual_options():
    basis = {"C": ["2s", "2p"], "H": ["1s"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=idp,
        options={
            "hubbard_u": {"C": {"2s": 12.0, "2p": 9.0}, "H": {"1s": 10.0}},
            "occupation": {"C": {"2s": 2, "2p": 2}, "H": {"1s": 1}},
            "onsite_e": {"C": {"2s": -0.5, "2p": 0.1}, "H": {"1s": -0.2}},
            "mass": {"C": 12.011, "H": 1.008},
            "r_max": {"C": 4.0, "H": 3.0},
            "use_database": False,
        },
        dtype=torch.float64,
    )

    assert params.skdict["HubdU"].shape == torch.Size([2, 2, 1])
    assert params.skdict["Occu"].shape == torch.Size([2, 2, 1])
    assert params.skdict["Mass"].shape == torch.Size([2, 1])
    cidx = idp.chemical_symbol_to_type["C"]
    sidx = idp.skonsite_maps["1s-1s"]
    pidx = idp.skonsite_maps["1p-1p"]
    assert params.skdict["HubdU"][cidx, sidx, 0] == 12.0
    assert params.skdict["HubdU"][cidx, pidx, 0] == 9.0
    assert params.skdict["Highest_Occu_U"][cidx, 0, 0] == 9.0



def test_scc_params_highest_occu_u_uses_highest_occupied_onsite_level():
    basis = {"B": ["2s", "2p"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=idp,
        options={
            "hubbard_u": {"B": {"2s": 12.0, "2p": 8.0}},
            "occupation": {"B": {"2s": 2, "2p": 1}},
            "onsite_e": {"B": {"2s": -0.7, "2p": 0.2}},
            "mass": {"B": 10.81},
            "r_max": {"B": 3.0},
            "use_database": False,
        },
        dtype=torch.float64,
    )

    bidx = idp.chemical_symbol_to_type["B"]
    assert params.skdict["Highest_Occu_U"][bidx, 0, 0] == 8.0


def test_scc_params_highest_occu_u_option_skips_onsite_e():
    basis = {"B": ["2s", "2p"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=idp,
        options={
            "hubbard_u": {"B": {"2s": 12.0, "2p": 8.0}},
            "occupation": {"B": {"2s": 2, "2p": 1}},
            "highest_occu_u": {"B": 6.5},
            "mass": {"B": 10.81},
            "r_max": {"B": 3.0},
            "use_database": False,
        },
        dtype=torch.float64,
    )

    assert params.skdict["Highest_Occu_U"][idp.chemical_symbol_to_type["B"], 0, 0] == 6.5


def test_scc_params_database_converts_hubbard_u():
    basis = {"B": ["2s", "2p"], "N": ["2s", "2p"], "C": ["2s", "2p"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=idp,
        options={"r_max": {"B": 3.0, "N": 3.0, "C": 3.0}, "use_database": True},
        dtype=torch.float64,
    )

    bidx = idp.chemical_symbol_to_type["B"]
    sidx = idp.skonsite_maps["1s-1s"]
    expected = Hubbard_U_dict["B"]["2s"] * Harte2eV
    assert torch.allclose(params.skdict["HubdU"][bidx, sidx, 0], torch.tensor(expected, dtype=torch.float64))
    assert params.skdict["Occu"][idp.chemical_symbol_to_type["N"], sidx, 0] == 2
    assert params.skdict["Mass"][idp.chemical_symbol_to_type["C"], 0] > 12.0


def test_scc_params_missing_without_database_raises():
    basis = {"H": ["1s"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    with pytest.raises(ValueError, match="Missing onsite_e values"):
        SCCParams.from_options(
            basis=basis,
            idp_sk=idp,
            options={
                "hubbard_u": {"H": {"1s": 10.0}},
                "occupation": {"H": {"1s": 1}},
                "mass": {"H": 1.008},
                "use_database": False,
                "r_max": {"H": 3.0},
            },
        )


def test_scc_params_metadata_priority():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=True, hopping={"method": "powerlaw", "rs": 3.5, "w": 0.2})
    model.scc_metadata = {
        "hubbard_u": {"H": {"1s": 8.0}},
        "occupation": {"H": {"1s": 1}},
        "highest_occu_u": {"H": 8.0},
        "mass": {"H": 1.5},
    }

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=model.idp_sk,
        options={"use_database": True},
        model=model,
    )
    assert params.skdict["HubdU"][0, 0, 0] == 8.0
    assert params.skdict["Mass"][0, 0] == 1.5
    assert params.r_max == {"H": 3.5}

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=model.idp_sk,
        options={"hubbard_u": {"H": {"1s": 9.0}}, "use_database": True},
        model=model,
    )
    assert params.skdict["HubdU"][0, 0, 0] == 9.0


def test_scc_params_from_model_uses_model_options_scc():
    basis = {"H": ["1s"]}
    model = NNSK(basis=basis, overlap=True, hopping={"method": "powerlaw", "rs": 3.5, "w": 0.2})
    model.model_options = {
        "scc": {
            "hubbard_u": {"H": {"1s": 7.0}},
            "occupation": {"H": {"1s": 1}},
            "highest_occu_u": {"H": 7.0},
            "mass": {"H": 1.2},
            "r_max": {"H": 3.1},
        }
    }

    params = SCCParams.from_model(model)

    assert params is not None
    assert params.skdict["HubdU"][0, 0, 0] == 7.0
    assert params.skdict["Highest_Occu_U"][0, 0, 0] == 7.0
    assert params.skdict["Mass"][0, 0] == 1.2
    assert params.r_max == {"H": 3.1}


def test_scc_params_metadata_roundtrip_persists_highest_occu_u_not_onsite_e():
    basis = {"B": ["2s", "2p"]}
    idp = OrbitalMapper(basis, method="sktb")
    idp.get_orbpair_maps()
    idp.get_skonsite_maps()

    params = SCCParams.from_options(
        basis=basis,
        idp_sk=idp,
        options={
            "hubbard_u": {"B": {"2s": 12.0, "2p": 8.0}},
            "occupation": {"B": {"2s": 2, "2p": 1}},
            "onsite_e": {"B": {"2s": -0.7, "2p": 0.2}},
            "mass": {"B": 10.81},
            "r_max": {"B": 3.0},
            "use_database": False,
        },
        dtype=torch.float64,
    )

    metadata = params.to_metadata()

    assert "highest_occu_u" in metadata
    assert "onsite_e" not in metadata
    assert metadata["highest_occu_u"]["B"] == 8.0


def test_scc_params_from_skparam_preserves_shapes():
    skdata = str(ROOTDIR / "dftb")
    skp = SKParam(basis={"C": ["2s", "2p"]}, skdata=skdata, cal_rcuts=True)
    params = SCCParams.from_skparam(skp)
    assert params.skdict["HubdU"].shape == skp.skdict["HubdU"].shape
    assert params.skdict["Highest_Occu_U"].shape == skp.skdict["Highest_Occu_U"].shape
    assert torch.allclose(params.skdict["Highest_Occu_U"], skp.skdict["Highest_Occu_U"])
    assert params.r_max["C"] > 0
