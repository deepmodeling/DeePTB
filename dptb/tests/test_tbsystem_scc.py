from pathlib import Path

import numpy as np
import pytest
import torch

from dptb.nn.build import build_model
from dptb.nn.dftb.scc_params import SCCParams
from dptb.nn.dftb.sk_param import SKParam
from dptb.postprocess.unified.system import TBSystem


ROOT = Path(__file__).resolve().parents[2]


def test_tbsystem_returns_scc_corrected_hk_for_orthogonal_nnsk():
    ckpt = ROOT / "examples" / "hBN_dftb" / "nnsk" / "checkpoint" / "nnsk.ep500.pth"
    struct = ROOT / "examples" / "dftb_scc" / "hBN_scc" / "data" / "struct.vasp"
    sk_path = ROOT / "examples" / "dftb_scc" / "hBN_scc" / "slakos"
    if not ckpt.exists() or not struct.exists() or not sk_path.exists():
        pytest.skip("hBN NNSK-SCC example data is not available.")

    model = build_model(str(ckpt))
    system = TBSystem(data=str(struct), calculator=model)
    skp = SKParam(basis=model.basis, skdata=str(sk_path), cal_rcuts=True, dtype=torch.float64)
    params = SCCParams.from_skparam(skp)

    system.enable_scc(
        params=params,
        nel_atom={"B": 3, "N": 5},
        kmeshgrid=[2, 2, 1],
        kgamma_center=True,
        krotational_symmetry=False,
        ktime_inversion_symmetry=True,
        AtomicData_options={"r_max": params.r_max},
        max_iter=5,
        mix_rate=0.25,
        Temp=0.1,
        smearing_method="Fermi-Dirac",
    )
    scc_state = system.run_scc()

    k_points = np.array([[0.0, 0.0, 0.0]], dtype=float)
    hk_bare, sk_bare = system.get_hk(k_points=k_points, use_scc=False)
    hk_scc, sk_scc = system.get_hk(k_points=k_points, use_scc=True)
    data, eigs, vecs = system.get_eigenstates(k_points=k_points, use_scc=True)
    system.band.set_kpath(
        method="array",
        kpath=k_points,
        labels=["G"],
        xlist=np.array([0.0]),
        high_sym_kpoints=np.array([0.0]),
    )
    band_data = system.band.compute()

    assert system.has_scc
    assert scc_state["scc_shift"] is not None
    assert sk_bare is None
    assert sk_scc is None
    assert hk_scc.shape == hk_bare.shape
    assert not torch.allclose(hk_scc, hk_bare.to(dtype=hk_scc.dtype))
    assert eigs.shape[0] == 1
    assert vecs.shape[0] == 1
    assert torch.isfinite(eigs).all()
    assert np.allclose(band_data.eigenvalues, eigs.detach().cpu().numpy())
