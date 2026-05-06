import pytest
import torch

from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from dptb.nn.energy import Eigh


class IdentityH2K(torch.nn.Module):
    def __init__(self, out_field, dtype, device):
        super().__init__()
        self.out_field = out_field
        self.dtype = dtype
        self.device = torch.device(device)

    def forward(self, data):
        return data


class IdentityS2K(torch.nn.Module):
    def forward(self, data):
        return data


def _make_eigh(overlap=False, dtype=torch.float64):
    idp = OrbitalMapper({"H": ["1s"]}, method="e3tb", device="cpu")
    solver = Eigh(
        idp=idp,
        h_out_field=AtomicDataDict.HAMILTONIAN_KEY,
        s_out_field=AtomicDataDict.OVERLAP_KEY if overlap else None,
        dtype=dtype,
        device=torch.device("cpu"),
    )
    solver.h2k = IdentityH2K(AtomicDataDict.HAMILTONIAN_KEY, dtype, torch.device("cpu"))
    if overlap:
        solver.overlap = True
        solver.s2k = IdentityS2K()
    return solver


def _base_data(H, S=None):
    data = {
        AtomicDataDict.KPOINT_KEY: torch.zeros(H.shape[0], 3, dtype=torch.float64),
        AtomicDataDict.HAMILTONIAN_KEY: H,
    }
    if S is not None:
        data[AtomicDataDict.OVERLAP_KEY] = S
    return data


def test_eigh_numpy_no_overlap_matches_torch():
    H = torch.tensor(
        [[[2.0, 1.0 - 0.2j], [1.0 + 0.2j, 3.0]]],
        dtype=torch.complex128,
    )
    solver = _make_eigh(overlap=False, dtype=torch.float64)

    torch_out = solver(_base_data(H.clone()), eig_solver="torch")
    numpy_out = solver(_base_data(H.clone()), eig_solver="numpy")

    vals_torch = torch_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0]
    vals_numpy = numpy_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0]
    vecs_numpy = numpy_out[AtomicDataDict.EIGENVECTOR_KEY]

    assert torch.allclose(vals_numpy, vals_torch, atol=1e-12)
    assert vecs_numpy.dtype == H.dtype
    residual = H @ vecs_numpy - vecs_numpy @ torch.diag_embed(vals_numpy.to(H.real.dtype)).to(H.dtype)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)


def test_eigh_numpy_overlap_matches_torch_and_preserves_orientation():
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    A = torch.tensor([[[1.5, 0.2], [0.3, 1.2]]], dtype=torch.float64)
    S = A @ A.transpose(-1, -2) + 0.5 * torch.eye(2, dtype=torch.float64).unsqueeze(0)
    solver = _make_eigh(overlap=True, dtype=torch.float64)

    torch_out = solver(_base_data(H.clone(), S.clone()), eig_solver="torch")
    numpy_out = solver(_base_data(H.clone(), S.clone()), eig_solver="numpy")

    vals_torch = torch_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0]
    vals_numpy = numpy_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0]
    vecs_numpy = numpy_out[AtomicDataDict.EIGENVECTOR_KEY]

    assert torch.allclose(vals_numpy, vals_torch, atol=1e-12)
    assert vecs_numpy.shape == torch_out[AtomicDataDict.EIGENVECTOR_KEY].shape
    x = vecs_numpy.transpose(-1, -2)
    residual = H @ x - S @ x @ torch.diag_embed(vals_numpy)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-12)


def test_eigh_solver_validation():
    H = torch.tensor([[[2.0, 1.0], [1.0, 3.0]]], dtype=torch.float64)
    solver = _make_eigh(overlap=False, dtype=torch.float64)

    with pytest.raises(ValueError):
        solver(_base_data(H.clone()), eig_solver="scipy")

    default_out = solver(_base_data(H.clone()), eig_solver=None)
    torch_out = solver(_base_data(H.clone()), eig_solver="torch")
    assert torch.allclose(
        default_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0],
        torch_out[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0],
    )
