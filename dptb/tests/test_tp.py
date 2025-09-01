from ase.io import read
from dptb.data import AtomicData, AtomicDataDict
from dptb.nn.tensor_product import SO2_Linear
from e3nn.o3 import Irreps, wigner_D, xyz_to_angles
import torch

# def test_so2_reflection(ir=Irreps("1x0e+1x1o+1x2e+1x3o+1x4e")):
#     so2l = SO2_Linear(
#         ir,
#         ir
#     )
#     a = torch.randn(2,ir.dim)
#     slices = ir.slices()
#     for i, (mul, (l, p)) in enumerate(ir):
#         if p == 1:
#             a[1][slices[i]] = a[0][slices[i]]
#         else:
#             a[1][slices[i]] = -a[0][slices[i]]
#     R = torch.randn(2,3)
#     R[1] = -R[0]
#     R = R / R.norm(dim=1, keepdim=True)
#     out = so2l(a, R)

#     return out

def test_so2_rotation(ir=Irreps("2x0e+3x1o+4x2e+5x3o+6x4e")):
    so2l = SO2_Linear(
        ir,
        ir
    )
    a = torch.randn(2,ir.dim)
    slices = ir.slices()

    R = torch.randn(2,3)
    R = R / R.norm(dim=1, keepdim=True)

    vec = torch.randn(3)
    vec /= vec.norm()
    alpha, beta = xyz_to_angles(vec[[1,2,0]])

    for i, (mul, (l, p)) in enumerate(ir):
        rot_mat_L = wigner_D(l, alpha, beta, torch.tensor(0.))
        a[1][slices[i]] = (rot_mat_L @ (a[0][slices[i]]).reshape(mul, 2*l+1).T).T.reshape(-1)

    rot_mat_L = wigner_D(1, alpha, beta, torch.tensor(0.))
    R[1] = (rot_mat_L @ R[0][[1,2,0]])[[2,0,1]]

    out = so2l(a, R)

    R[1] = (R[1][[1,2,0]] @ rot_mat_L)[[2,0,1]]
    for i, (mul, (l, p)) in enumerate(ir):
        rot_mat_L = wigner_D(l, alpha, beta, torch.tensor(0.))
        out[1][slices[i]] = (out[1][slices[i]].reshape(mul, 2*l+1) @ rot_mat_L).reshape(-1)

    assert torch.allclose(out[1], out[0], atol=5e-5), "SO2 rotation test failed"