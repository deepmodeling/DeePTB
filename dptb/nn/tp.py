from e3nn.o3 import xyz_to_angles, Irrep
import math
import torch
import torch.nn as nn
from torch.nn import Linear
import os



_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"))

def wigner_D(l, alpha, beta, gamma):
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )

    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot_mat(alpha, l)
    Xb = _z_rot_mat(beta, l)
    Xc = _z_rot_mat(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


def _z_rot_mat(angle, l):
    shape, device, dtype = angle.shape, angle.device, angle.dtype
    M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
    inds = torch.arange(0, 2 * l + 1, 1, device=device)
    reversed_inds = torch.arange(2 * l, -1, -1, device=device)
    frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
    M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
    M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
    return M

class SO2_Linear(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        irreps_in,
        irreps_out,
    ):
        super(SO2_Linear, self).__init__()
        

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = irreps_out.simplify()
        self.m_linear = nn.ModuleList()
        self.fc_m0 = Linear(self.irreps_in.num_irreps, self.irreps_out.num_irreps, bias=True)
        for m in range(1, self.irreps_out.lmax + 1):
            self.m_linear.append(SO2_m_Linear(m, self.irreps_in, self.irreps_out))
        
        # generate m mask
        self.m_in_mask = torch.zeros(self.irreps_in.lmax+1, self.irreps_in.dim, dtype=torch.bool)
        self.m_out_mask = torch.zeros(self.irreps_in.lmax+1, self.irreps_out.dim, dtype=torch.bool)
        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l+1):
                self.m_in_mask[m, start_id+l+m] = True
                self.m_in_mask[m, start_id+l-m] = True
            offset += mul * (2 * l + 1)
        
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l+1):
                if m <= self.irreps_in.lmax:
                    self.m_out_mask[m, start_id+l+m] = True
                    self.m_out_mask[m, start_id+l-m] = True
            offset += mul * (2 * l + 1)

    def forward(self, x, R):
        n, _ = x.shape
        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.irreps_out.lmax+1):
            if m == 0:
                out[:, self.m_out_mask[m]] += self.fc_m0(x[:, self.m_in_mask[m]])
            else:
                out[:, self.m_out_mask[m]] += self.m_linear[m-1](x[:, self.m_in_mask[m]].reshape(n, 2, -1)).reshape(n, -1)
        out.contiguous()

        for (mul, (l,p)), slice in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                angle = xyz_to_angles(R[:,[1,2,0]]) # (tensor(N), tensor(N))
                # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
                rot_mat_L = wigner_D(l, angle[0], angle[1], torch.zeros_like(angle[0]))
                out[:, slice] = torch.einsum('nij,nmj->nmi', rot_mat_L, out[:, slice].reshape(n,mul,-1)).reshape(n,-1)

        return out

class SO2_m_Linear(torch.nn.Module):
    """
    SO(2) Conv: Perform an SO(2) convolution on features corresponding to +- m

    Args:
        m (int):                    Order of the spherical harmonic coefficients
        sphere_channels (int):      Number of spherical channels
        m_output_channels (int):    Number of output channels used during the SO(2) conv
        lmax_list (list:int):       List of degrees (l) for each resolution
        mmax_list (list:int):       List of orders (m) for each resolution
    """
    def __init__(
        self,
        m,
        irreps_in,
        irreps_out,
    ):
        super(SO2_m_Linear, self).__init__()
        
        self.m = m
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        assert self.irreps_in.lmax >= m
        assert self.irreps_out.lmax >= m

        self.num_in_channel = 0
        for mul, (l, p) in self.irreps_in:
            if l >= m:
                self.num_in_channel += mul

        self.num_out_channel = 0
        for mul, (l, p) in self.irreps_out:
            if l >= m:
                self.num_out_channel += mul

        self.fc = Linear(self.num_in_channel,
            2 * self.num_out_channel,
            bias=False)
        self.fc.weight.data.mul_(1 / math.sqrt(2))
 
    def forward(self, x_m):
        # x_m ~ [N, 2, n_irreps_m]
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.fc.out_features // 2)
        x_i = x_m.narrow(2, self.fc.out_features // 2, self.fc.out_features // 2)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1) #x_r[:, 0] - x_i[:, 1]
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1) #x_r[:, 1] + x_i[:, 0]
        x_out = torch.cat((x_m_r, x_m_i), dim=1)
        
        return x_out
