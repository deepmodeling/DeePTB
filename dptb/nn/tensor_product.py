from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
import torch.nn as nn
from torch.nn import Linear
import os



_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)

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
        radial_emb: bool = False,
        latent_dim: int = None,
        radial_channels: list = None,
        extra_m0_outsize: int = 0,
    ):
        super(SO2_Linear, self).__init__()
        

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + irreps_out).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim
        self.m_linear = nn.ModuleList()
        self.fc_m0 = Linear(self.irreps_in.num_irreps, self.irreps_out.num_irreps, bias=True)
        for m in range(1, self.irreps_out.lmax + 1):
            self.m_linear.append(SO2_m_Linear(m, self.irreps_in, self.irreps_out))
        
        # generate m mask
        self.m_in_mask = torch.zeros(self.irreps_in.lmax+1, self.irreps_in.dim, dtype=torch.bool)
        self.m_out_mask = torch.zeros(self.irreps_in.lmax+1, self.irreps_out.dim, dtype=torch.bool)
        
        if self.irreps_in.dim <= self.irreps_out.dim:
            front = True
            self.m_in_num = [0] * (self.irreps_in.lmax+1)
        else:
            front = False
            self.m_in_num = [0] * (self.irreps_out.lmax+1)
    
            
        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l+1):
                self.m_in_mask[m, start_id+l+m] = True
                self.m_in_mask[m, start_id+l-m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)

        # assert sum(self.m_in_num) == self.irreps_in.dim
            
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l+1):
                if m <= self.irreps_in.lmax:
                    self.m_out_mask[m, start_id+l+m] = True
                    self.m_out_mask[m, start_id+l-m] = True
                    if not front:
                        self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)

        self.m_in_index = [0] + list(torch.cumsum(torch.tensor(self.m_in_num), dim=0))
        if radial_emb:
            self.radial_emb = RadialFunction([latent_dim]+radial_channels+[self.m_in_index[-1]])
        self.front = front

    def forward(self, x, R, latents=None):
        n, _ = x.shape

        if self.radial_emb:
            weights = self.radial_emb(latents)
        
        x_ = torch.zeros(n, self.irreps_in.dim, dtype=x.dtype, device=x.device)
        for (mul, (l,p)), slice in zip(self.irreps_in, self.irreps_in.slices()):
            if l > 0:
                angle = xyz_to_angles(R[:,[1,2,0]]) # (tensor(N), tensor(N))
                # The roataion matrix is SO3 rotation, therefore Irreps(l,1), is used here.
                rot_mat_L = wigner_D(l, angle[0], angle[1], torch.zeros_like(angle[0]))
                x_[:, slice] = torch.einsum('nji,nmj->nmi', rot_mat_L, x[:, slice].reshape(n,mul,-1)).reshape(n,-1)

        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.irreps_out.lmax+1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m+1]].unsqueeze(1) if self.radial_emb else 1.
            if m == 0:
                if self.front and self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]] * radial_weight.squeeze(1))
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]]) * radial_weight.squeeze(1)
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]])
            else:
                if self.front and self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.m_linear[m-1](x_[:, self.m_in_mask[m]].reshape(n, 2, -1)*radial_weight).reshape(n, -1)
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += (self.m_linear[m-1](x_[:, self.m_in_mask[m]].reshape(n, 2, -1))*radial_weight).reshape(n, -1)
                else:
                    out[:, self.m_out_mask[m]] += self.m_linear[m-1](x_[:, self.m_in_mask[m]].reshape(n, 2, -1)).reshape(n, -1)
                    
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

class RadialFunction(nn.Module):
    '''
        Contruct a radial function (linear layers + layer normalization + SiLU) given a list of channels
    '''
    def __init__(self, channels_list):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(len(channels_list)):
            if i == 0:
                continue
            
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            
            if i == len(channels_list) - 1:
                break
            
            modules.append(nn.LayerNorm(channels_list[i]))
            modules.append(torch.nn.SiLU())
        
        self.net = nn.Sequential(*modules)

        
    def forward(self, inputs):
        return self.net(inputs)