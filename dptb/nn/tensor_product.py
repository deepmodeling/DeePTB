import os
import math
import torch
import torch.nn as nn
from torch.nn import Linear
from typing import List, Optional
from e3nn.o3 import xyz_to_angles, Irreps
from e3nn.util.jit import compile_mode

_Jd_file = os.path.join(os.path.dirname(__file__), "Jd.pt")
if os.path.exists(_Jd_file):
    _Jd = torch.load(_Jd_file)
else:
    raise RuntimeError(f"Jd.pt not found at {_Jd_file}. Wigner D functions will fail.")


def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if not l < len(_Jd):
        raise NotImplementedError(
            f"wigner D maximum l implemented is {len(_Jd) - 1}, send us an email to ask for more"
        )
    alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
    J = _Jd[l].to(dtype=alpha.dtype, device=alpha.device)
    Xa = _z_rot(alpha, l)
    Xb = _z_rot(beta, l)
    Xc = _z_rot(gamma, l)
    return Xa @ J @ Xb @ J @ Xc


@torch.jit.script
def _z_rot(angle: torch.Tensor, l: int) -> torch.Tensor:
    freqs = torch.arange(l, -l - 1, -1, dtype=angle.dtype, device=angle.device)
    diag_elements = torch.cos(freqs * angle.unsqueeze(-1))
    anti_diag_elements = torch.sin(freqs * angle.unsqueeze(-1))
    diag_matrix = torch.diag_embed(diag_elements)
    anti_diag_matrix = torch.flip(torch.diag_embed(anti_diag_elements), dims=[-1])
    return diag_matrix + anti_diag_matrix


@compile_mode("script")
class SO2_Linear(nn.Module):
    in_dim: int
    out_dim: int
    in_num_irreps: int
    out_num_irreps: int
    has_radial: bool

    def __init__(
            self,
            irreps_in: Irreps,
            irreps_out: Irreps,
            radial_emb: bool = False,
            latent_dim: int = 0,
            radial_channels: Optional[List[int]] = None,
            extra_m0_outsize: int = 0,
    ):
        super().__init__()
        self.Jd: List[torch.Tensor] = _Jd

        irreps_in_s = irreps_in.simplify()
        irreps_out_s = (Irreps(f"{extra_m0_outsize}x0e") + irreps_out).simplify()

        self.irreps_out = irreps_out_s
        self.in_dim = irreps_in_s.dim
        self.out_dim = irreps_out_s.dim
        self.in_num_irreps = irreps_in_s.num_irreps
        self.out_num_irreps = irreps_out_s.num_irreps
        self.has_radial = radial_emb

        # Buffers for irreps layout
        in_offsets, in_mul, in_l = [], [], []
        offset = 0
        for mul, (l, _) in irreps_in_s:
            in_offsets.append(offset)
            in_mul.append(mul)
            in_l.append(l)
            offset += mul * (2 * l + 1)
        in_offsets.append(offset)
        self.register_buffer('in_offsets', torch.tensor(in_offsets, dtype=torch.long))
        self.register_buffer('in_mul', torch.tensor(in_mul, dtype=torch.long))
        self.register_buffer('in_l', torch.tensor(in_l, dtype=torch.long))

        out_offsets, out_mul, out_l = [], [], []
        offset = 0
        for mul, (l, _) in irreps_out_s:
            out_offsets.append(offset)
            out_mul.append(mul)
            out_l.append(l)
            offset += mul * (2 * l + 1)
        out_offsets.append(offset)
        self.register_buffer('out_offsets', torch.tensor(out_offsets, dtype=torch.long))
        self.register_buffer('out_mul', torch.tensor(out_mul, dtype=torch.long))
        self.register_buffer('out_l', torch.tensor(out_l, dtype=torch.long))

        # m-in mask and count
        m_in_mask = torch.zeros(irreps_in_s.lmax + 1, self.in_dim, dtype=torch.bool)
        cnt_list = [0] * (irreps_in_s.lmax + 1)
        cur = 0
        for mul, (l, _) in irreps_in_s:
            for k in range(mul):
                base = cur + k * (2 * l + 1)
                for m_val in range(l + 1):
                    if m_val == 0:
                        m_in_mask[m_val, base + l] = True
                        cnt_list[m_val] += 1
                    else:
                        m_in_mask[m_val, base + l + m_val] = True
                        m_in_mask[m_val, base + l - m_val] = True
                        cnt_list[m_val] += 1
            cur += mul * (2 * l + 1)
        self.register_buffer('m_in_mask', m_in_mask)
        self.register_buffer('cnt', torch.tensor(cnt_list, dtype=torch.long))
        self.register_buffer('m_idx', torch.cat([torch.tensor([0], dtype=torch.long), torch.cumsum(torch.tensor(cnt_list, dtype=torch.long), dim=0)]))

        # m-out mask
        m_out_mask = torch.zeros(irreps_out_s.lmax + 1, self.out_dim, dtype=torch.bool)
        cur = 0
        for mul, (l, _) in irreps_out_s:
            for k in range(mul):
                base = cur + k * (2 * l + 1)
                for m_val in range(l + 1):
                    if m_val <= irreps_in_s.lmax:
                        if m_val == 0:
                            m_out_mask[m_val, base + l] = True
                        else:
                            m_out_mask[m_val, base + l + m_val] = True
                            m_out_mask[m_val, base + l - m_val] = True
            cur += mul * (2 * l + 1)
        self.register_buffer('m_out_mask', m_out_mask)

        # fc0 and m_linears
        self.fc0 = Linear(self.in_num_irreps, self.out_num_irreps, bias=True)
        self.m_linears = nn.ModuleList([SO2_m_Linear(mv, irreps_in_s, irreps_out_s) for mv in range(1, irreps_out_s.lmax + 1)])

        # radial embedding
        if self.has_radial:
            layers_list: List[nn.Module] = []
            current_dim = latent_dim
            all_radial_layer_dims = (radial_channels if radial_channels is not None else []) + [int(self.m_idx[-1].item())]
            for i, out_ch in enumerate(all_radial_layer_dims):
                layers_list.append(Linear(current_dim, out_ch, bias=True))
                current_dim = out_ch
                if i < len(all_radial_layer_dims) - 1:  # Not the last layer
                    layers_list.append(nn.LayerNorm(out_ch))
                    layers_list.append(nn.SiLU())
            self.radial = nn.Sequential(*layers_list)
        else:
            self.radial = nn.Identity()

    def _wigner(self, l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        J = self.Jd[l].to(dtype=alpha.dtype, device=alpha.device)
        return _z_rot(alpha, l) @ J @ _z_rot(beta, l) @ J @ _z_rot(gamma, l)

    def forward(
            self,
            x: torch.Tensor,
            R: torch.Tensor,
            latents: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        n = x.size(0)
        alpha, beta = xyz_to_angles(R[:, [1, 2, 0]])
        gamma = torch.zeros_like(alpha)

        # initialize radial weights tensor to empty or computed
        w = torch.ones(n, int(self.m_idx[-1].item()), dtype=x.dtype, device=x.device)
        if self.has_radial:
            if latents is None:
                raise RuntimeError("`latents` must be provided when `radial_emb=True`")
            w = self.radial(latents)

        # initialize x_rot to zero
        x_rot = torch.zeros_like(x)
        for i in range(len(self.in_mul)):
            start = int(self.in_offsets[i].item())
            end = int(self.in_offsets[i + 1].item())
            mul = int(self.in_mul[i].item())
            l_val = int(self.in_l[i].item())
            if l_val > 0:
                rot = self._wigner(l_val, alpha, beta, gamma)
                vals = x[:, start:end].reshape(n, mul, 2 * l_val + 1)
                x_rot[:, start:end] = torch.einsum('nji,nmj->nmi', rot, vals).reshape(n, -1)

        out = x.new_zeros(n, self.out_dim)
        # m=0
        seg0 = x_rot[:, self.m_in_mask[0]]
        if w is not None and seg0.numel() > 0:
            seg0 = seg0 * w[:, self.m_idx[0]:self.m_idx[1]]
        out[:, self.m_out_mask[0]] += self.fc0(seg0)
        # m>0
        for idx, layer in enumerate(self.m_linears):
            m_val = idx + 1
            mask = self.m_in_mask[m_val]
            if mask.any():
                seg = x_rot[:, mask].reshape(n, 2, -1)
                if w is not None and seg.numel() > 0:
                    seg = seg * w[:, self.m_idx[m_val]:self.m_idx[m_val+1]].unsqueeze(1)
                out[:, self.m_out_mask[m_val]] += layer(seg).reshape(n, -1)
        # final rotation
        for i in range(len(self.out_mul)):
            start = int(self.out_offsets[i].item())
            end = int(self.out_offsets[i + 1].item())
            l_val = int(self.out_l[i].item())
            mul = int(self.out_mul[i].item())
            if l_val > 0:
                rot = self._wigner(l_val, alpha, beta, gamma)
                vals = out[:, start:end].reshape(n, mul, 2 * l_val + 1)
                out[:, start:end] = torch.einsum('nji,nmj->nmi', rot, vals).reshape(n, -1)
        return out


@compile_mode("script")
class SO2_m_Linear(nn.Module):
    def __init__(self, m_val: int, irreps_in_s: Irreps, irreps_out_s: Irreps):
        super().__init__()
        # count input/output channels for order m_val
        num_in = sum(mul for mul, (l, _) in irreps_in_s if l >= m_val)
        num_out = sum(mul for mul, (l, _) in irreps_out_s if l >= m_val)
        self.fc = Linear(num_in, 2 * num_out, bias=False)
        if num_in > 0 and num_out > 0:
            self.fc.weight.data.mul_(1.0 / math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        num_out = y.size(2) // 2
        re = y[:, 0, :num_out] - y[:, 1, num_out:]
        im = y[:, 0, num_out:] + y[:, 1, :num_out]
        return torch.stack((re, im), dim=1)
