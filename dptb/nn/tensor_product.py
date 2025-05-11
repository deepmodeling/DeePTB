import os
import math
import torch
import torch.nn as nn
from torch.nn import Linear
from typing import List, Optional, Tuple
from e3nn.o3 import xyz_to_angles, Irreps
from e3nn.util.jit import compile_mode


_Jd_file = os.path.join(os.path.dirname(__file__), "Jd.pt")
if os.path.exists(_Jd_file):
    _Jd = torch.load(_Jd_file)
else:
    print(f"Warning: Jd.pt not found at {_Jd_file}. Wigner D functions will fail.")
    _Jd = []


def wigner_D(l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
    if not _Jd:
        raise RuntimeError("Jd.pt was not loaded. Cannot compute Wigner D matrices.")
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
        if not _Jd:
            raise RuntimeError("Jd.pt was not loaded. SO2_Linear cannot be initialized.")
        self.Jd: List[torch.Tensor] = _Jd

        irreps_in_s = irreps_in.simplify()
        irreps_out_s = (Irreps(f"{extra_m0_outsize}x0e") + irreps_out).simplify()

        self.irreps_out: Irreps = irreps_out_s
        self.in_dim = irreps_in_s.dim
        self.out_dim = irreps_out_s.dim
        self.in_num_irreps = irreps_in_s.num_irreps
        self.out_num_irreps = irreps_out_s.num_irreps
        self.has_radial = radial_emb

        if radial_channels is None:
            radial_channels = []

        in_offsets_list: List[int] = []
        in_mul_list: List[int] = []
        in_l_list: List[int] = []
        current_offset = 0
        for mul, (l, p_val) in irreps_in_s:
            in_offsets_list.append(current_offset)
            in_mul_list.append(mul)
            in_l_list.append(l)
            current_offset += mul * (2 * l + 1)
        in_offsets_list.append(current_offset)
        self.register_buffer('in_offsets', torch.tensor(in_offsets_list, dtype=torch.long))
        self.register_buffer('in_mul', torch.tensor(in_mul_list, dtype=torch.long))
        self.register_buffer('in_l', torch.tensor(in_l_list, dtype=torch.long))

        out_offsets_list: List[int] = []
        out_mul_list: List[int] = []
        out_l_list: List[int] = []
        current_offset = 0
        for mul, (l, p_val) in irreps_out_s:
            out_offsets_list.append(current_offset)
            out_mul_list.append(mul)
            out_l_list.append(l)
            current_offset += mul * (2 * l + 1)
        out_offsets_list.append(current_offset)
        self.register_buffer('out_offsets', torch.tensor(out_offsets_list, dtype=torch.long))
        self.register_buffer('out_mul', torch.tensor(out_mul_list, dtype=torch.long))
        self.register_buffer('out_l', torch.tensor(out_l_list, dtype=torch.long))

        m_in_mask = torch.zeros(irreps_in_s.lmax + 1, self.in_dim, dtype=torch.bool)
        cnt_list = [0] * (irreps_in_s.lmax + 1)

        current_offset_for_mask = 0
        for i in range(len(irreps_in_s)):
            mul, (l, p_val) = irreps_in_s[i]
            for k_mul in range(mul):
                base_idx = current_offset_for_mask + k_mul * (2 * l + 1)
                for m_val in range(l + 1):
                    if m_val == 0:
                        m_in_mask[m_val, base_idx + l] = True
                        cnt_list[m_val] += 1
                    else:
                        m_in_mask[m_val, base_idx + l + m_val] = True
                        m_in_mask[m_val, base_idx + l - m_val] = True
                        cnt_list[m_val] += 1
            current_offset_for_mask += mul * (2 * l + 1)
        self.register_buffer('m_in_mask', m_in_mask)
        self.register_buffer('cnt', torch.tensor(cnt_list, dtype=torch.long))

        m_idx = torch.cat([torch.tensor([0], dtype=torch.long), torch.cumsum(self.cnt, dim=0)])
        self.register_buffer('m_idx', m_idx)

        m_out_mask = torch.zeros(irreps_out_s.lmax + 1, self.out_dim, dtype=torch.bool)
        current_offset_for_mask = 0
        for i in range(len(irreps_out_s)):
            mul, (l, p_val) = irreps_out_s[i]
            for k_mul in range(mul):
                base_idx = current_offset_for_mask + k_mul * (2 * l + 1)
                for m_val in range(l + 1):
                    if m_val <= irreps_in_s.lmax:
                        if m_val == 0:
                            m_out_mask[m_val, base_idx + l] = True
                        else:
                            m_out_mask[m_val, base_idx + l + m_val] = True
                            m_out_mask[m_val, base_idx + l - m_val] = True
            current_offset_for_mask += mul * (2 * l + 1)
        self.register_buffer('m_out_mask', m_out_mask)

        self.fc0 = Linear(self.in_num_irreps, self.out_num_irreps, bias=True)

        self.m_linears = nn.ModuleList([
            SO2_m_Linear(m, irreps_in_s, irreps_out_s) for m in range(1, irreps_out_s.lmax + 1)
        ])

        if self.has_radial:
            if latent_dim <= 0:
                raise ValueError("latent_dim must be > 0 if radial_emb is True")
            layers_list: List[nn.Module] = []
            current_ch_radial = latent_dim
            all_radial_net_channels = radial_channels + [int(m_idx[-1].item())]
            for i, next_ch_radial in enumerate(all_radial_net_channels):
                layers_list.append(Linear(current_ch_radial, next_ch_radial, bias=True))
                current_ch_radial = next_ch_radial
                if i < len(all_radial_net_channels) - 1:
                    layers_list.append(nn.LayerNorm(next_ch_radial))
                    layers_list.append(nn.SiLU())
            self.radial: nn.Module = nn.Sequential(*layers_list)
        else:
            self.radial: nn.Module = nn.Identity()  # Explicitly type self.radial here for clarity

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

        # MODIFIED PART FOR w CALCULATION
        w: Optional[torch.Tensor] = None
        if self.has_radial:
            if latents is None:
                raise RuntimeError("`latents` must be provided and be a Tensor when `radial_emb=True`")
            w = self.radial(latents)
            # END OF MODIFIED PART

        x_rot = x.clone()
        for i in range(len(self.in_mul)):
            start = int(self.in_offsets[i].item())
            end = int(self.in_offsets[i + 1].item())
            mul = int(self.in_mul[i].item())
            l_val = int(self.in_l[i].item())

            if l_val > 0:
                rot_mat = self._wigner(l_val, alpha, beta, gamma)
                vals = x_rot[:, start:end].reshape(n, mul, 2 * l_val + 1)
                rotated_vals = torch.einsum('nji,nmj->nmi', rot_mat, vals)
                x_rot[:, start:end] = rotated_vals.reshape(n, -1)

        out = x.new_zeros(n, self.out_dim)

        seg0_raw = x_rot[:, self.m_in_mask[0]]
        seg0_for_fc0 = seg0_raw.clone()

        current_col_in_seg0 = 0
        for i_irrep in range(len(self.in_l)):
            l_val_of_input_irrep = int(self.in_l[i_irrep].item())
            mul_of_input_irrep = int(self.in_mul[i_irrep].item())

            if l_val_of_input_irrep == 0:
                seg0_for_fc0[:, current_col_in_seg0: current_col_in_seg0 + mul_of_input_irrep] = 0.0
            current_col_in_seg0 += mul_of_input_irrep

        if w is not None:
            start_w = int(self.m_idx[0].item())
            end_w = int(self.m_idx[1].item())
            w_m0 = w[:, start_w:end_w]
            if seg0_for_fc0.size(1) == w_m0.size(1):  # Ensure dimensions match for broadcasting/element-wise mul
                seg0_for_fc0 = seg0_for_fc0 * w_m0
            elif seg0_for_fc0.size(1) != 0 and w_m0.size(1) != 0:  # Both non-zero but mismatch
                raise RuntimeError(
                    f"Dimension mismatch for radial weights at m=0: seg0 has {seg0_for_fc0.size(1)}, w_m0 has {w_m0.size(1)}")
            # If one is zero dim, multiplication might be okay or do nothing, depends on exact case.
            # For safety, only multiply if dims match and are non-zero. If seg0 is empty, w_m0 should also be.

        out[:, self.m_out_mask[0]] += self.fc0(seg0_for_fc0)

        for idx, m_linear_layer in enumerate(self.m_linears):
            m_val = idx + 1
            if self.m_in_mask[m_val].any():
                seg_m = x_rot[:, self.m_in_mask[m_val]].reshape(n, 2, -1)

                if w is not None:
                    start_w = int(self.m_idx[m_val].item())
                    end_w = int(self.m_idx[m_val + 1].item())
                    w_slice = w[:, start_w:end_w]
                    if seg_m.size(2) == w_slice.size(1) and seg_m.size(2) > 0:
                        seg_m = seg_m * w_slice.unsqueeze(1)
                    elif seg_m.size(2) != 0 and w_slice.size(1) != 0:
                        raise RuntimeError(
                            f"Dimension mismatch for radial weights at m={m_val}: seg_m has {seg_m.size(2)}, w_slice has {w_slice.size(1)}")

                processed_seg_m = m_linear_layer(seg_m).reshape(n, -1)
                out[:, self.m_out_mask[m_val]] += processed_seg_m

        for i in range(len(self.out_mul)):
            start = int(self.out_offsets[i].item())
            end = int(self.out_offsets[i + 1].item())
            mul = int(self.out_mul[i].item())
            l_val = int(self.out_l[i].item())

            if l_val > 0:
                rot_mat = self._wigner(l_val, alpha, beta, gamma)
                vals = out[:, start:end].reshape(n, mul, 2 * l_val + 1)
                out[:, start:end] = torch.einsum('nji,nmj->nmi', rot_mat, vals).reshape(n, -1)

        return out


@compile_mode("script")
class SO2_m_Linear(nn.Module):
    def __init__(self, m: int, irreps_in_s: Irreps, irreps_out_s: Irreps):
        super().__init__()
        num_in = sum(mul for mul, (l, p_val) in irreps_in_s if l >= m)
        num_out = sum(mul for mul, (l, p_val) in irreps_out_s if l >= m)

        self.fc = Linear(num_in, 2 * num_out, bias=False)
        if num_in > 0 and num_out > 0:
            self.fc.weight.data.mul_(1.0 / math.sqrt(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(2) == 0:
            if self.fc.out_features == 0:
                return torch.empty((x.size(0), 2, 0), dtype=x.dtype, device=x.device)
            # If num_in is 0, but num_out > 0, fc(x) will still produce output of shape [N, 2, 2*num_out]
            # where the input to fc was effectively zeros.
            # So, proceed with fc(x) even if x.size(2) == 0, as fc handles it.

        y = self.fc(x)

        num_out_channels = y.size(2) // 2
        if num_out_channels == 0:
            return torch.empty((x.size(0), 2, 0), dtype=x.dtype, device=x.device)

        out_re = y[:, 0, :num_out_channels] - y[:, 1, num_out_channels:]
        out_im = y[:, 0, num_out_channels:] + y[:, 1, :num_out_channels]

        return torch.stack((out_re, out_im), dim=1)

