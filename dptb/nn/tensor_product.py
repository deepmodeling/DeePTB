from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
import torch.nn as nn
from torch.nn import Linear
import os
import torch.nn.functional as F
from collections import defaultdict

_Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)
_idx_data = torch.load(os.path.join(os.path.dirname(__file__), "z_rot_indices_lmax12.pt"), weights_only=False)


def build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, sizes):
    """
    angle_stack: (3*N, )    # Input with alpha, beta, gamma stacked together
    l_max: int

    Returns: (Xa, Xb, Xc) # Each is of shape (N, D_total, D_total)
    """
    N_all = angle_stack.shape[0]
    N = N_all // 3

    D_total = sizes.sum().item()

    # Step 1: Vectorized computation of sine and cosine values
    angle_expand = angle_stack[None, :, None]  # (1, 3N, 1)
    freq_expand = freq[:, None, :]  # (L, 1, Mmax)
    sin_val = torch.sin(freq_expand * angle_expand)  # (L, 3N, Mmax)
    cos_val = torch.cos(freq_expand * angle_expand)  # (L, 3N, Mmax)

    # Step 2: Construct the block-diagonal matrix
    M_total = angle_stack.new_zeros((N_all, D_total, D_total))
    idx_l, idx_row = torch.where(mask)  # (K,), (K,)
    idx_col_diag = idx_row
    idx_col_anti = reversed_inds[idx_l, idx_row]
    global_row = offsets[idx_l] + idx_row  # (K,)
    global_col_diag = offsets[idx_l] + idx_col_diag
    global_col_anti = offsets[idx_l] + idx_col_anti

    # Assign values to the diagonal
    M_total[:, global_row, global_col_diag] = cos_val[idx_l, :, idx_row].transpose(0, 1)
    # Assign values to non-overlapping anti-diagonals
    overlap_mask = (global_row == global_col_anti)
    M_total[:, global_row[~overlap_mask], global_col_anti[~overlap_mask]] = sin_val[idx_l[~overlap_mask], :,
                                                                            idx_row[~overlap_mask]].transpose(0, 1)

    # Step 3: Split into three components corresponding to alpha, beta, gamma
    Xa = M_total[:N]
    Xb = M_total[N:2 * N]
    Xc = M_total[2 * N:]

    return Xa, Xb, Xc


def batch_wigner_D(l_max, alpha, beta, gamma, _Jd):
    """
    Compute Wigner D matrices for all L (from 0 to l_max) in a single batch.
    Returns a tensor of shape [N, D, D], where D = sum(2l+1 for l in 0..l_max).
    """
    device = alpha.device
    N = alpha.shape[0]
    idx_data = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in _idx_data.items()}

    # Load static data
    sizes = idx_data["sizes"][:l_max + 1]
    offsets = idx_data["offsets"][:l_max + 1]
    mask = idx_data["mask"][:l_max + 1]
    freq = idx_data["freq"][:l_max + 1]
    reversed_inds = idx_data["reversed_inds"][:l_max + 1]

    # Precompute block structure information
    dims = [2 * l + 1 for l in range(l_max + 1)]
    D_total = sum(dims)

    # Construct block-diagonal J matrix
    J_full_small = torch.zeros(D_total, D_total, device=device)
    for l in range(l_max + 1):
        start = offsets[l]
        J_full_small[start:start + 2 * l + 1, start:start + 2 * l + 1] = _Jd[l]

    J_full = J_full_small.unsqueeze(0).expand(N, -1, -1)
    angle_stack = torch.cat([alpha, beta, gamma], dim=0)
    Xa, Xb, Xc = build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, sizes)

    return Xa @ J_full @ Xb @ J_full @ Xc


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


class InterpolationBlock(nn.Module):
    """
    A small MLP with two hidden layers for smooth feature transformation.
    """
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.out_features = out_features
        hidden_features1 = max(1, int(in_features * 2 / 3 + out_features * 1 / 3))
        hidden_features2 = max(1, int(in_features * 1 / 3 + out_features * 2 / 3))
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_features1, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_features1, hidden_features2, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_features2, out_features, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Convolutional layer.
    """
    def __init__(
            self,
            irreps_in,
            irreps_out,
            radial_emb: bool = False,
            latent_dim: int = None,
            radial_channels: list = None,
            extra_m0_outsize: int = 0,
            # MODIFICATION: Add a single flag to control behavior.
            use_interpolation: bool = False,
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = irreps_in.simplify()
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + irreps_out).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim
        self.m_linear = nn.ModuleList()

        # MODIFICATION: Simplified logic based solely on the `use_interpolation` flag.
        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        # if use_interpolation:
        #     self.fc_m0 = InterpolationBlock(num_in_m0, num_out_m0, bias=True)
        # else:
        #     self.fc_m0 = Linear(num_in_m0, num_out_m0, bias=True)

        self.fc_m0 = Linear(num_in_m0, num_out_m0, bias=True)

        for m in range(1, self.irreps_out.lmax + 1):
            self.m_linear.append(SO2_m_Linear(m, self.irreps_in, self.irreps_out, use_interpolation=use_interpolation))

        # --- The rest of the __init__ method is unchanged ---
        self.m_in_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_in.dim, dtype=torch.bool)
        self.m_out_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_out.dim, dtype=torch.bool)
        if self.irreps_in.dim <= self.irreps_out.dim:
            front = True
            self.m_in_num = [0] * (self.irreps_in.lmax + 1)
        else:
            front = False
            self.m_in_num = [0] * (self.irreps_out.lmax + 1)
        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l + 1):
                self.m_in_mask[m, start_id + l + m] = True
                self.m_in_mask[m, start_id + l - m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l + 1):
                if m <= self.irreps_in.lmax:
                    self.m_out_mask[m, start_id + l + m] = True
                    self.m_out_mask[m, start_id + l - m] = True
                    if not front:
                        self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        self.m_in_index = [0] + list(torch.cumsum(torch.tensor(self.m_in_num), dim=0))
        if radial_emb:
            self.radial_emb = RadialFunction([latent_dim] + radial_channels + [self.m_in_index[-1]])
        self.front = front
        self.l_max = max((l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0), default=0)
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]

    def forward(self, x, R, latents=None):
        n, _ = x.shape
        if self.radial_emb:
            weights = self.radial_emb(latents)
        x_ = torch.zeros_like(x)
        if self.l_max > 0:
            angle = xyz_to_angles(R[:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)
        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
            if l == 0:
                x_[:, slice_info] = x[:, slice_info]
        for l, group in groups.items():
            if l == 0 or not group:
                continue
            muls, slices = zip(*group)
            x_parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
            x_combined = torch.cat(x_parts, dim=1)
            start = self.offsets[l]
            rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
            transformed = torch.bmm(x_combined, rot_mat)
            for part, slice_info, mul in zip(transformed.split(muls, dim=1), slices, muls):
                x_[:, slice_info] = part.reshape(n, -1)
        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.irreps_out.lmax + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(1) if self.radial_emb else 1.
            if m == 0:
                if self.front and self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]] * radial_weight.squeeze(1))
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]]) * radial_weight.squeeze(1)
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_[:, self.m_in_mask[m]])
            else:
                x_m_in = x_[:, self.m_in_mask[m]].reshape(n, -1, 2).transpose(1, 2).contiguous()
                if self.front and self.radial_emb:
                    x_m_in.mul_(radial_weight)
                    linear_output = self.m_linear[m - 1](x_m_in)
                elif self.radial_emb:
                    linear_output = self.m_linear[m - 1](x_m_in)
                    linear_output.mul_(radial_weight)
                else:
                    linear_output = self.m_linear[m - 1](x_m_in)
                final_addition = linear_output.transpose(1, 2).contiguous().reshape(n, -1)
                out[:, self.m_out_mask[m]] += final_addition
        for (mul, (l, p)), slice_in in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                start = self.offsets[l]
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
                x_slice = out[:, slice_in].reshape(n, mul, -1)
                rotated = torch.einsum('nij,nmj->nmi', rot_mat, x_slice)
                out[:, slice_in] = rotated.reshape(n, -1)
        return out.contiguous()


class SO2_m_Linear(torch.nn.Module):
    """
    SO(2) Convolution for a specific order m > 0.
    """
    def __init__(
            self,
            m,
            irreps_in,
            irreps_out,
            # MODIFICATION: Accept the flag from the parent module.
            use_interpolation: bool = False,
    ):
        super(SO2_m_Linear, self).__init__()
        self.m = m
        self.num_in_channel = sum(mul for mul, (l, p) in irreps_in if l >= m)
        self.num_out_channel = sum(mul for mul, (l, p) in irreps_out if l >= m)

        # MODIFICATION: Simplified logic based solely on the `use_interpolation` flag.
        if use_interpolation:
            self.fc = InterpolationBlock(self.num_in_channel, 2 * self.num_out_channel, bias=False)
        else:
            self.fc = Linear(self.num_in_channel, 2 * self.num_out_channel, bias=False)
            self.fc.weight.data.mul_(1 / math.sqrt(2))

    def forward(self, x_m):
        # x_m ~ [N, 2, n_channels]
        x_m = self.fc(x_m)
        x_r = x_m.narrow(2, 0, self.num_out_channel)
        x_i = x_m.narrow(2, self.num_out_channel, self.num_out_channel)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        return torch.cat((x_m_r, x_m_i), dim=1)


class RadialFunction(nn.Module):
    '''
    A simple MLP for radial basis functions.
    '''
    def __init__(self, channels_list):
        super().__init__()
        modules = []
        input_channels = channels_list[0]
        for i in range(1, len(channels_list)):
            modules.append(nn.Linear(input_channels, channels_list[i], bias=True))
            input_channels = channels_list[i]
            if i < len(channels_list) - 1:
                modules.append(nn.LayerNorm(channels_list[i]))
                modules.append(nn.SiLU())
        self.net = nn.Sequential(*modules)

    def forward(self, inputs):
        return self.net(inputs)