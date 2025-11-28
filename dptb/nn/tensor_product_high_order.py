from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
from torch.nn import Linear
import os
import torch.nn.functional as F
from collections import defaultdict
# tensor_product.py
from typing import Optional, List, Union
from torch import nn
from e3nn.o3 import Irreps, Linear as O3Linear


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




class LinearRadial(nn.Module):
    """
    极简 radial：latent -> per-m 标量权重，输出约等于 1，梯度路径很短，信息主要保留在 latent 里。
    weights = 1 + scale * W * latent
    """

    def __init__(self, latent_dim: int, out_dim: int, init_scale: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(latent_dim, out_dim)
        # 初始化为 0，只靠训练学出偏离
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # 控制调制幅度
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.get_default_dtype()))

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        # 初始时 ≈ 1，训练过程中从 1 线性偏离
        return 1.0 + self.scale * self.proj(latents)


class SO2_m_Linear(nn.Module):
    """
    对给定 m>0 的 SO(2) “频段”做通道线性混合。
    输入:  [N, 2, C_in_m]
    输出:  [N, 2, C_out_m]
    其中 2 对应 cos/sin（或实部/虚部），每个 m-block 在旋转下乘以 e^{imθ}，
    这里的线性只在通道维度上做混合，对两行共用同一个权重矩阵，从而保持 SO(2) 等变性。
    """

    def __init__(
        self,
        m: int,
        irreps_in: Irreps,
        irreps_out: Irreps,
        use_interpolation: bool = False,  # 保留接口，不使用
    ):
        super().__init__()
        assert m > 0
        self.m = m

        # 计算该 m 对应的输入/输出通道数：∑_{l >= m} mul_l
        in_ch = 0
        for mul, ir in irreps_in:
            if ir.l >= m:
                in_ch += mul

        out_ch = 0
        for mul, ir in irreps_out:
            if ir.l >= m:
                out_ch += mul

        self.in_channels = in_ch
        self.out_channels = out_ch

        if in_ch == 0 or out_ch == 0:
            # 没有对应的通道，做恒等（或恒 0）映射
            self.register_parameter("weight", None)
        else:
            self.weight = nn.Parameter(torch.empty(out_ch, in_ch))
            # Kaiming 初始化
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, 2, C_in_m]
        返回: [N, 2, C_out_m]
        """
        if self.weight is None:
            # 直接扩展/截断为原形状
            return x.new_zeros(x.size(0), 2, self.out_channels)

        # 在通道维度上做线性变换，对两行共用同一权重
        # y[n, c_out] = sum_{c_in} W[c_out, c_in] * x[n, c_in]
        # 保持第二个维度 size=2 不变
        y = torch.einsum("oc,nic->nio", self.weight, x)  # [N, 2, C_out]
        return y


class SO2_Linear(nn.Module):
    """
    SO(2) Convolutional layer.

    - 对 m=0：用一个 O3Linear(fc_m0) 在“所有 irrep 的 m=0 分量”之间做标量 mixing。
    - 对 m>0：用一组 SO2_m_Linear 分别对每个 m 频段做通道线性混合。
    - 可选 radial_emb：使用 LinearRadial 把 latent 映射到每个 m-block 的标量 weight，上下乘法门控。
    """

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        radial_emb: bool = False,
        latent_dim: int = None,
        radial_channels: list = None,  # 兼容旧接口，内部不用
        extra_m0_outsize: int = 0,
        use_interpolation: bool = False,
    ):
        super().__init__()

        irreps_in = Irreps(irreps_in).simplify()
        irreps_out = Irreps(irreps_out).simplify()

        self.irreps_in = irreps_in
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + irreps_out).simplify()
        self.radial_emb_flag = radial_emb
        self.latent_dim = latent_dim
        self.m_linear = nn.ModuleList()

        # -------- m=0 标量通道的 fc --------
        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        self.fc_m0 = Linear(num_in_m0, num_out_m0, bias=True)

        for m in range(1, self.irreps_out.lmax + 1):
            self.m_linear.append(
                SO2_m_Linear(
                    m,
                    self.irreps_in,
                    self.irreps_out,
                    use_interpolation=use_interpolation,
                )
            )

        # -------- 构造 m_in_mask / m_out_mask / m_in_index --------
        self.m_in_mask = torch.zeros(
            self.irreps_in.lmax + 1, self.irreps_in.dim, dtype=torch.bool
        )
        self.m_out_mask = torch.zeros(
            self.irreps_in.lmax + 1, self.irreps_out.dim, dtype=torch.bool
        )

        if self.irreps_in.dim <= self.irreps_out.dim:
            front = True
            self.m_in_num = [0] * (self.irreps_in.lmax + 1)
        else:
            front = False
            self.m_in_num = [0] * (self.irreps_out.lmax + 1)

        offset = 0
        for mul, (l, p) in self.irreps_in:
            start_id = offset + torch.arange(mul) * (2 * l + 1)
            for m in range(l + 1):
                self.m_in_mask[m, start_id + l + m] = True
                self.m_in_mask[m, start_id + l - m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)

        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.arange(mul) * (2 * l + 1)
            for m in range(l + 1):
                if m <= self.irreps_in.lmax:
                    self.m_out_mask[m, start_id + l + m] = True
                    self.m_out_mask[m, start_id + l - m] = True
                    if not front:
                        self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)

        self.m_in_index = [0] + list(torch.cumsum(torch.tensor(self.m_in_num), dim=0))

        # -------- radial_emb：用极简 LinearRadial 生成所有 m-block 的标量权重 --------
        if radial_emb:
            assert latent_dim is not None
            self.radial_emb = LinearRadial(
                latent_dim=latent_dim,
                out_dim=self.m_in_index[-1],
                init_scale=0.1,
            )
        else:
            self.radial_emb = None

        self.front = front
        self.l_max = max(
            (l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0),
            default=0,
        )
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]

    def forward(
        self,
        x: torch.Tensor,
        R: torch.Tensor,
        latents: Optional[torch.Tensor] = None,
        wigner_D_all: Optional[torch.Tensor] = None,
    ):
        """
        x: [N, irreps_in.dim]
        R: [N, 3, 3] 旋转矩阵
        latents: [E, latent_dim]，当 radial_emb=True 时使用
        """
        n, _ = x.shape

        # ---- radial weights ----
        if self.radial_emb is not None:
            assert latents is not None
            weights = self.radial_emb(latents)  # [N(or E), sum_m m_in_num[m]]
        else:
            weights = None

        x_ = torch.zeros_like(x)

        # ---- 构造 Wigner-D ----
        if wigner_D_all is None:
            if self.l_max > 0:
                angle = xyz_to_angles(R[:, [1, 2, 0]])
                wigner_D_all = batch_wigner_D(
                    self.l_max,
                    angle[0],
                    angle[1],
                    torch.zeros_like(angle[0]),
                    _Jd,
                )

        # ---- 先按 l 旋转各阶 ----
        from collections import defaultdict

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
            x_combined = torch.cat(x_parts, dim=1)  # [N, sum_mul, 2l+1]
            start = self.offsets[l]
            rot_mat = wigner_D_all[
                :, start : start + self.dims[l], start : start + self.dims[l]
            ]  # [N, 2l+1, 2l+1]
            transformed = torch.bmm(x_combined, rot_mat)  # [N, sum_mul, 2l+1]
            for part, slice_info, mul in zip(
                transformed.split(muls, dim=1), slices, muls
            ):
                x_[:, slice_info] = part.reshape(n, -1)

        # ---- 按 m 处理 ----
        out = torch.zeros(
            n, self.irreps_out.dim, dtype=x.dtype, device=x.device
        )

        for m in range(self.irreps_out.lmax + 1):
            if self.radial_emb is not None:
                radial_weight = weights[
                    :, self.m_in_index[m] : self.m_in_index[m + 1]
                ].unsqueeze(1)  # [N,1,C_m]
            else:
                radial_weight = None

            if m == 0:
                # m=0: 把所有 irrep 的 m=0 视作标量特征，做一次 fc
                x_m = x_[:, self.m_in_mask[m]]  # [N, C_in_m0]
                if radial_weight is not None:
                    if self.front:
                        x_m = x_m * radial_weight.squeeze(1)
                        out[:, self.m_out_mask[m]] += self.fc_m0(x_m)
                    else:
                        out[:, self.m_out_mask[m]] += (
                            self.fc_m0(x_m) * radial_weight.squeeze(1)
                        )
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(x_m)
            else:
                # m>0: 形状 [N,2,C_in_m]
                x_m_in = (
                    x_[:, self.m_in_mask[m]]
                    .reshape(n, -1, 2)
                    .transpose(1, 2)
                    .contiguous()
                )
                if radial_weight is not None:
                    if self.front:
                        x_m_in = x_m_in * radial_weight  # [N,2,C_m]
                        linear_output = self.m_linear[m - 1](x_m_in)
                    else:
                        linear_output = self.m_linear[m - 1](x_m_in)
                        linear_output = linear_output * radial_weight
                else:
                    linear_output = self.m_linear[m - 1](x_m_in)

                final_addition = (
                    linear_output.transpose(1, 2)
                    .contiguous()
                    .reshape(n, -1)
                )
                out[:, self.m_out_mask[m]] += final_addition

        # ---- 输出再按 l 旋转回去 ----
        for (mul, (l, p)), slice_in in zip(
            self.irreps_out, self.irreps_out.slices()
        ):
            if l > 0:
                start = self.offsets[l]
                rot_mat = wigner_D_all[
                    :, start : start + self.dims[l], start : start + self.dims[l]
                ]
                x_slice = out[:, slice_in].reshape(n, mul, -1)
                rotated = torch.einsum("nij,nmj->nmi", rot_mat, x_slice)
                out[:, slice_in] = rotated.reshape(n, -1)

        return out.contiguous(), wigner_D_all