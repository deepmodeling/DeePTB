from e3nn.o3 import xyz_to_angles, Irreps
import math
import torch
import torch.nn as nn
from e3nn.o3 import Linear as e3nn_Linear
from torch.nn import Linear
import os
import torch.nn.functional as F
from collections import defaultdict
from .tensor_product import InterpolationBlock, RadialFunction

# Load helpers (Keep original logic)
try:
    _Jd = torch.load(os.path.join(os.path.dirname(__file__), "Jd.pt"), weights_only=False)
    _idx_data = torch.load(os.path.join(os.path.dirname(__file__), "z_rot_indices_lmax12.pt"), weights_only=False)
except (FileNotFoundError, RuntimeError):
    # Fallback for dry-run or missing files
    _Jd = []
    _idx_data = {}


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


# ------------------------------------------------------------------------------
# MOLE COMPONENTS (Added)
# ------------------------------------------------------------------------------

class MOLEGlobals:
    """Stores routing information for the current forward pass."""

    def __init__(self, coefficients=None, sizes=None):
        self.coefficients = coefficients  # [Batch, Num_Experts]
        self.sizes = sizes  # [Batch] (Edge counts per system)


class MOLERouterV3(nn.Module):
    """
    DeepSeek-V3 Style Router Implementation.

    Features:
    1. Sigmoid Affinity: Uses Sigmoid instead of Softmax for independent expert scoring.
    2. Aux-Loss-Free Balancing: Uses dynamic bias updates instead of auxiliary loss.
    3. L1 Normalization: Normalizes selected expert weights to sum to 1.
    4. Monitoring: Returns mean(max_prob) instead of z-loss for easier interpretation.
    """

    def __init__(self, in_features, num_experts=24, top_k=1,
                 aux_loss_free=True, bias_update_speed=0.001):
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss_free = aux_loss_free
        self.bias_update_speed = bias_update_speed

        # 1. 路由网络: Linear -> SiLU -> Linear
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )

        # 2. 专家偏置 (Buffer): 不参与梯度下降，手动更新
        self.register_buffer('expert_bias', torch.zeros(num_experts))

        # ======= 找到 MOLERouterV3 类中的 forward 方法 =======

    def forward(self, global_features):
        # global_features: [Batch, Dim]

        # --- Step 1: 计算基础分数 (Sigmoid) ---
        logits = self.net(global_features)
        scores = torch.sigmoid(logits)  # [Batch, Num_Experts]

        # --- Step 2: 准备路由依据 (Routing Scores) ---
        if self.aux_loss_free and self.training:
            scores_for_selection = scores + self.expert_bias
        else:
            scores_for_selection = scores

        # --- Step 3: Top-K 选择 ---
        if self.top_k is not None:
            topk_scores_biased, topk_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1)

            # [新增修改] --- 无论是否 training，都提取专家负载统计作监控 ---
            with torch.no_grad():
                mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
                current_load = mask.sum(dim=(0, 1))  # [Num_Experts]
                # 计算负载变异系数 CV = std / mean。值为0代表绝对均衡。
                expert_load_cv = current_load.std() / (current_load.mean() + 1e-8)

            # --- Step 4: 动态负载均衡更新 (Aux-Loss-Free Update) ---
            if self.aux_loss_free and self.training:
                with torch.no_grad():
                    # 4.2 计算目标负载 (假设完全均匀)
                    batch_size = scores.size(0)
                    target_load = (batch_size * self.top_k) / self.num_experts

                    # 4.3 动态更新 Bias
                    error = current_load - target_load
                    self.expert_bias -= torch.sign(error) * self.bias_update_speed
                    self.expert_bias -= self.expert_bias.mean()

            # --- Step 5: 计算最终权重 (基于原始 Sigmoid 分数) ---
            topk_scores_original = torch.gather(scores, 1, topk_indices)
            denominators = topk_scores_original.sum(dim=-1, keepdim=True) + 1e-8
            topk_probs = topk_scores_original / denominators

            # --- Step 6: 构建稀疏输出系数 ---
            coeffs = torch.zeros_like(scores)
            coeffs.scatter_(1, topk_indices, topk_probs)

            # --- Step 7: 监控指标 (Mean Max Probability) ---
            monitor_val = topk_probs.max(dim=-1)[0].mean().detach()

            # [修改返回值] 多返回一个 expert_load_cv
            return coeffs, monitor_val, expert_load_cv.detach()

        else:
            # Fallback: Dense Mode (全激活，调试用)
            denominators = scores.sum(dim=-1, keepdim=True) + 1e-8
            probs = scores / denominators
            monitor_val = probs.max(dim=-1)[0].mean().detach()
            # [修改返回值] Fallback时补0
            return probs, monitor_val, torch.tensor(0.0, device=scores.device)


class MOLELinear(nn.Module):
    """
    DeepSeek-V3 Style Expert Layer with Linear Experts.

    Structure: Output = Routed_Experts(x) + Shared_Experts(x)
    Optimization: Merges Shared Weights into Routed Weights for 0 extra inference overhead.
    """

    def __init__(self, in_features, out_features, num_experts=8,
                 use_shared_expert=True, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert

        # 1. 路由专家权重
        self.weight_experts = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        if bias:
            self.bias_experts = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter('bias_experts', None)

        # 2. 共享专家权重 (Shared Expert)
        if self.use_shared_expert:
            self.weight_shared = nn.Parameter(torch.empty(out_features, in_features))
            if bias:
                self.bias_shared = nn.Parameter(torch.empty(out_features))
            else:
                self.register_parameter('bias_shared', None)

        self.reset_parameters()

    def reset_parameters(self):
        k = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight_experts, -k, k)
        if self.bias_experts is not None:
            nn.init.uniform_(self.bias_experts, -k, k)

        if self.use_shared_expert:
            nn.init.uniform_(self.weight_shared, -k, k)
            if self.bias_shared is not None:
                nn.init.uniform_(self.bias_shared, -k, k)

    def forward(self, x, mole_globals: MOLEGlobals):
        # 安全回退
        if mole_globals is None or mole_globals.coefficients is None:
            w_avg = self.weight_experts.mean(0)
            if self.use_shared_expert:
                w_avg = w_avg + self.weight_shared
            b_avg = None
            if self.bias_experts is not None:
                b_avg = self.bias_experts.mean(0)
                if self.use_shared_expert and self.bias_shared is not None:
                    b_avg = b_avg + self.bias_shared
            return F.linear(x, w_avg, b_avg)

        # === 核心逻辑: 权重融合 (Weight Merging) ===
        # 1. 混合路由专家权重
        # coefficients: [Batch, Num_Experts]
        # weight_experts: [Num_Experts, Out, In]
        # mixed_weights: [Batch, Out, In]
        mixed_weights = torch.einsum("be, eoi -> boi", mole_globals.coefficients, self.weight_experts)

        # 2. 【关键】融合共享专家权重
        # 利用分配律: (W_routed + W_shared) * x
        if self.use_shared_expert:
            mixed_weights = mixed_weights + self.weight_shared.unsqueeze(0)

        # 3. 处理 Bias
        mixed_bias = None
        if self.bias_experts is not None:
            mixed_bias = torch.einsum("be, eo -> bo", mole_globals.coefficients, self.bias_experts)
            if self.use_shared_expert and self.bias_shared is not None:
                mixed_bias = mixed_bias + self.bias_shared.unsqueeze(0)

        # 4. 执行线性变换
        # 根据系统大小拆分 Input，因为每个系统(Graph)对应一个混合后的权重
        x_split = torch.split(x, mole_globals.sizes.tolist(), dim=0)
        out_parts = []

        # 循环执行 (虽然是 Python 循环，但通常 System 数量不多，开销可控)
        # 如果追求极致性能，可以使用 torch.func.vmap 或 Grouped GEMM，但对于 Linear 维度通常可以直接 Loop
        for i, x_sys in enumerate(x_split):
            w = mixed_weights[i]
            b = mixed_bias[i] if mixed_bias is not None else None
            out_parts.append(F.linear(x_sys, w, b))

        return torch.cat(out_parts, dim=0)
# ------------------------------------------------------------------------------

class SO2_Attention(torch.nn.Module):
    def __init__(self, node_irreps, latent_dim: int, use_so2_att_proj: bool = True):
        super().__init__()
        self.irreps_in = node_irreps.simplify()
        self.l_max = max((l for (_, (l, _)), _ in zip(self.irreps_in, self.irreps_in.slices()) if l > 0), default=0)
        self.dims = {l: 2 * l + 1 for l in range(self.l_max + 1)}
        self.offsets = {}
        offset = 0
        for l in range(self.l_max + 1):
            self.offsets[l] = offset
            offset += self.dims[l]

        self.lin_center = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True, biases=True)
        self.lin_neighbor = e3nn_Linear(node_irreps, node_irreps, shared_weights=True, internal_weights=True,
                                        biases=True)

        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
        self.groups = groups

        # --- 修改：为每个 l 建立输入维为 (total_mul * (2l+1)) 的线性映射
        self.sim_linears = nn.ModuleDict()
        for l, g in groups.items():
            total_mul = sum(m for m, _ in g)
            in_dim = total_mul * self.dims[l]  # m * d
            self.sim_linears[f"l{l}"] = nn.Sequential(
                nn.Linear(in_dim, latent_dim),
                nn.SiLU(),
            )

        # --- 修改：用一个 final_mlp 替代简单求和（把所有 l 的 latent_dim 串联后再做一次融合）
        num_l = len(groups)
        # final_mlp: (num_l * latent_dim) -> latent_dim
        self.final_mlp = nn.Sequential(
            nn.Linear(num_l * latent_dim, 2 * latent_dim),
            nn.SiLU(),  # 平滑非线性（比 ReLU 更稳定）
            nn.Linear(2 * latent_dim, latent_dim)
        )

    def forward(self, node_features, active_edge_vector, active_edge_index, wigner_D_all=None):
        n, _ = node_features.shape
        # keep node features as-is (no per-edge rotation here)
        rot_n_feat_ = node_features.new_zeros(node_features.shape)

        if wigner_D_all is None and self.l_max > 0:
            angle = xyz_to_angles(active_edge_vector[:, [1, 2, 0]])
            wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)

        # keep scalar parts unchanged
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            if l == 0:
                rot_n_feat_[:, slice_info] = node_features[:, slice_info]

        # keep the raw (unrotated) node parts in rot_n_feat_ so linear layers can be applied
        for l, group in self.groups.items():
            if l == 0 or not group:
                continue
            for mul, sl in group:
                rot_n_feat_[:, sl] = node_features[:, sl]

        # apply linear maps (these are node-wise)
        rot_center_node_feat = self.lin_center(rot_n_feat_)
        rot_center_node_feat = rot_center_node_feat[active_edge_index[0]]  # shape: (n_edges, dim)

        rot_neighbor_node_feat = self.lin_neighbor(rot_n_feat_)
        rot_neighbor_node_feat = rot_neighbor_node_feat[active_edge_index[1]]  # shape: (n_edges, dim)

        latent_list = []
        # Now for each l, build per-edge (n_edges, total_mul, 2l+1) and apply per-edge rotation
        for l, group in self.groups.items():
            muls, slices = zip(*group)
            total_mul = sum(m for m, _ in group)
            # center/neighbor parts now have batch = n_edges
            # each part reshape -> (n_edges, mul, 2l+1)
            center_node_parts = [rot_center_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            center_node_combined = torch.cat(center_node_parts, dim=1)  # (n_edges, total_mul, d)

            neighbor_node_parts = [rot_neighbor_node_feat[:, sl].reshape(-1, mul, self.dims[l]) for mul, sl in group]
            neighbor_node_combined = torch.cat(neighbor_node_parts, dim=1)  # (n_edges, total_mul, d)

            if l == 0:
                # l=0: dims[l] == 1, no rotation needed; keep consistent flow
                # center_node_combined, neighbor_node_combined have shape (e, total_mul, 1)
                center_rot = center_node_combined
                neighbor_rot = neighbor_node_combined
            else:
                # get per-edge rotation matrices: shape (n_edges, 2l+1, 2l+1)
                start = self.offsets[l]
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]

                # rotate center & neighbor per-edge:
                # center_combined: (e, m, d), rot_mat: (e, d, d) -> rotated_center: (e, m, d)
                center_rot = torch.einsum('emd,edq->emq', center_node_combined, rot_mat)
                neighbor_rot = torch.einsum('emd,edq->emq', neighbor_node_combined, rot_mat)

            # --- 修改：不对 d 求和，而是保留 (e, m, d)，做 elementwise 相乘后展平为 (e, m*d)
            # elementwise product as similarity per-component
            sim_tensor = center_rot * neighbor_rot  # (e, m, d)
            e = sim_tensor.shape[0]
            sim_flat = sim_tensor.reshape(e, -1)  # (e, m * d)

            # map flattened similarity (m*d) to latent_dim
            sim_mapped = self.sim_linears[f"l{l}"](sim_flat)  # (e, latent_dim)
            latent_list.append(sim_mapped)

        # latent_list: list of (e, latent_dim), one per l
        # stack along new l-dim -> (e, num_l, latent_dim)
        latent_stack = torch.stack(latent_list, dim=1)
        # flatten (e, num_l * latent_dim) and fuse via final_mlp
        e = latent_stack.shape[0]
        fused = latent_stack.reshape(e, -1)
        latent = self.final_mlp(fused)  # (e, latent_dim)

        return latent


class SO2_Linear(torch.nn.Module):
    """
    SO(2) Convolutional layer with MoE and Rotate Control.
    """

    def __init__(
            self,
            irreps_in,
            irreps_out,
            radial_emb: bool = False,
            latent_dim: int = None,
            radial_channels: list = None,
            extra_m0_outsize: int = 0,
            use_interpolation: bool = False,
            # === MoE 参数 ===
            num_experts: int = 8,
            # === Rotation 控制参数 (Keep-in-Frame) ===
            rotate_in: bool = True,
            rotate_out: bool = True,
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + Irreps(irreps_out)).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim

        # 保存 flag
        self.rotate_in = rotate_in
        self.rotate_out = rotate_out
        self.num_experts = num_experts

        self.m_linear = nn.ModuleList()

        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        # MODIFICATION: Use MOLELinear for scalar projection (bias=True as per original)
        self.fc_m0 = MOLELinear(num_in_m0, num_out_m0, num_experts=num_experts, bias=True)

        for m in range(1, self.irreps_out.lmax + 1):
            # 假设 SO2_m_Linear 已经支持 num_experts 参数
            self.m_linear.append(SO2_m_Linear(
                m,
                self.irreps_in,
                self.irreps_out,
                use_interpolation=use_interpolation,
                num_experts=num_experts
            ))

        # --- Mask 和 Index 构建逻辑 (保持不变) ---
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

    def forward(self, x, R, mole_globals: MOLEGlobals, latents=None, wigner_D_all=None):
        """
        Args:
            x: Input features
            R: Edge vectors (for rotation)
            mole_globals: MoE routing info
            latents: Latent features for radial embedding
            wigner_D_all: Precomputed Wigner D matrices (optional)
        """
        n, _ = x.shape
        if self.radial_emb:
            weights = self.radial_emb(latents)
        x_ = torch.zeros_like(x)

        # === 1. 旋转矩阵准备 (Rotate Control) ===
        if wigner_D_all is None:
            # 只有当需要 rotate_in 或者 rotate_out 时才必须计算 D
            if (self.rotate_in or self.rotate_out) and self.l_max > 0:
                angle = xyz_to_angles(R[:, [1, 2, 0]])
                wigner_D_all = batch_wigner_D(self.l_max, angle[0], angle[1], torch.zeros_like(angle[0]), _Jd)

        # === 2. Rotate In (Global -> Local) ===
        groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_in, self.irreps_in.slices()):
            groups[l].append((mul, slice_info))
            if l == 0:
                x_[:, slice_info] = x[:, slice_info]

        for l, group in groups.items():
            if l == 0 or not group:
                continue
            muls, slices = zip(*group)

            # --- Flag Check: 如果 rotate_in 为 False，直接复制不旋转 ---
            if not self.rotate_in:
                for mul, sl in group:
                    x_[:, sl] = x[:, sl]
                continue
            # ----------------------------------------------------

            x_parts = [x[:, sl].reshape(n, mul, 2 * l + 1) for mul, sl in group]
            x_combined = torch.cat(x_parts, dim=1)
            start = self.offsets[l]
            rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
            transformed = torch.bmm(x_combined, rot_mat)
            for part, slice_info, mul in zip(transformed.split(muls, dim=1), slices, muls):
                x_[:, slice_info] = part.reshape(n, -1)

        # === 3. Convolution (Linear / MoE) ===
        out = torch.zeros(n, self.irreps_out.dim, dtype=x.dtype, device=x.device)
        for m in range(self.irreps_out.lmax + 1):
            radial_weight = weights[:, self.m_in_index[m]:self.m_in_index[m + 1]].unsqueeze(
                1) if self.radial_emb else 1.

            if m == 0:
                # MoE Logic for m=0
                inp = x_[:, self.m_in_mask[m]]
                if self.front and self.radial_emb:
                    # mole_globals passed here
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp * radial_weight.squeeze(1), mole_globals)
                elif self.radial_emb:
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp, mole_globals) * radial_weight.squeeze(1)
                else:
                    out[:, self.m_out_mask[m]] += self.fc_m0(inp, mole_globals)
            else:
                # MoE Logic for m>0
                x_m_in = x_[:, self.m_in_mask[m]].reshape(n, -1, 2).transpose(1, 2).contiguous()

                if self.front and self.radial_emb:
                    x_m_in.mul_(radial_weight)
                    # mole_globals passed here
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                elif self.radial_emb:
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)
                    linear_output.mul_(radial_weight)
                else:
                    linear_output = self.m_linear[m - 1](x_m_in, mole_globals)

                final_addition = linear_output.transpose(1, 2).contiguous().reshape(n, -1)
                out[:, self.m_out_mask[m]] += final_addition

        # === 4. Rotate Out (Local -> Global) ===
        # --- Flag Check: 如果 rotate_out 为 False，直接返回 ---
        if not self.rotate_out:
            return out.contiguous(), wigner_D_all
        # --------------------------------------------------

        for (mul, (l, p)), slice_in in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                start = self.offsets[l]
                rot_mat = wigner_D_all[:, start:start + self.dims[l], start:start + self.dims[l]]
                x_slice = out[:, slice_in].reshape(n, mul, -1)
                rotated = torch.einsum('nij,nmj->nmi', rot_mat, x_slice)
                out[:, slice_in] = rotated.reshape(n, -1)

        return out.contiguous(), wigner_D_all


class SO2_m_Linear(torch.nn.Module):
    """
    SO(2) Convolution for a specific order m > 0.
    """

    def __init__(
            self,
            m,
            irreps_in,
            irreps_out,
            use_interpolation: bool = False,
            num_experts: int = 8,  # Added
    ):
        super(SO2_m_Linear, self).__init__()
        self.m = m
        self.num_in_channel = sum(mul for mul, (l, p) in irreps_in if l >= m)
        self.num_out_channel = sum(mul for mul, (l, p) in irreps_out if l >= m)

        # MODIFICATION: MOLE Logic with bias=False (original was bias=False)
        if use_interpolation:
            self.fc = InterpolationBlock(self.num_in_channel, 2 * self.num_out_channel, bias=False)
            self.is_mole = False
        else:
            self.fc = MOLELinear(self.num_in_channel, 2 * self.num_out_channel, num_experts=num_experts, bias=False)
            with torch.no_grad():
                self.fc.weight_experts.data.mul_(1 / math.sqrt(2))
            self.is_mole = True

    def forward(self, x_m, mole_globals: MOLEGlobals):  # Added mole_globals
        # x_m ~ [N, 2, n_channels]
        if self.is_mole:
            x_m = self.fc(x_m, mole_globals)
        else:
            x_m = self.fc(x_m)

        x_r = x_m.narrow(2, 0, self.num_out_channel)
        x_i = x_m.narrow(2, self.num_out_channel, self.num_out_channel)
        x_m_r = x_r.narrow(1, 0, 1) - x_i.narrow(1, 1, 1)
        x_m_i = x_r.narrow(1, 1, 1) + x_i.narrow(1, 0, 1)
        return torch.cat((x_m_r, x_m_i), dim=1)

