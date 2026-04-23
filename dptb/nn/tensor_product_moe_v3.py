
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

_WIGNER_STATIC_CACHE = {}


def build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, d_total: int):
    """
    angle_stack: (3*N, )    # Input with alpha, beta, gamma stacked together
    l_max: int

    Returns: (Xa, Xb, Xc) # Each is of shape (N, D_total, D_total)
    """
    N_all = angle_stack.shape[0]
    N = N_all // 3

    # Step 1: Vectorized computation of sine and cosine values
    angle_expand = angle_stack[None, :, None]  # (1, 3N, 1)
    freq_expand = freq[:, None, :]  # (L, 1, Mmax)
    sin_val = torch.sin(freq_expand * angle_expand)  # (L, 3N, Mmax)
    cos_val = torch.cos(freq_expand * angle_expand)  # (L, 3N, Mmax)

    # Step 2: Construct the block-diagonal matrix
    M_total = angle_stack.new_zeros((N_all, d_total, d_total))
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


def _get_wigner_static(l_max: int, device: torch.device, dtype: torch.dtype):
    key = (int(l_max), str(device), dtype)
    cached = _WIGNER_STATIC_CACHE.get(key)
    if cached is not None:
        return cached

    idx_data = {
        k: (v.to(device=device) if isinstance(v, torch.Tensor) else v)
        for k, v in _idx_data.items()
    }
    sizes = idx_data["sizes"][:l_max + 1]
    offsets = idx_data["offsets"][:l_max + 1]
    mask = idx_data["mask"][:l_max + 1]
    freq = idx_data["freq"][:l_max + 1]
    reversed_inds = idx_data["reversed_inds"][:l_max + 1]

    dims = [2 * l + 1 for l in range(l_max + 1)]
    d_total = sum(dims)
    J_full_small = torch.zeros(d_total, d_total, dtype=dtype, device=device)
    for l, dim in enumerate(dims):
        start = l * l
        J_full_small[start:start + dim, start:start + dim] = _Jd[l].to(dtype=dtype, device=device)

    cached = {
        "sizes": sizes,
        "offsets": offsets,
        "mask": mask,
        "freq": freq,
        "reversed_inds": reversed_inds,
        "J_full_small": J_full_small,
        "d_total": d_total,
    }
    _WIGNER_STATIC_CACHE[key] = cached
    return cached


def batch_wigner_D(l_max, alpha, beta, gamma, _Jd):
    """
    Compute Wigner D matrices for all L (from 0 to l_max) in a single batch.
    Returns a tensor of shape [N, D, D], where D = sum(2l+1 for l in 0..l_max).
    """
    device = alpha.device
    N = alpha.shape[0]
    static = _get_wigner_static(l_max, device, alpha.dtype)
    d_total = static["d_total"]

    offsets = static["offsets"]
    mask = static["mask"]
    freq = static["freq"]
    reversed_inds = static["reversed_inds"]
    J_full_small = static["J_full_small"]

    J_full = J_full_small.unsqueeze(0).expand(N, -1, -1)
    angle_stack = torch.cat([alpha, beta, gamma], dim=0)
    Xa, Xb, Xc = build_z_rot_multi(angle_stack, mask, freq, reversed_inds, offsets, d_total)

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


class SO2WignerBlocks:
    """Per-l Wigner rotation blocks without materializing the full [N, D, D] matrix."""

    __slots__ = ("blocks",)

    def __init__(self, blocks):
        self.blocks = tuple(blocks)

    def block(self, l: int):
        return self.blocks[l]


def batch_wigner_D_blocks(l_max, alpha, beta, gamma, _Jd):
    """Compute Wigner D as compact per-l blocks instead of a dense block-diagonal matrix."""
    return SO2WignerBlocks(wigner_D(l, alpha, beta, gamma) for l in range(l_max + 1))


def _normalize_wigner_apply_mode(wigner_apply_mode: str) -> str:
    if wigner_apply_mode not in ("full_dense", "compact_blocks"):
        raise ValueError(
            "wigner_apply_mode must be 'full_dense' or 'compact_blocks', "
            f"got {wigner_apply_mode!r}"
        )
    return wigner_apply_mode


def _make_wigner_rotation(l_max, alpha, beta, gamma, wigner_apply_mode: str):
    if wigner_apply_mode == "compact_blocks":
        return batch_wigner_D_blocks(l_max, alpha, beta, gamma, _Jd)
    return batch_wigner_D(l_max, alpha, beta, gamma, _Jd)


def _select_wigner_block(wigner_D_all, l: int, offsets, dims):
    if isinstance(wigner_D_all, SO2WignerBlocks):
        return wigner_D_all.block(l)
    start = offsets[l]
    dim = dims[l]
    return wigner_D_all[:, start:start + dim, start:start + dim]


# ------------------------------------------------------------------------------
# MOLE COMPONENTS (Added)
# ------------------------------------------------------------------------------

class MOLEGlobals:
    """Stores routing information for the current forward pass."""

    def __init__(self, coefficients=None, sizes=None, split_sizes=None):
        self.coefficients = coefficients  # [Batch, Num_Experts]
        self.sizes = sizes  # [Batch] (Edge counts per system)
        self.split_sizes = self._normalize_split_sizes(sizes, split_sizes)

    @staticmethod
    def _normalize_split_sizes(sizes, split_sizes):
        if split_sizes is not None:
            if torch.is_tensor(split_sizes):
                return MOLEGlobals._tensor_to_split_tuple(split_sizes)
            return tuple(int(v) for v in split_sizes)
        if sizes is None:
            return None
        if torch.is_tensor(sizes):
            return MOLEGlobals._tensor_to_split_tuple(sizes)
        return tuple(int(v) for v in sizes)

    @staticmethod
    def _tensor_to_split_tuple(values):
        values = values.detach().reshape(-1)
        if values.device.type != "cpu":
            # Compatibility fallback for direct callers that still pass CUDA sizes.
            values = values.cpu()
        return tuple(int(v) for v in values.tolist())


def _mole_split_sizes(mole_globals, n_rows: int):
    split_sizes = getattr(mole_globals, "split_sizes", None)
    if split_sizes is None:
        split_sizes = (n_rows,)
    if sum(split_sizes) != n_rows:
        raise ValueError(
            f"MOLE split sizes sum to {sum(split_sizes)}, but input has {n_rows} rows."
        )
    return split_sizes


def _mole_graph_index(mole_globals, n_rows: int, *, device):
    """Return sorted graph ids per row, matching the existing split-loop semantics."""
    split_sizes = _mole_split_sizes(mole_globals, n_rows)
    cache = getattr(mole_globals, "_graph_index_cache", None)
    if cache is None:
        cache = {}
        setattr(mole_globals, "_graph_index_cache", cache)

    key = (str(device), split_sizes)
    graph_index = cache.get(key)
    if graph_index is None:
        sizes = torch.tensor(split_sizes, dtype=torch.long, device=device)
        # cuEquivariance indexed_linear requires sorted indices; the split_sizes
        # contract means rows are graph-contiguous, matching the old split loop.
        graph_index = torch.repeat_interleave(
            torch.arange(len(split_sizes), dtype=torch.long, device=device), sizes
        )
        cache[key] = graph_index
    return graph_index


def _expand_graph_index_for_leading_dims(graph_index: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Expand [E] graph ids to match x.reshape(-1, in_features)."""
    if x.ndim == 2:
        return graph_index

    expand_shape = [graph_index.shape[0]] + list(x.shape[1:-1])
    return graph_index.reshape(-1, *([1] * (x.ndim - 2))).expand(expand_shape).reshape(-1)


def _normalize_mole_linear_mode(mode: str) -> str:
    allowed = {"split_loop", "indexed_ref", "cueq_indexed_linear"}
    if mode not in allowed:
        raise ValueError(f"mole_linear_mode must be one of {sorted(allowed)}, got {mode!r}")
    return mode


class MOLERouterV3(nn.Module):
    def __init__(self, in_features, num_experts=48, top_k=6,
                 aux_loss_free=True,
                 bias_update_speed=0.005):  # 修改1: 固定 Bias 更新速度，不再衰减
        super().__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss_free = aux_loss_free

        # 固定的惩罚力度
        self.bias_update_speed = bias_update_speed

        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.SiLU(),
            nn.Linear(128, num_experts)
        )

        self.register_buffer('expert_bias', torch.zeros(num_experts))
        self.register_buffer('ema_load', torch.ones(num_experts) * (top_k / num_experts))

        # 修改1: 删除了 step_count 等用于衰减的 Buffer

    def forward(self, global_features, sizes=None):
        # 修改1: 删除了 Jitter (探索噪声) 的注入逻辑，完全依赖网络的自然 Logits
        logits = self.net(global_features)
        scores = torch.sigmoid(logits)

        # 加上 Bias 用于选择 Top-K (Aux-loss-free 核心机制)
        if self.aux_loss_free and self.training:
            scores_for_selection = scores + self.expert_bias
        else:
            scores_for_selection = scores

        if self.top_k is not None:
            topk_scores_biased, topk_indices = torch.topk(scores_for_selection, k=self.top_k, dim=-1)

            with torch.no_grad():
                mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()

                # 计算负载 (保留了 V1 支持 sizes 的优秀特性)
                if sizes is not None:
                    weight = sizes.view(-1, 1, 1)
                    weighted_mask = mask * weight
                    current_load = weighted_mask.sum(dim=(0, 1))
                    target_load = (sizes.sum() * self.top_k) / self.num_experts
                else:
                    current_load = mask.sum(dim=(0, 1))
                    target_load = (scores.size(0) * self.top_k) / self.num_experts

                # 使用 EMA 平滑历史负载统计，使返回的 CV 指标极其稳定
                if self.training:
                    self.ema_load.mul_(0.9).add_(current_load, alpha=0.1)
                expert_load_cv = self.ema_load.std() / (self.ema_load.mean() + 1e-8)

            # 修改1: 使用恒定力度 (0.005) 更新 Bias，持续进行负载均衡
            if self.aux_loss_free and self.training and self.bias_update_speed > 0.0:
                with torch.no_grad():
                    error = current_load - target_load
                    self.expert_bias -= torch.sign(error) * self.bias_update_speed
                    # 保持 Bias 整体均值为 0，防止激活值整体漂移
                    self.expert_bias -= self.expert_bias.mean()

            # 修改2: 强制 L1 归一化 (防止路由专家被共享专家 "饿死")
            topk_scores_original = torch.gather(scores, 1, topk_indices)
            denominators = topk_scores_original.sum(dim=-1, keepdim=True) + 1e-8
            topk_probs = topk_scores_original / denominators

            # 构建稀疏输出系数
            coeffs = torch.zeros_like(scores)
            coeffs.scatter_(1, topk_indices, topk_probs)

            # 监控指标：计算最大概率的均值 (反映 Router 的置信度，比计算全部均值更有意义)
            monitor_val = topk_probs.max(dim=-1)[0].mean().detach()

            return coeffs, monitor_val, expert_load_cv.detach()

        else:
            # Fallback 逻辑保持稳定
            denominators = scores.sum(dim=-1, keepdim=True) + 1e-8
            probs = scores / denominators
            monitor_val = probs.max(dim=-1)[0].mean().detach()
            return probs, monitor_val, torch.tensor(0.0, device=scores.device)


class MOLELinear(nn.Module):
    """
    DeepSeek-V3 Style Expert Layer with Linear Experts.

    Structure: Output = Routed_Experts(x) + Shared_Experts(x)
    Optimization: Merges Shared Weights into Routed Weights for 0 extra inference overhead.
    """

    def __init__(
            self,
            in_features,
            out_features,
            num_experts=8,
            num_shared_experts=1,
            bias=True,
            mole_linear_mode=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.mole_linear_mode = _normalize_mole_linear_mode(
            mole_linear_mode or os.environ.get("DPTB_MOLE_LINEAR_MODE", "split_loop")
        )
        self._cueq_indexed_linear_cache = {}
        self._cueq_weight_order = None

        # 1. 路由专家权重
        self.weight_experts = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        if bias:
            self.bias_experts = nn.Parameter(torch.empty(num_experts, out_features))
        else:
            self.register_parameter('bias_experts', None)

        # 2. 共享专家权重 (Shared Expert) 支持配置数量
        if self.num_shared_experts > 0:
            self.weight_shared = nn.Parameter(torch.empty(num_shared_experts, out_features, in_features))
            if bias:
                self.bias_shared = nn.Parameter(torch.empty(num_shared_experts, out_features))
            else:
                self.register_parameter('bias_shared', None)
        else:
            self.register_parameter('weight_shared', None)
            self.register_parameter('bias_shared', None)

        self.reset_parameters()

    def reset_parameters(self):
        k = math.sqrt(1.0 / self.in_features)
        nn.init.uniform_(self.weight_experts, -k, k)
        if self.bias_experts is not None:
            nn.init.uniform_(self.bias_experts, -k, k)

        if self.num_shared_experts > 0:
            nn.init.uniform_(self.weight_shared, -k, k)
            if self.bias_shared is not None:
                nn.init.uniform_(self.bias_shared, -k, k)

    def _apply_indexed_ref(self, x, mixed_weights, mixed_bias, graph_index):
        flat_x = x.reshape(-1, self.in_features)
        flat_graph_index = _expand_graph_index_for_leading_dims(graph_index, x)
        flat_w = mixed_weights.index_select(0, flat_graph_index)
        flat_out = torch.bmm(flat_w, flat_x.unsqueeze(-1)).squeeze(-1)
        if mixed_bias is not None:
            flat_out = flat_out + mixed_bias.index_select(0, flat_graph_index)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def _cueq_flatten_weight(self, mixed_weights, order: str):
        scale = math.sqrt(self.in_features)
        if order == "io_scaled":
            flat = mixed_weights.transpose(1, 2).contiguous() * scale
        elif order == "oi_scaled":
            flat = mixed_weights.contiguous() * scale
        elif order == "io":
            flat = mixed_weights.transpose(1, 2).contiguous()
        elif order == "oi":
            flat = mixed_weights.contiguous()
        else:
            raise ValueError(f"unknown cueq weight order {order!r}")
        return flat.reshape(mixed_weights.shape[0], -1)

    def _get_cueq_indexed_linear(self, num_graphs: int, *, dtype, device):
        if device.type != "cuda":
            raise RuntimeError("cueq_indexed_linear requires CUDA; use split_loop or indexed_ref on CPU.")
        if dtype not in (torch.float32, torch.float64):
            raise RuntimeError(
                "cueq_indexed_linear is currently validated only for float32/float64. "
                "Disable AMP/autocast for this experimental backend or use split_loop."
            )

        try:
            import cuequivariance as cue
            import cuequivariance_torch as cuet
        except ImportError as exc:
            raise ImportError(
                "mole_linear_mode='cueq_indexed_linear' requires cuequivariance and "
                "cuequivariance_torch."
            ) from exc

        key = (num_graphs, str(dtype), str(device), self.in_features, self.out_features)
        mod = self._cueq_indexed_linear_cache.get(key)
        if mod is None:
            irreps_in = cue.Irreps(cue.O3, f"{self.in_features}x0e")
            irreps_out = cue.Irreps(cue.O3, f"{self.out_features}x0e")
            mod = cuet.Linear(
                irreps_in,
                irreps_out,
                shared_weights=True,
                internal_weights=False,
                weight_classes=num_graphs,
                layout=cue.ir_mul,
                device=device,
                dtype=dtype,
                method="indexed_linear",
            )
            self._cueq_indexed_linear_cache[key] = mod
        return mod

    def _infer_cueq_weight_order(self, cue_lin, flat_x, mixed_weights, flat_graph_index):
        if self._cueq_weight_order is not None:
            return self._cueq_weight_order

        with torch.no_grad():
            n_probe = min(int(flat_x.shape[0]), 64)
            probe_x = flat_x[:n_probe]
            probe_idx = flat_graph_index[:n_probe]
            ref_w = mixed_weights.index_select(0, probe_idx)
            ref = torch.bmm(ref_w, probe_x.unsqueeze(-1)).squeeze(-1)

            best_order, best_err = None, None
            for order in ("io_scaled", "oi_scaled", "io", "oi"):
                try:
                    weight = self._cueq_flatten_weight(mixed_weights, order)
                    out = cue_lin(probe_x, weight=weight, weight_indices=probe_idx)
                    err_val = float((out - ref).abs().max().detach().cpu())
                except Exception:
                    continue
                if best_err is None or err_val < best_err:
                    best_order, best_err = order, err_val

        if best_order is None or best_err is None or best_err > 1e-4:
            raise RuntimeError(
                "Could not infer cuEquivariance scalar Linear weight order; "
                f"best_order={best_order}, best_err={best_err}."
            )

        self._cueq_weight_order = best_order
        return best_order

    def _apply_cueq_indexed_linear(self, x, mixed_weights, mixed_bias, graph_index):
        flat_x = x.reshape(-1, self.in_features)
        flat_graph_index = _expand_graph_index_for_leading_dims(graph_index, x)
        num_graphs = int(mixed_weights.shape[0])
        cue_lin = self._get_cueq_indexed_linear(num_graphs, dtype=x.dtype, device=x.device)

        order = self._infer_cueq_weight_order(cue_lin, flat_x, mixed_weights, flat_graph_index)
        flat_weight = self._cueq_flatten_weight(mixed_weights, order)
        flat_out = cue_lin(flat_x, weight=flat_weight, weight_indices=flat_graph_index)
        if mixed_bias is not None:
            flat_out = flat_out + mixed_bias.index_select(0, flat_graph_index)
        return flat_out.reshape(*x.shape[:-1], self.out_features)

    def forward(self, x, mole_globals: MOLEGlobals):
        # 安全回退
        if mole_globals is None or mole_globals.coefficients is None:
            w_avg = self.weight_experts.mean(0)
            if self.num_shared_experts > 0:
                w_avg = w_avg + self.weight_shared.sum(0)
            b_avg = None
            if self.bias_experts is not None:
                b_avg = self.bias_experts.mean(0)
                if self.num_shared_experts > 0 and self.bias_shared is not None:
                    b_avg = b_avg + self.bias_shared.sum(0)
            return F.linear(x, w_avg, b_avg)

        # === 核心逻辑: 权重融合 (Weight Merging) ===
        # 1. 混合路由专家权重
        # coefficients: [Batch, Num_Experts]
        # weight_experts: [Num_Experts, Out, In]
        # mixed_weights: [Batch, Out, In]
        mixed_weights = torch.einsum("be, eoi -> boi", mole_globals.coefficients, self.weight_experts)

        # 2. 【关键】融合共享专家权重
        # 利用分配律: (W_routed + sum(W_shared)) * x
        if self.num_shared_experts > 0:
            mixed_weights = mixed_weights + self.weight_shared.sum(0).unsqueeze(0)

        # 3. 处理 Bias
        mixed_bias = None
        if self.bias_experts is not None:
            mixed_bias = torch.einsum("be, eo -> bo", mole_globals.coefficients, self.bias_experts)
            if self.num_shared_experts > 0 and self.bias_shared is not None:
                mixed_bias = mixed_bias + self.bias_shared.sum(0).unsqueeze(0)

        # 4. 执行线性变换
        # 根据系统大小拆分 Input，因为每个系统(Graph)对应一个混合后的权重
        mode = self.mole_linear_mode
        if mode != "split_loop":
            graph_index = _mole_graph_index(mole_globals, x.shape[0], device=x.device)
            if graph_index.numel() != x.shape[0]:
                raise ValueError(
                    f"MOLE graph_index has {graph_index.numel()} rows, but input has {x.shape[0]} rows."
                )
            if mode == "indexed_ref":
                return self._apply_indexed_ref(x, mixed_weights, mixed_bias, graph_index)
            if mode == "cueq_indexed_linear":
                return self._apply_cueq_indexed_linear(x, mixed_weights, mixed_bias, graph_index)
            raise AssertionError(f"unreachable mole_linear_mode={mode!r}")

        split_sizes = mole_globals.split_sizes
        if split_sizes is None:
            split_sizes = (x.shape[0],)
        if sum(split_sizes) != x.shape[0]:
            raise ValueError(
                f"MOLE split sizes sum to {sum(split_sizes)}, but input has {x.shape[0]} rows."
            )

        x_split = torch.split(x, split_sizes, dim=0)
        out_parts = []

        # 循环执行 (虽然是 Python 循环，但通常 System 数量不多，开销可控)
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
            num_shared_experts: int = 1, # Added
            # === Rotation 控制参数 (Keep-in-Frame) ===
            rotate_in: bool = True,
            rotate_out: bool = True,
            wigner_apply_mode: str = "compact_blocks",
            mole_linear_mode=None,
    ):
        super(SO2_Linear, self).__init__()

        self.irreps_in = Irreps(irreps_in).simplify()
        self.irreps_out = (Irreps(f"{extra_m0_outsize}x0e") + Irreps(irreps_out)).simplify()
        self.radial_emb = radial_emb
        self.latent_dim = latent_dim

        # 保存 flag
        self.rotate_in = rotate_in
        self.rotate_out = rotate_out
        self.wigner_apply_mode = _normalize_wigner_apply_mode(wigner_apply_mode)
        self.num_experts = num_experts

        self.m_linear = nn.ModuleList()

        num_in_m0 = self.irreps_in.num_irreps
        num_out_m0 = self.irreps_out.num_irreps

        # MODIFICATION: Use MOLELinear for scalar projection (bias=True as per original)
        self.fc_m0 = MOLELinear(
            num_in_m0,
            num_out_m0,
            num_experts=num_experts,
            num_shared_experts=num_shared_experts,
            bias=True,
            mole_linear_mode=mole_linear_mode,
        )

        for m in range(1, self.irreps_out.lmax + 1):
            # 假设 SO2_m_Linear 已经支持 num_experts 参数
            self.m_linear.append(SO2_m_Linear(
                m,
                self.irreps_in,
                self.irreps_out,
                use_interpolation=use_interpolation,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                mole_linear_mode=mole_linear_mode,
            ))

        # --- Mask 和 Index 构建逻辑 (保持不变) ---
        m_in_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_in.dim, dtype=torch.bool)
        m_out_mask = torch.zeros(self.irreps_in.lmax + 1, self.irreps_out.dim, dtype=torch.bool)
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
                m_in_mask[m, start_id + l + m] = True
                m_in_mask[m, start_id + l - m] = True
                if front:
                    self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        offset = 0
        for mul, (l, p) in self.irreps_out:
            start_id = offset + torch.LongTensor(list(range(mul))) * (2 * l + 1)
            for m in range(l + 1):
                if m <= self.irreps_in.lmax:
                    m_out_mask[m, start_id + l + m] = True
                    m_out_mask[m, start_id + l - m] = True
                    if not front:
                        self.m_in_num[m] += mul
            offset += mul * (2 * l + 1)
        self.register_buffer("m_in_mask", m_in_mask)
        self.register_buffer("m_out_mask", m_out_mask)
        self.m_in_index = [0] + [int(v) for v in torch.cumsum(torch.tensor(self.m_in_num), dim=0).tolist()]
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
                wigner_D_all = _make_wigner_rotation(
                    self.l_max,
                    angle[0],
                    angle[1],
                    torch.zeros_like(angle[0]),
                    self.wigner_apply_mode,
                )

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
            rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
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

        out_groups = defaultdict(list)
        for (mul, (l, p)), slice_info in zip(self.irreps_out, self.irreps_out.slices()):
            if l > 0:
                out_groups[l].append((mul, slice_info))

        for l, group in out_groups.items():
            muls, slices = zip(*group)
            out_parts = [out[:, sl].reshape(n, mul, self.dims[l]) for mul, sl in group]
            out_combined = torch.cat(out_parts, dim=1)
            rot_mat = _select_wigner_block(wigner_D_all, l, self.offsets, self.dims)
            rotated = torch.bmm(out_combined, rot_mat.transpose(1, 2))
            for part, slice_info, mul in zip(rotated.split(muls, dim=1), slices, muls):
                out[:, slice_info] = part.reshape(n, -1)

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
            num_shared_experts: int = 1, # Added
            mole_linear_mode=None,
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
            self.fc = MOLELinear(
                self.num_in_channel,
                2 * self.num_out_channel,
                num_experts=num_experts,
                num_shared_experts=num_shared_experts,
                bias=False,
                mole_linear_mode=mole_linear_mode,
            )
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
