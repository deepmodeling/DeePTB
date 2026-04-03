import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter_mean
from typing import Union

import torch
from torch import nn
from e3nn import o3
from typing import Union, Optional


class MySeperableLayerNorm(nn.Module):
    """
    Gated Bottleneck Norm (Unconditioned Version).
    Fuses:
    1. Absolute Scale Preservation (Multiplicative Gate, No Division).
    2. Correctness: Strict scalar channel mapping (Fixes interleaved irreps bug).
    3. Performance: Scatter-based Vectorized RMS computation.
    """

    def __init__(
            self,
            irreps,
            eps=1e-6,
            affine=True,
            normalization='component',  # 兼容老版本签名
            std_balance_degrees=True,  # 兼容老版本签名
            bottleneck_ratio=0.5,
            gate_norm: Optional[str] = 'none',  # 可选: 'none', 'rms', 'layer'
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super().__init__()
        print(f'bottleneck_ratio: {bottleneck_ratio}')
        self.irreps = o3.Irreps(irreps).simplify()
        self.device = torch.device(device) if isinstance(device, str) else device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        self.eps = eps
        self.affine = affine
        self.gate_norm = gate_norm.lower() if gate_norm else 'none'

        # ── 1. 构建切片索引 (Scatter 友好版 & 修复错位 Bug) ──
        scalar_idx = []
        scalar_ch = []  # ✨ 记录标量对应的真实通道 ID
        vector_idx = []  # 记录所有高阶特征的展平索引
        vector_ch_local = []  # 用于 scatter_mean 的局部 channel id (0, 1, 2...)
        ch_expand = []  # 门控输出广播到全特征的映射

        ch = 0  # 全局通道 ID (0 to num_features - 1)
        vec_ch = 0  # 向量专用局部 ID (0 to num_vector - 1)
        offset = 0  # 展平后的特征偏移量

        for mul, ir in self.irreps:
            for _ in range(mul):
                if ir.l == 0:
                    scalar_idx.append(offset)
                    scalar_ch.append(ch)  # 精确绑定标量的门控通道
                    ch_expand.append(ch)
                    offset += 1
                else:
                    for _ in range(ir.dim):
                        vector_idx.append(offset)
                        vector_ch_local.append(vec_ch)
                        ch_expand.append(ch)
                        offset += 1
                    vec_ch += 1
                ch += 1

        self.num_scalar = len(scalar_idx)
        self.num_vector = vec_ch
        self.num_features = ch
        self.total_dim = offset

        # 注册所有索引为 buffer
        self.register_buffer("scalar_idx", torch.tensor(scalar_idx, dtype=torch.long, device=self.device))
        self.register_buffer("scalar_ch", torch.tensor(scalar_ch, dtype=torch.long, device=self.device))
        self.register_buffer("ch_expand", torch.tensor(ch_expand, dtype=torch.long, device=self.device))

        if self.num_vector > 0:
            self.register_buffer("vector_idx", torch.tensor(vector_idx, dtype=torch.long, device=self.device))
            self.register_buffer("vector_ch_local", torch.tensor(vector_ch_local, dtype=torch.long, device=self.device))

        # ── 2. 门控 MLP ──
        gate_in_dim = self.num_scalar + self.num_vector
        bn_dim = max(int(gate_in_dim * bottleneck_ratio), 4)

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_in_dim, bn_dim, dtype=self.dtype, device=self.device),
            nn.SiLU(),
            nn.Linear(bn_dim, self.num_features, dtype=self.dtype, device=self.device),
        )

        # ✨ 保证初始阶段乘积系数为 1.0 (残差无损流转)
        nn.init.zeros_(self.gate_mlp[-1].weight)
        nn.init.zeros_(self.gate_mlp[-1].bias)

        # ── 3. Affine 参数 ──
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.num_features, dtype=self.dtype, device=self.device))
            if self.num_scalar > 0:
                self.affine_bias = nn.Parameter(torch.zeros(1, self.num_scalar, dtype=self.dtype, device=self.device))
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        batch_size = x.size(0)

        # ---------------------------------------------------------
        # 1. 标量特征 (去均值保持相对符号)
        # ---------------------------------------------------------
        if self.num_scalar > 0:
            scalars = x[:, self.scalar_idx]
            scalars_centered = scalars - scalars.mean(dim=1, keepdim=True)
        else:
            scalars = x.new_zeros((batch_size, 0))
            scalars_centered = scalars

        # ---------------------------------------------------------
        # 2. 向量 RMS (GPU 高效并行版 Scatter_Mean)
        # ---------------------------------------------------------
        if self.num_vector > 0:
            vec_sq = x[:, self.vector_idx].pow(2)
            # scatter_mean 一次性聚合算出所有向量的均方
            local_idx_expanded = self.vector_ch_local.unsqueeze(0).expand(batch_size, -1)
            vec_mean_sq = scatter_mean(vec_sq, local_idx_expanded, dim=1, dim_size=self.num_vector)

            # 必须开根号，获得线性尺度的 RMS，防止特征平方引发极大值爆炸
            vectors_rms = (vec_mean_sq + self.eps).sqrt()
        else:
            vectors_rms = x.new_zeros((batch_size, 0))

        # ---------------------------------------------------------
        # 3. 门控信号拼装与规范化 (默认 'none'，如果抖动可以改成 'rms')
        # ---------------------------------------------------------
        gate_in = torch.cat([scalars_centered, vectors_rms], dim=1)

        if self.gate_norm == 'rms':
            rms_scale = torch.rsqrt(gate_in.pow(2).mean(dim=1, keepdim=True) + self.eps)
            gate_in = gate_in * rms_scale
        elif self.gate_norm == 'layer':
            gate_in = torch.nn.functional.layer_norm(gate_in, (gate_in.shape[-1],))

        # ---------------------------------------------------------
        # 4. 计算门控分数 (2 * Sigmoid 恒等初始化)
        # ---------------------------------------------------------
        gate_scores = 2.0 * torch.sigmoid(self.gate_mlp(gate_in))

        if self.affine:
            gate_scores = gate_scores * self.affine_weight

        # ---------------------------------------------------------
        # 5. 主特征流缩放 (绝不除以模长！)
        # ---------------------------------------------------------
        x_out = x * gate_scores[:, self.ch_expand]

        # ---------------------------------------------------------
        # 6. 恢复标量偏置 (使用 scalar_ch 进行精准通道映射)
        # ---------------------------------------------------------
        if self.num_scalar > 0:
            # 必须用精确通道索引，无惧 Irreps 乱序交错
            x_out[:, self.scalar_idx] = scalars_centered * gate_scores[:, self.scalar_ch]
            if self.affine:
                x_out[:, self.scalar_idx] = x_out[:, self.scalar_idx] + self.affine_bias

        return x_out


class SeperableLayerNorm(nn.Module):
    '''
        1. Normalize over L = 0.
        2. Normalize across all m components from degrees L > 0.
        3. Do not normalize separately for different L (L > 0).
    '''

    def __init__(
            self,
            irreps,
            eps=1e-5,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu")
    ):
        super().__init__()
        if isinstance(irreps, o3.Irreps):
            self.irreps = irreps.simplify()
        else:
            self.irreps = o3.Irreps(irreps).simplify()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        if isinstance(device, str):
            device = torch.device(device)

        self.eps = eps
        self.lmax = self.irreps.lmax
        self.affine = affine
        self.num_scalar = 0
        self.std_balance_degrees = std_balance_degrees
        self.device = device
        self.dtype = dtype

        self.num_features = self.irreps.num_irreps

        count_scales = 0
        count_shift = 0
        self.shift_index = []
        self.scale_index = []
        self.scalar_weight_index = []
        for mul, ir in self.irreps:
            if str(ir) == "0e":
                self.num_scalar += mul
                self.shift_index += list(range(count_shift, count_shift + mul))
                count_shift += mul
            else:
                self.shift_index += [-1] * mul * ir.dim

            for _ in range(mul):
                if str(ir) == "0e":
                    self.scalar_weight_index += [count_scales]
                self.scale_index += [count_scales] * ir.dim
                count_scales += 1

        self.shift_index = torch.as_tensor(self.shift_index, dtype=torch.int64, device=self.device)
        self.scale_index = torch.as_tensor(self.scale_index, dtype=torch.int64, device=self.device)
        self.scalar_weight_index = torch.as_tensor(self.scalar_weight_index, dtype=torch.int64, device=self.device)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(1, self.irreps.num_irreps))
            self.affine_bias = nn.Parameter(torch.zeros(1, self.num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros(self.irreps.num_irreps)
            count = 0
            for mul, ir in self.irreps:
                if ir.l == 0:
                    balance_degree_weight[count:count + mul] = self.lmax  # to make the devided value to 1.0
                else:
                    balance_degree_weight[count:count + mul] = (1.0 / ir.dim / mul)
                count += mul
            balance_degree_weight = balance_degree_weight / sum([1 for mul, ir in self.irreps if ir.l > 0])
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(irreps={self.irreps}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, x):
        '''
            Assume input is of shape [N, sphere_basis, C] or [N, dim]
        '''
        batch, dim = x.shape
        x = x.reshape(batch, dim)  # [batch, stacked features]

        # 1. 0e 特征去均值
        # 注意：这里我们使用 mask 提取，计算 mean，然后再减去
        # 为了避免 inplace 修改原始输入，我们先 clone 一个 x，或者显式构建新变量
        # 这里的 x = x + 0. 操作虽然创建了副本，但后续的切片赋值 x[:] = ... 依然是 inplace
        # 但在这个阶段（计算 norm 之前），inplace 修改 x 是允许的，因为它是新的变量

        x = x + 0.  # Create a copy

        # 提取标量部分
        scalar_mask = self.shift_index.ge(0)
        if scalar_mask.any():
            # 计算均值
            feature_mean = x[:, scalar_mask].mean(dim=1, keepdim=True)
            # 减去均值 (Inplace on x is okay here because x is not used for gradient before this)
            x[:, scalar_mask] = x[:, scalar_mask] - feature_mean

        # 2. 计算 Norm (基于去均值后的 x)
        # compute norm of x0
        if scalar_mask.any():
            scalar_norm = 1.0 / (x[:, scalar_mask].norm(dim=1, keepdim=True) + self.eps)  # [N, 1]
        else:
            scalar_norm = None

        # 3. compute the norm across all irreps except for 0e
        vector_mask = self.shift_index.lt(0)
        feature_norm = None

        if self.lmax > 0 and vector_mask.any():
            if self.normalization == 'norm':
                feature_norm = x[:, vector_mask].pow(2).sum(1, keepdim=True)  # [N, 1]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    # 全局加权
                    feature_sq = x.pow(2) * self.balance_degree_weight[self.scale_index]
                    feature_norm = feature_sq[:, vector_mask].sum(1, keepdim=True)  # [N, 1]
                else:
                    feature_norm = x[:, vector_mask].pow(2).sum(dim=1, keepdim=True)  # [N, 1]

            feature_norm = (feature_norm + self.eps).pow(-0.5)

        # 4. 应用 Scaling
        if self.affine:
            # 这里的 x 已经被修改过（减去均值），可以直接乘
            # 构建完整的 weight 向量
            # weight shape: [N, num_features]

            # 由于 vector 和 scalar 混合，我们需要正确的索引
            # feature_norm [N, 1], scalar_norm [N, 1]

            # 使用 self.scale_index 映射 [0, 0, 1, 1, 1, ...]
            # scalar 指向的 index 对应 scalar_norm，vector 对应 feature_norm

            # 组合 norms 到一个 tensor [N, num_irreps_groups]
            # 这是一个稍微复杂点的地方，原代码通过 weight[:, self.scale_index] 巧妙解决了

            # 还原原代码逻辑，这是安全的
            # 先给 feature_norm 广播
            weight_base = feature_norm if feature_norm is not None else torch.ones_like(scalar_norm)

            # 乘以可学习参数
            weight = self.affine_weight * weight_base  # [N, num_irreps]

            # 修正 Scalar 部分的 norm (替换为 scalar_norm)
            if scalar_norm is not None:
                weight[:, self.scalar_weight_index] = self.affine_weight[:, self.scalar_weight_index] * scalar_norm

            # 广播到所有维度 [N, dim]
            x = x * weight[:, self.scale_index]

            # 加 bias
            if scalar_mask.any():
                x[:, scalar_mask] = x[:, scalar_mask] + self.affine_bias

        else:
            # === [Fix for affine=False] ===
            # 这里不能做 x[:] = ... 因为 x 已经被用来计算 norm 了

            # 构建一个全 1 的缩放矩阵
            scale_map = torch.ones_like(x)

            # 填入 scalar norm
            if scalar_norm is not None:
                # expand 到 scalar 的列数
                scale_map[:, scalar_mask] = scalar_norm.expand(-1, scalar_mask.sum())

            # 填入 vector norm
            if feature_norm is not None:
                # expand 到 vector 的列数
                scale_map[:, vector_mask] = feature_norm.expand(-1, vector_mask.sum())

            # 最后做一次非原地的乘法
            x = x * scale_map

        return x


@compile_mode("unsupported")
class TypeNorm(nn.Module):
    """Batch normalization for orthonormal representations

    It normalizes by the norm of the representations.
    Note that the norm is invariant only for orthonormal representations.
    Irreducible representations `wigner_D` are orthonormal.

    Parameters
    ----------
    irreps : `o3.Irreps`
        representation

    eps : float
        avoid division by zero when we normalize by the variance

    momentum : float
        momentum of the running average

    affine : bool
        do we have weight and bias parameters

    reduce : {'mean', 'max'}
        method used to reduce

    """

    def __init__(self, irreps, eps=1e-5, momentum=0.1, affine=True, num_type=1, reduce="mean", normalization="component"):
        super().__init__()

        self.irreps = o3.Irreps(irreps)
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.num_type = num_type

        num_scalar = sum(mul for mul, ir in self.irreps if ir.is_scalar())
        num_features = self.irreps.num_irreps

        self.register_buffer("running_mean", torch.zeros(num_type, num_scalar))
        self.register_buffer("running_var", torch.ones(num_type, num_features))

        if affine:
            self.weight = nn.Parameter(torch.ones(num_type, num_features))
            self.bias = nn.Parameter(torch.zeros(num_type, num_scalar))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        assert isinstance(reduce, str), "reduce should be passed as a string value"
        assert reduce in ["mean", "max"], "reduce needs to be 'mean' or 'max'"
        self.reduce = reduce

        assert normalization in ["norm", "component"], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps}, momentum={self.momentum})"

    def _roll_avg(self, curr, update):
        mask = (update.norm(dim=-1) > 1e-7)
        out = curr.clone()
        out[mask] = (1 - self.momentum) * curr[mask] + self.momentum * update[mask].detach()
        return out


    def forward(self, input, input_type):
        """evaluate

        Parameters
        ----------
        input : `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        input_type : `torch.Tensor`
            tensor of shape ``(batch)``

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``(batch, ..., irreps.dim)``
        """
        
        batch, *size, dim = input.shape
        input = input.reshape(batch, -1, dim)  # [batch, sample, stacked features]

        if self.training:
            new_means = []
            new_vars = []

        fields = []
        ix = 0
        irm = 0
        irv = 0
        iw = 0
        ib = 0

        for mul, ir in self.irreps:
            d = ir.dim
            field = input[:, :, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d

            # [batch, sample, mul, repr]
            field = field.reshape(batch, -1, mul, d)

            if ir.is_scalar():  # scalars
                if self.training:
                    field_mean = field.mean(1).reshape(batch, mul)  # [batch, mul]
                    field_mean = scatter_mean(field_mean, input_type, dim=0, dim_size=self.num_type)  # [num_type, mul]
                    new_means.append(self._roll_avg(self.running_mean[:, irm : irm + mul], field_mean))
                else:
                    field_mean = self.running_mean[:, irm : irm + mul]
                irm += mul

                # [batch, sample, mul, repr]
                field = field - field_mean.reshape(-1, 1, mul, 1)[input_type]

            if self.training:
                if self.normalization == "norm":
                    field_norm = field.pow(2).sum(3)  # [batch, sample, mul]
                elif self.normalization == "component":
                    field_norm = field.pow(2).mean(3)  # [batch, sample, mul]
                else:
                    raise ValueError("Invalid normalization option {}".format(self.normalization))

                if self.reduce == "mean":
                    field_norm = field_norm.mean(1)  # [batch, mul]
                elif self.reduce == "max":
                    field_norm = field_norm.max(1).values  # [batch, mul]
                else:
                    raise ValueError("Invalid reduce option {}".format(self.reduce))

                field_norm = scatter_mean(field_norm, input_type, dim=0, dim_size=self.num_type)  # [num_type, mul]
                new_vars.append(self._roll_avg(self.running_var[:, irv : irv + mul], field_norm))
            else:
                field_norm = self.running_var[:, irv : irv + mul]
            irv += mul

            field_norm = (field_norm + self.eps).pow(-0.5)  # [(batch,) mul]

            if self.affine:
                weight = self.weight[:, iw : iw + mul]  # [mul]
                iw += mul

                field_norm = field_norm * weight  # [num_type, mul]

            field = field * field_norm.reshape(-1, 1, mul, 1)[input_type]  # [batch, sample, mul, repr]

            if self.affine and ir.is_scalar():  # scalars
                bias = self.bias[:, ib : ib + mul]  # [mul]
                ib += mul
                field += bias.reshape(-1, 1, mul, 1)[input_type]  # [batch, sample, mul, repr]

            fields.append(field.reshape(batch, -1, mul * d))  # [batch, sample, mul * repr]

        if ix != dim:
            fmt = "`ix` should have reached input.size(-1) ({}), but it ended at {}"
            msg = fmt.format(dim, ix)
            raise AssertionError(msg)

        if self.training:
            assert irm == self.running_mean.size(-1)
            assert irv == self.running_var.size(-1)
        if self.affine:
            assert iw == self.weight.size(-1)
            assert ib == self.bias.size(-1)

        if self.training:
            if len(new_means) > 0:
                torch.cat(new_means, dim=-1, out=self.running_mean)
            if len(new_vars) > 0:
                torch.cat(new_vars, dim=-1, out=self.running_var)

        output = torch.cat(fields, dim=2)  # [batch, sample, stacked features]
        return output.reshape(batch, *size, dim)
    
# The following code are copied from the EquiformerV2 package
# https://github.com/atomicarchitects/equiformer_v2/tree/main
    
    
'''
    1. Normalize features of shape (N, sphere_basis, C), 
    with sphere_basis = (lmax + 1) ** 2.
    
    2. The difference from `layer_norm.py` is that all type-L vectors have 
    the same number of channels and input features are of shape (N, sphere_basis, C).
'''

import torch
import torch.nn as nn


def get_normalization_layer(norm_type, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
    assert norm_type in ['layer_norm', 'layer_norm_sh', 'rms_norm_sh']
    if norm_type == 'layer_norm':
        norm_class = EquivariantLayerNormArray
    elif norm_type == 'layer_norm_sh':
        norm_class = EquivariantLayerNormArraySphericalHarmonics
    elif norm_type == 'rms_norm_sh':
        norm_class = EquivariantRMSNormArraySphericalHarmonicsV2
    else:
        raise ValueError
    return norm_class(lmax, num_channels, eps, affine, normalization)


def get_l_to_all_m_expand_index(lmax):
    expand_index = torch.zeros([(lmax + 1) ** 2]).long()
    for l in range(lmax + 1):
        start_idx = l ** 2
        length = 2 * l + 1
        expand_index[start_idx : (start_idx + length)] = l
    return expand_index


class EquivariantLayerNormArray(nn.Module):
    
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(lmax + 1, num_channels))
            self.affine_bias   = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.amp.autocast(device_type="cuda",enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        
        out = []
        
        for l in range(self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            
            feature = node_input.narrow(1, start_idx, length)
            
            # For scalars, first compute and subtract the mean
            if l == 0:
                feature_mean = torch.mean(feature, dim=2, keepdim=True)
                feature = feature - feature_mean
                
            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)
            
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)     # [1, C]
                weight = weight.view(1, 1, -1)                  # [1, 1, C]
                feature_norm = feature_norm * weight            # [N, 1, C]
            
            feature = feature * feature_norm
            
            if self.affine and l == 0: 
                bias = self.affine_bias
                bias = bias.view(1, 1, -1)
                feature = feature + bias
            
            out.append(feature)
        
        out = torch.cat(out, dim=1)
        
        return out 



class EquivariantLayerNormArraySphericalHarmonics(nn.Module):
    '''
        1. Normalize over L = 0.
        2. Normalize across all m components from degrees L > 0.
        3. Do not normalize separately for different L (L > 0).
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', std_balance_degrees=True):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.std_balance_degrees = std_balance_degrees
        
        # for L = 0
        self.norm_l0 = torch.nn.LayerNorm(self.num_channels, eps=self.eps, elementwise_affine=self.affine)

        # for L > 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.lmax, self.num_channels))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2 - 1, 1)
            for l in range(1, self.lmax + 1):
                start_idx = l ** 2 - 1
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / self.lmax
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast(device_type="cuda",enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        
        out = []

        # for L = 0
        feature = node_input.narrow(1, 0, 1)
        feature = self.norm_l0(feature)
        out.append(feature)

        # for L > 0
        if self.lmax > 0:
            num_m_components = (self.lmax + 1) ** 2
            feature = node_input.narrow(1, 1, num_m_components - 1)

            # Then compute the rescaling factor (norm of each feature vector)
            # Rescaling of the norms themselves based on the option "normalization"
            if self.normalization == 'norm':
                feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2 - 1, C], without L = 0
                    feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
                else:
                    feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            
            feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
            feature_norm = (feature_norm + self.eps).pow(-0.5)

            for l in range(1, self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                feature = node_input.narrow(1, start_idx, length)       # [N, (2L + 1), C]
                if self.affine:
                    weight = self.affine_weight.narrow(0, (l - 1), 1)       # [1, C]
                    weight = weight.view(1, 1, -1)                          # [1, 1, C]
                    feature_scale = feature_norm * weight                   # [N, 1, C]
                else:
                    feature_scale = feature_norm
                feature = feature * feature_scale
                out.append(feature)
            
        out = torch.cat(out, dim=1)
        return out

    
class EquivariantRMSNormArraySphericalHarmonics(nn.Module):
    '''
        1. Normalize across all m components from degrees L >= 0.
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component'):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        
        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps})"


    @torch.amp.autocast(device_type="cuda",enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        
        out = []

        # for L >= 0
        feature = node_input    
        if self.normalization == 'norm':
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
        elif self.normalization == 'component':
            feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        for l in range(0, self.lmax + 1):
            start_idx = l ** 2
            length = 2 * l + 1
            feature = node_input.narrow(1, start_idx, length)       # [N, (2L + 1), C]
            if self.affine:
                weight = self.affine_weight.narrow(0, l, 1)         # [1, C]
                weight = weight.view(1, 1, -1)                      # [1, 1, C]
                feature_scale = feature_norm * weight               # [N, 1, C]
            else:
                feature_scale = feature_norm
            feature = feature * feature_scale
            out.append(feature)
            
        out = torch.cat(out, dim=1)
        return out


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    '''
        1. Normalize across all m components from degrees L >= 0.
        2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    '''
    def __init__(self, lmax, num_channels, eps=1e-5, affine=True, normalization='component', centering=True, std_balance_degrees=True):
        super().__init__()

        self.lmax = lmax
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees
        
        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones((self.lmax + 1), self.num_channels))
            if self.centering:
                self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))
            else:
                self.register_parameter('affine_bias', None)
        else:
            self.register_parameter('affine_weight', None)
            self.register_parameter('affine_bias', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer('expand_index', expand_index)

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros((self.lmax + 1) ** 2, 1)
            for l in range(self.lmax + 1):
                start_idx = l ** 2
                length = 2 * l + 1
                balance_degree_weight[start_idx : (start_idx + length), :] = (1.0 / length)
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast(device_type="cuda",enabled=False)
    def forward(self, node_input):
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        
        feature = node_input    
        
        if self.centering:
            feature_l0 = feature.narrow(1, 0, 1)
            feature_l0_mean = feature_l0.mean(dim=2, keepdim=True) # [N, 1, 1]
            feature_l0 = feature_l0 - feature_l0_mean
            feature = torch.cat((feature_l0, feature.narrow(1, 1, feature.shape[1] - 1)), dim=1)
            
        # for L >= 0
        if self.normalization == 'norm':
            assert not self.std_balance_degrees
            feature_norm = feature.pow(2).sum(dim=1, keepdim=True)      # [N, 1, C]
        elif self.normalization == 'component':
            if self.std_balance_degrees:
                feature_norm = feature.pow(2)                               # [N, (L_max + 1)**2, C]
                feature_norm = torch.einsum('nic, ia -> nac', feature_norm, self.balance_degree_weight) # [N, 1, C]
            else:
                feature_norm = feature.pow(2).mean(dim=1, keepdim=True)     # [N, 1, C]
            
        feature_norm = torch.mean(feature_norm, dim=2, keepdim=True)    # [N, 1, 1]
        feature_norm = (feature_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight.view(1, (self.lmax + 1), self.num_channels)     # [1, L_max + 1, C]
            weight = torch.index_select(weight, dim=1, index=self.expand_index)         # [1, (L_max + 1)**2, C]
            feature_norm = feature_norm * weight                                        # [N, (L_max + 1)**2, C]
        
        out = feature * feature_norm

        if self.affine and self.centering:
            out[:, 0:1, :] = out.narrow(1, 0, 1) + self.affine_bias.view(1, 1, self.num_channels)

        return out


class EquivariantDegreeLayerScale(nn.Module):
    '''
        1. Similar to Layer Scale used in CaiT (Going Deeper With Image Transformers (ICCV'21)), we scale the output of both attention and FFN. 
        2. For degree L > 0, we scale down the square root of 2 * L, which is to emulate halving the number of channels when using higher L. 
    '''
    def __init__(self, lmax, num_channels, scale_factor=2.0):
        super().__init__()
        
        self.lmax = lmax
        self.num_channels = num_channels
        self.scale_factor = scale_factor

        self.affine_weight = nn.Parameter(torch.ones(1, (self.lmax + 1), self.num_channels))
        for l in range(1, self.lmax + 1):
            self.affine_weight.data[0, l, :].mul_(1.0 / math.sqrt(self.scale_factor * l))        
        expand_index = get_l_to_all_m_expand_index(self.lmax)
        self.register_buffer('expand_index', expand_index)


    def __repr__(self):
        return f"{self.__class__.__name__}(lmax={self.lmax}, num_channels={self.num_channels}, scale_factor={self.scale_factor})"

    
    def forward(self, node_input):
        weight = torch.index_select(self.affine_weight, dim=1, index=self.expand_index)     # [1, (L_max + 1)**2, C]
        node_input = node_input * weight                                                    # [N, (L_max + 1)**2, C]
        return node_input
