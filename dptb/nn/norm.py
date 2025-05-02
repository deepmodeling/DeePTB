import torch
from torch import nn

from e3nn import o3
from e3nn.util.jit import compile_mode
from torch_scatter import scatter_mean
from typing import Union

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

        count_scales= 0
        count_shift = 0
        self.shift_index = []
        self.scale_index = []
        for mul, ir in self.irreps:
            if str(ir) == "0e":
                self.num_scalar += mul
                self.shift_index += list(range(count_shift, count_shift + mul))
                count_shift += mul
            else:
                self.shift_index += [-1] * mul * ir.dim

            for _ in range(mul):
                self.scale_index += [count_scales] * ir.dim
                count_scales += 1
        
        self.shift_index = torch.as_tensor(self.shift_index, dtype=torch.int64, device=self.device)
        self.scale_index = torch.as_tensor(self.scale_index, dtype=torch.int64, device=self.device)

        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.irreps.num_irreps))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_scalar))
        else:
            self.register_parameter('affine_weight', None)

        assert normalization in ['norm', 'component']
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros(self.irreps.num_irreps)
            count = 0
            for mul, ir in self.irreps:
                if ir.l == 0:
                    balance_degree_weight[count:count+mul] = self.lmax # to make the devided value to 1.0
                else:
                    balance_degree_weight[count:count+mul] = (1.0 / ir.dim / mul)
                count += mul
            balance_degree_weight = balance_degree_weight / sum([1 for mul, ir in self.irreps if ir.l > 0])
            self.register_buffer('balance_degree_weight', balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self):
        return f"{self.__class__.__name__}(irreps={self.irreps}, eps={self.eps}, std_balance_degrees={self.std_balance_degrees})"


    @torch.amp.autocast(device_type="cuda",enabled=False)
    def forward(self, x):
        '''
            Assume input is of shape [N, sphere_basis, C]
        '''
        batch, dim = x.shape
        x = x.reshape(batch, dim)  # [batch, stacked features]

        # 1. shift the 0e value with their mean
        # 2. compute the weight of all the features
        # 3. do multiplication

        # 1
        feature_mean = x[:, self.shift_index.ge(0)].mean(dim=1, keepdim=True)
        x = x + 0. # to avoid the inplace operation of x
        x[:, self.shift_index.ge(0)] = x[:, self.shift_index.ge(0)] - feature_mean

        # 2. compute the norm across all irreps except for 0e
        if self.lmax > 0:
            if self.normalization == 'norm':
                feature_norm = x[:,self.shift_index.lt(0)].pow(2).sum(1, keepdim=True)              # [N, 1]
            elif self.normalization == 'component':
                if self.std_balance_degrees:
                    feature_norm = x.pow(2)                              
                    feature_norm = feature_norm * self.balance_degree_weight[self.scale_index] # [N, dim]
                    feature_norm = feature_norm[:, self.shift_index.lt(0)].sum(1, keepdim=True) # [N, 1]
                else:
                    feature_norm = x[:,self.shift_index.lt(0)].pow(2).sum(dim=1, keepdim=True)     # [N, 1]
        else:
            if self.normalization == 'norm':
                feature_norm = x[:,self.shift_index.lt(0)].pow(2).sum(1, keepdim=True)
            else:
                feature_norm = x[:,self.shift_index.lt(0)].pow(2).mean(1, keepdim=True)

        feature_norm = (feature_norm + self.eps).pow(-0.5)
        weight = self.affine_weight * feature_norm # [1, n_ir] * [N, 1] = [N, n_ir]

        x = x * weight[:, self.scale_index]
        x[:, self.shift_index.ge(0)] = x[:, self.shift_index.ge(0)] + self.affine_bias

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
