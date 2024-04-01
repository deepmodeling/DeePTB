import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import degree
from scipy.optimize import brentq
from scipy import special as sp
from e3nn.util.jit import compile_mode
from e3nn.o3 import Irrep, Irreps, wigner_3j, matrix_to_angles, Linear, FullyConnectedTensorProduct, TensorProduct, SphericalHarmonics
from e3nn.nn import Extract
import numpy as np
from typing import Union
import e3nn.o3 as o3
from ...cutoff import polynomial_cutoff
import sympy as sym


def spherical_bessel_formulas(n):
    """
    Computes the sympy formulas for the spherical bessel functions up to order n (excluded)
    """
    x = sym.symbols('x')

    f = [sym.sin(x)/x]
    a = sym.sin(x)/x
    for i in range(1, n):
        b = sym.diff(a, x)/x
        f += [sym.simplify(b*(-x)**i)]
        a = sym.simplify(b)
    return f

def Jn(r, n):
    """
    numerical spherical bessel functions of order n
    """
    return np.sqrt(np.pi/(2*r)) * sp.jv(n+0.5, r)


def Jn_zeros(n, k):
    """
    Compute the first k zeros of the spherical bessel functions up to order n (excluded)
    """
    zerosj = np.zeros((n, k), dtype="float32")
    zerosj[0] = np.arange(1, k + 1) * np.pi
    points = np.arange(1, k + n) * np.pi
    racines = np.zeros(k + n - 1, dtype="float32")
    for i in range(1, n):
        for j in range(k + n - 1 - i):
            foo = brentq(Jn, points[j], points[j + 1], (i,))
            racines[j] = foo
        points = racines
        zerosj[i][:k] = racines[:k]

    return zerosj

def bessel_basis(n, k):
    """
    Compute the sympy formulas for the normalized and rescaled spherical bessel functions up to
    order n (excluded) and maximum frequency k (excluded).
    """

    zeros = Jn_zeros(n, k)
    normalizer = []
    for order in range(n):
        normalizer_tmp = []
        for i in range(k):
            normalizer_tmp += [0.5*Jn(zeros[order, i], order+1)**2]
        normalizer_tmp = 1/np.array(normalizer_tmp)**0.5
        normalizer += [normalizer_tmp]

    f = spherical_bessel_formulas(n)
    x = sym.symbols('x')
    bess_basis = []
    for order in range(n):
        bess_basis_tmp = []
        for i in range(k):
            bess_basis_tmp += [sym.simplify(normalizer[order]
                                            [i]*f[order].subs(x, zeros[order, i]*x))]
        bess_basis += [bess_basis_tmp]
    return bess_basis

class sort_irreps(torch.nn.Module):
    def __init__(self, irreps_in):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        sorted_irreps = irreps_in.sort()
        
        irreps_out_list = [((mul, ir),) for mul, ir in sorted_irreps.irreps]
        instructions = [(i,) for i in sorted_irreps.inv]
        self.extr = Extract(irreps_in, irreps_out_list, instructions)
        
        irreps_in_list = [((mul, ir),) for mul, ir in irreps_in]
        instructions_inv = [(i,) for i in sorted_irreps.p]
        self.extr_inv = Extract(sorted_irreps.irreps, irreps_in_list, instructions_inv)
        
        self.irreps_in = irreps_in
        self.irreps_out = sorted_irreps.irreps.simplify()
    
    def forward(self, x):
        r'''irreps_in -> irreps_out'''
        extracted = self.extr(x)
        return torch.cat(extracted, dim=-1)

    def inverse(self, x):
        r'''irreps_out -> irreps_in'''
        extracted_inv = self.extr_inv(x)
        return torch.cat(extracted_inv, dim=-1)



        

class e3LayerNorm(nn.Module):
    def __init__(self, irreps_in, eps=1e-5, affine=True, normalization='component', subtract_mean=True, divide_norm=False):
        super().__init__()
        
        self.irreps_in = Irreps(irreps_in)
        self.eps = eps
        
        if affine:
            ib, iw = 0, 0
            weight_slices, bias_slices = [], []
            for mul, ir in irreps_in:
                if ir.is_scalar(): # bias only to 0e
                    bias_slices.append(slice(ib, ib + mul))
                    ib += mul
                else:
                    bias_slices.append(None)
                weight_slices.append(slice(iw, iw + mul))
                iw += mul
            self.weight = nn.Parameter(torch.ones([iw]))
            self.bias = nn.Parameter(torch.zeros([ib]))
            self.bias_slices = bias_slices
            self.weight_slices = weight_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        self.subtract_mean = subtract_mean
        self.divide_norm = divide_norm
        assert normalization in ['component', 'norm']
        self.normalization = normalization
            
        self.reset_parameters()
    
    def reset_parameters(self):
        if self.weight is not None:
            self.weight.data.fill_(1)
            # nn.init.uniform_(self.weight)
        if self.bias is not None:
            self.bias.data.fill_(0)
            # nn.init.uniform_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        # input x must have shape [num_node(edge), dim]
        # if first dimension of x is node index, then batch should be batch.batch
        # if first dimension of x is edge index, then batch should be batch.batch[batch.edge_index[0]]
        
        if batch is None:
            batch = torch.full([x.shape[0]], 0, dtype=torch.int64)

        # from torch_geometric.nn.norm.LayerNorm

        batch_size = int(batch.max()) + 1 
        batch_degree = degree(batch, batch_size, dtype=torch.int64).clamp_(min=1).to(dtype=x.dtype)
        
        out = []
        ix = 0
        for index, (mul, ir) in enumerate(self.irreps_in):        
            field = x[:, ix: ix + mul * ir.dim].reshape(-1, mul, ir.dim) # [node, mul, repr]
            
            # compute and subtract mean
            if self.subtract_mean or ir.l == 0: # do not subtract mean for l>0 irreps if subtract_mean=False
                mean = scatter(field, batch, dim=0, dim_size=batch_size,
                            reduce='add').mean(dim=1, keepdim=True) / batch_degree[:, None, None] # scatter_mean does not support complex number
                field = field - mean[batch]
                
            # compute and divide norm
            if self.divide_norm or ir.l == 0: # do not divide norm for l>0 irreps if divide_norm=False
                norm = scatter(field.abs().pow(2), batch, dim=0, dim_size=batch_size,
                            reduce='mean').mean(dim=[1,2], keepdim=True) # add abs here to deal with complex numbers
                if self.normalization == 'norm':
                    norm = norm * ir.dim
                field = field / (norm.sqrt()[batch] + self.eps)
            
            # affine
            if self.weight is not None:
                weight = self.weight[self.weight_slices[index]]
                field = field * weight[None, :, None]
            if self.bias is not None and ir.is_scalar():
                bias = self.bias[self.bias_slices[index]]
                field = field + bias[None, :, None]
            
            out.append(field.reshape(-1, mul * ir.dim))
            ix += mul * ir.dim
            
        out = torch.cat(out, dim=-1)
                
        return out

class e3ElementWise:
    def __init__(self, irreps_in):
        self.irreps_in = Irreps(irreps_in)
        
        len_weight = 0
        for mul, ir in self.irreps_in:
            len_weight += mul
        
        self.len_weight = len_weight
    
    def __call__(self, x: torch.Tensor, weight: torch.Tensor):
        # x should have shape [edge/node, channels]
        # weight should have shape [edge/node, self.len_weight]
        
        ix = 0
        iw = 0
        out = []
        for mul, ir in self.irreps_in:
            field = x[:, ix: ix + mul * ir.dim]
            field = field.reshape(-1, mul, ir.dim)
            field = field * weight[:, iw: iw + mul][:, :, None]
            field = field.reshape(-1, mul * ir.dim)
            
            ix += mul * ir.dim
            iw += mul
            out.append(field)
        
        return torch.cat(out, dim=-1)


class SkipConnection(nn.Module):
    def __init__(self, irreps_in, irreps_out, is_complex=False):
        super().__init__()
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        self.sc = None
        if irreps_in == irreps_out:
            self.sc = None
        else:
            self.sc = Linear(irreps_in=irreps_in, irreps_out=irreps_out)
    
    def forward(self, old, new):
        if self.sc is not None:
            old = self.sc(old)
        
        return old + new


class SelfTp(nn.Module):
    def __init__(self, irreps_in, irreps_out, **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}x_k (k>=j)'''
        super().__init__()
        
        assert not kwargs.pop('internal_weights', False) # internal weights must be True
        assert kwargs.pop('shared_weights', True) # shared weights must be false
        
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)
        
        instr_tp = []
        weights1, weights2 = [], []
        for i1, (mul1, ir1) in enumerate(irreps_in):
            for i2 in range(i1, len(irreps_in)):
                mul2, ir2 = irreps_in[i2]
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
        
        self.tp = TensorProduct(irreps_in, irreps_in, irreps_out, instr_tp, internal_weights=False, shared_weights=True, **kwargs)
        
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)
        
    def forward(self, x):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        weights = torch.cat(weights)
        return self.tp(x, x, weights)
    
@compile_mode("script")
class SeparateWeightTensorProduct(nn.Module):
    def __init__(self, irreps_in1: Union[str, o3.Irreps], irreps_in2: Union[str, o3.Irreps], irreps_out: Union[str, o3.Irreps], **kwargs):
        '''z_i = W'_{ij}x_j W''_{ik}y_k'''
        super(SeparateWeightTensorProduct, self).__init__()
        
        assert not kwargs.pop('internal_weights', False) # internal weights must be True
        assert kwargs.pop('shared_weights', True) # shared weights must be false
        
        irreps_in1 = Irreps(irreps_in1)
        irreps_in2 = Irreps(irreps_in2)
        irreps_out = Irreps(irreps_out)
        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
                
        instr_tp = []
        weights1, weights2 = [], []
        # weight = []
        for i1, (mul1, ir1) in enumerate(irreps_in1):
            for i2, (mul2, ir2) in enumerate(irreps_in2):
                for i_out, (mul_out, ir3) in enumerate(irreps_out):
                    if ir3 in ir1 * ir2:
                        weights1.append(nn.Parameter(torch.randn(mul1, mul_out)))
                        weights2.append(nn.Parameter(torch.randn(mul2, mul_out)))
                        # weight.append(nn.Parameter(torch.randn(mul1, mul2, mul_out)).view(-1))
                        instr_tp.append((i1, i2, i_out, 'uvw', True, 1.0))
        
        self.tp = TensorProduct(irreps_in1, irreps_in2, irreps_out, instr_tp, internal_weights=False, shared_weights=True, **kwargs)
        
        self.weights1 = nn.ParameterList(weights1)
        self.weights2 = nn.ParameterList(weights2)
        # self.weight = nn.ParameterList(weight)
        
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        weights = []
        for weight1, weight2 in zip(self.weights1, self.weights2):
            weight = weight1[:, None, :] * weight2[None, :, :]
            weights.append(weight.view(-1))
        # for w in self.weight:
            # weights.append(w)
        weights = torch.cat(weights)
        return self.tp(x1, x2, weights)


class SphericalBasis(nn.Module):
    def __init__(self, target_irreps, rcutoff, eps=1e-7, dtype=torch.get_default_dtype()):
        super().__init__()
        
        target_irreps = Irreps(target_irreps)
        
        self.sh = SphericalHarmonics(
            irreps_out=target_irreps,
            normalize=True,
            normalization='component',
        )
        
        max_order = max(map(lambda x: x[1].l, target_irreps)) # maximum angular momentum l
        max_freq = max(map(lambda x: x[0], target_irreps)) # maximum multiplicity
        
        basis = bessel_basis(max_order + 1, max_freq)
        lambdify_torch = {
            # '+': torch.add,
            # '-': torch.sub,
            # '*': torch.mul,
            # '/': torch.div,
            # '**': torch.pow,
            'sin': torch.sin,
            'cos': torch.cos
        }
        x = sym.symbols('x')
        funcs = []
        for mul, ir in target_irreps:
            for freq in range(mul):
                funcs.append(sym.lambdify([x], basis[ir.l][freq], [lambdify_torch]))
                
        self.bessel_funcs = funcs
        self.multiplier = e3ElementWise(target_irreps)
        self.dtype = dtype
        self.cutoff = polynomial_cutoff
        self.register_buffer('rcutoff', torch.Tensor([rcutoff]))
        self.irreps_out = target_irreps
        self.register_buffer('eps', torch.Tensor([eps]))
        
    def forward(self, length, direction):
        # direction should be in y, z, x order
        sh = self.sh(direction).type(self.dtype)
        sbf = torch.stack([f((length + self.eps) / self.rcutoff) for f in self.bessel_funcs], dim=-1)
        return self.multiplier(sh, sbf) * self.cutoff(x=length, r_max=self.rcutoff, p=6).flatten()[:, None]