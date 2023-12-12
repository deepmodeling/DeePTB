from typing import Optional, List, Union, Dict
import math
import functools
import warnings

import torch
from torch_runstats.scatter import scatter

from torch import fx
from e3nn.util.codegen import CodeGenMixin
from e3nn import o3
from e3nn.nn import Gate, Activation
from e3nn.o3 import TensorProduct, Linear, SphericalHarmonics
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode

from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from dptb.nn.graph_mixin import GraphModuleMixin
from dptb.nn.embedding.from_deephe3.deephe3 import tp_path_exists
from dptb.data import _keys
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
import math
from dptb.data.transforms import OrbitalMapper
from ..type_encode.one_hot import OneHotAtomEncoding
from dptb.data.AtomicDataDict import with_edge_vectors, with_env_vectors, with_batch

from math import ceil

@Embedding.register("e3baseline")
class E3BaseLineModel(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            # required params
            n_atom: int=1,
            n_layers: int=3,
            n_radial_basis: int=10,
            r_max: float=5.0,
            lmax: int=4,
            irreps_hidden: o3.Irreps=None,
            avg_num_neighbors: Optional[float] = None,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            # general hyperparameters:
            linear_after_env_embed: bool = False,
            env_embed_multiplicity: int = 32,
            sh_normalized: bool = True,
            sh_normalization: str = "component",
            # MLP parameters:
            latent_kwargs={
                "mlp_latent_dimensions": [256, 512, 1024],
                "mlp_nonlinearity": "silu",
                "mlp_initialization": "uniform"
            },
            latent_resnet: bool = True,
            latent_resnet_update_ratios: Optional[List[float]] = None,
            latent_resnet_update_ratios_learnable: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            ):
        
        super(E3BaseLineModel, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)

        irreps_sh=o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        node_irreps = self.idp.node_irreps.sort()[0].simplify()
        pair_irreps = self.idp.pair_irreps.sort()[0].simplify()

        # check if the irreps setting satisfied the requirement of idp
        assert all(ir in irreps_hidden for _, ir in pair_irreps), "hidden irreps should at least cover all the reqired irreps in the hamiltonian data {}.format(pair_irreps)"
        assert all(ir in irreps_hidden for _, ir in node_irreps), "hidden irreps should at least cover all the reqired irreps in the hamiltonian data {}.format(node_irreps)"

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)

        self.init_layer = InitLayer(
            num_types=n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max,
            irreps_sh=irreps_sh,
            irreps_out=irreps_hidden,
            # MLP parameters:
            two_body_latent_kwargs=latent_kwargs,
            env_embed_kwargs = {
                "mlp_latent_dimensions": [],
                "mlp_nonlinearity": None,
                "mlp_initialization": "uniform"
            },
            # cutoffs
            r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p,
            cutoff_type=cutoff_type,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList()
        latent_in =latent_kwargs["mlp_latent_dimensions"][-1]
        # actually, we can derive the least required irreps_in and out from the idp's node and pair irreps
        for _ in range(n_layers):
            self.layers.append(Layer(
                avg_num_neighbors=avg_num_neighbors,
                irreps_sh=irreps_sh,
                irreps_in=irreps_hidden,
                irreps_out=irreps_hidden,
                # general hyperparameters:
                linear_after_env_embed=linear_after_env_embed,
                env_embed_multiplicity=env_embed_multiplicity,
                # MLP parameters:
                latent_kwargs=latent_kwargs,
                latent_in=latent_in,
                latent_resnet=latent_resnet,
                latent_resnet_update_ratios=latent_resnet_update_ratios,
                latent_resnet_update_ratios_learnable=latent_resnet_update_ratios_learnable,
                )
            )

        # initilize output_layer
        self.out_edge = Linear(self.layers[-1].irreps_out, self.idp.pair_irreps, shared_weights=True, internal_weights=True)
        self.out_node = torch.nn.Linear(latent_in, self.idp.node_irreps.dim, bias=True)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        # data = with_env_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        
        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        latents, features, cutoff_coeffs, active_edges = self.init_layer(edge_index, edge_sh, edge_length, node_one_hot)

        for layer in self.layers:
            latents, features, cutoff_coeffs, active_edges = layer(edge_index, edge_sh, atom_type, latents, features, cutoff_coeffs, active_edges)

        
        if self.layers[-1].env_sum_normalizations.ndim < 1:
            norm_const = self.layers[-1].env_sum_normalizations
        else:
            norm_const = self.layers[-1].env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)

        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.pair_irreps.dim, dtype=self.dtype, device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges, self.out_edge(features))
        node_features = scatter(latents, edge_index[0], dim=0)
        data[_keys.NODE_FEATURES_KEY] = self.out_node(node_features * norm_const)

        return data

def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

def get_gate_nonlin(irreps_in1, irreps_in2, irreps_out, 
                    act={1: torch.nn.functional.silu, -1: torch.tanh}, 
                    act_gates={1: torch.sigmoid, -1: torch.tanh}
                    ):
    # get gate nonlinearity after tensor product
    # irreps_in1 and irreps_in2 are irreps to be multiplied in tensor product
    # irreps_out is desired irreps after gate nonlin
    # notice that nonlin.irreps_out might not be exactly equal to irreps_out
            
    irreps_scalars = o3.Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l == 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    irreps_gated = o3.Irreps([
        (mul, ir)
        for mul, ir in irreps_out
        if ir.l > 0 and tp_path_exists(irreps_in1, irreps_in2, ir)
    ]).simplify()
    if irreps_gated.dim > 0:
        if tp_path_exists(irreps_in1, irreps_in2, "0e"):
            ir = "0e"
        elif tp_path_exists(irreps_in1, irreps_in2, "0o"):
            ir = "0o"
            warnings.warn('Using odd representations as gates')
        else:
            raise ValueError(
                f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} is unable to produce gates needed for irreps_gated={irreps_gated}")
    else:
        ir = None
    irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated]).simplify()

    gate_nonlin = Gate(
        irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],  # scalar
        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        irreps_gated  # gated tensors
    )
    
    return gate_nonlin


@compile_mode("script")
class MakeWeightedChannels(torch.nn.Module):
    weight_numel: int
    multiplicity_out: Union[int, list]
    _num_irreps: int

    def __init__(
        self,
        irreps_in,
        multiplicity_out: Union[int, list],
        pad_to_alignment: int = 1,
    ):
        super().__init__()
        assert all(mul == 1 for mul, _ in irreps_in)
        assert multiplicity_out >= 1
        # Each edgewise output multiplicity is a per-irrep weighted sum over the input
        # So we need to apply the weight for the ith irrep to all DOF in that irrep
        w_index = []
        idx = 0
        self._num_irreps = 0
        for (mul, ir) in irreps_in:
            w_index += sum(([ix] * ir.dim for ix in range(idx, idx + mul)), [])
            idx += mul
            self._num_irreps += mul
        # w_index = sum(([i] * ir.dim for i, (mul, ir) in enumerate(irreps_in)), [])
        # pad to padded length
        n_pad = (
            int(ceil(irreps_in.dim / pad_to_alignment)) * pad_to_alignment
            - irreps_in.dim
        )
        # use the last weight, what we use doesn't matter much
        w_index += [w_index[-1]] * n_pad
        self.register_buffer("_w_index", torch.as_tensor(w_index, dtype=torch.long))
        # there is
        self.multiplicity_out = multiplicity_out
        self.weight_numel = self._num_irreps * multiplicity_out

    def forward(self, edge_attr, weights):
        # weights are [z, u, num_i]
        # edge_attr are [z, i]
        # i runs over all irreps, which is why the weights need
        # to be indexed in order to go from [num_i] to [i]
        return torch.einsum(
            "zi,zui->zui",
            edge_attr,
            weights.view(
                -1,
                self.multiplicity_out,
                self._num_irreps,
            )[:, :, self._w_index],
        )
    
@torch.jit.script
def ShiftedSoftPlus(x):
    return torch.nn.functional.softplus(x) - math.log(2.0)

class ScalarMLPFunction(CodeGenMixin, torch.nn.Module):
    """Module implementing an MLP according to provided options."""

    in_features: int
    out_features: int

    def __init__(
        self,
        mlp_input_dimension: Optional[int],
        mlp_latent_dimensions: List[int],
        mlp_output_dimension: Optional[int],
        mlp_nonlinearity: Optional[str] = "silu",
        mlp_initialization: str = "normal",
        mlp_dropout_p: float = 0.0,
        mlp_batchnorm: bool = False,
    ):
        super().__init__()
        nonlinearity = {
            None: None,
            "silu": torch.nn.functional.silu,
            "ssp": ShiftedSoftPlus,
        }[mlp_nonlinearity]
        if nonlinearity is not None:
            nonlin_const = normalize2mom(nonlinearity).cst
        else:
            nonlin_const = 1.0

        dimensions = (
            ([mlp_input_dimension] if mlp_input_dimension is not None else [])
            + mlp_latent_dimensions
            + ([mlp_output_dimension] if mlp_output_dimension is not None else [])
        )
        assert len(dimensions) >= 2  # Must have input and output dim
        num_layers = len(dimensions) - 1

        self.in_features = dimensions[0]
        self.out_features = dimensions[-1]

        # Code
        params = {}
        graph = fx.Graph()
        tracer = fx.proxy.GraphAppendingTracer(graph)

        def Proxy(n):
            return fx.Proxy(n, tracer=tracer)

        features = Proxy(graph.placeholder("x"))
        norm_from_last: float = 1.0

        base = torch.nn.Module()

        for layer, (h_in, h_out) in enumerate(zip(dimensions, dimensions[1:])):
            # do dropout
            if mlp_dropout_p > 0:
                # only dropout if it will do something
                # dropout before linear projection- https://stats.stackexchange.com/a/245137
                features = Proxy(graph.call_module("_dropout", (features.node,)))

            # make weights
            w = torch.empty(h_in, h_out)

            if mlp_initialization == "normal":
                w.normal_()
            elif mlp_initialization == "uniform":
                # these values give < x^2 > = 1
                w.uniform_(-math.sqrt(3), math.sqrt(3))
            elif mlp_initialization == "orthogonal":
                # this rescaling gives < x^2 > = 1
                torch.nn.init.orthogonal_(w, gain=math.sqrt(max(w.shape)))
            else:
                raise NotImplementedError(
                    f"Invalid mlp_initialization {mlp_initialization}"
                )

            # generate code
            params[f"_weight_{layer}"] = w
            w = Proxy(graph.get_attr(f"_weight_{layer}"))
            w = w * (
                norm_from_last / math.sqrt(float(h_in))
            )  # include any nonlinearity normalization from previous layers
            features = torch.matmul(features, w)

            if mlp_batchnorm:
                # if we call batchnorm, do it after the nonlinearity
                features = Proxy(graph.call_module(f"_bn_{layer}", (features.node,)))
                setattr(base, f"_bn_{layer}", torch.nn.BatchNorm1d(h_out))

            # generate nonlinearity code
            if nonlinearity is not None and layer < num_layers - 1:
                features = nonlinearity(features)
                # add the normalization const in next layer
                norm_from_last = nonlin_const

        graph.output(features.node)

        for pname, p in params.items():
            setattr(base, pname, torch.nn.Parameter(p))

        if mlp_dropout_p > 0:
            # with normal dropout everything blows up
            base._dropout = torch.nn.AlphaDropout(p=mlp_dropout_p)

        self._codegen_register({"_forward": fx.GraphModule(base, graph)})

    def forward(self, x):
        return self._forward(x)

class InitLayer(torch.nn.Module):
    def __init__(
            self,
            # required params
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            irreps_sh: o3.Irreps=None,
            irreps_out: o3.Irreps=None,
            # MLP parameters:
            two_body_latent_kwargs={
                "mlp_latent_dimensions": [128, 256, 512, 1024],
                "mlp_nonlinearity": "silu",
                "mlp_initialization": "uniform"
            },
            env_embed_kwargs = {
                "mlp_latent_dimensions": [],
                "mlp_nonlinearity": None,
                "mlp_initialization": "uniform"
            },
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        self.r_max = torch.tensor(r_max, device=device, dtype=dtype)
        self.two_body_latent_kwargs = two_body_latent_kwargs
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype

        assert all(mul==1 for mul, _ in irreps_sh)
        # env_embed_irreps = o3.Irreps([(1, ir) for _, ir in irreps_sh])
        assert (
            irreps_sh[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

        # Node invariants for center and neighbor (chemistry)
        # Plus edge invariants for the edge (radius).
        self.two_body_latent = ScalarMLPFunction(
                        mlp_input_dimension=(2 * num_types + n_radial_basis),
                        mlp_output_dimension=None,
                        **two_body_latent_kwargs,
                    )
        
        tp_irreps_out = []
        for mul, ir1 in irreps_sh:
            for mul, ir2 in irreps_sh:
                for ir_out in ir1*ir2:
                    if ir_out in irreps_out:
                        tp_irreps_out.append((1, ir_out))
        tp_irreps_out = o3.Irreps(tp_irreps_out)
        assert all(ir in tp_irreps_out for _, ir in irreps_out), "embeded spherical irreps should cover the space of required output, enlarge lmax if necessary"

        self.tp = o3.TensorSquare(
            irreps_in=irreps_sh,
            irreps_out=tp_irreps_out,
            irrep_normalization="component"
        )

        self._env_weighter = Linear(
            irreps_in=self.tp.irreps_out,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
            path_normalization = "element", # if path normalization is element and input irreps has 1 mul, it should not have effect ! 
        )

        self.env_embed_mlp = ScalarMLPFunction(
                        mlp_input_dimension=self.two_body_latent.out_features,
                        mlp_output_dimension=self._env_weighter.weight_numel,
                        **env_embed_kwargs,
                    )
        self.bessel = BesselBasis(r_max=self.r_max, num_basis=n_radial_basis, trainable=True)



    def forward(self, edge_index, edge_sh, edge_length, node_one_hot):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        edge_invariants = self.bessel(edge_length)
        node_invariants = node_one_hot

        # Vectorized precompute per layer cutoffs
        if self.cutoff_type == "cosine":
            cutoff_coeffs = cosine_cutoff(
                edge_length,
                self.r_max.reshape(-1),
                r_start_cos_ratio=self.r_start_cos_ratio,
            ).flatten()

        elif self.cutoff_type == "polynomial":
            cutoff_coeffs = polynomial_cutoff(
                edge_length, self.r_max.reshape(-1), p=self.polynomial_cutoff_p
            ).flatten()

        else:
            # This branch is unreachable (cutoff type is checked in __init__)
            # But TorchScript doesn't know that, so we need to make it explicitly
            # impossible to make it past so it doesn't throw
            # "cutoff_coeffs_all is not defined in the false branch"
            assert False, "Invalid cutoff type"

        # Determine which edges are still in play
        prev_mask = cutoff_coeffs > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        # Compute latents
        latents = torch.zeros(
            (edge_sh.shape[0], self.two_body_latent.out_features),
            dtype=edge_sh.dtype,
            device=edge_sh.device,
        )
        
        new_latents = self.two_body_latent(torch.cat([
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ], dim=-1)[prev_mask])
        # Apply cutoff, which propagates through to everything else
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        latents = torch.index_copy(latents, 0, active_edges, new_latents)
        weights = self.env_embed_mlp(latents[active_edges])

        # embed initial edge
        features = self._env_weighter(
            self.tp(edge_sh[prev_mask]), weights
        )  # features is edge_attr

        return latents, features, cutoff_coeffs, active_edges # the radial embedding x and the sperical hidden V

class Layer(torch.nn.Module):
    def __init__(
        self,
        # required params
        avg_num_neighbors: Optional[float] = None,
        irreps_sh: o3.Irreps=None,
        irreps_in: o3.Irreps=None,
        irreps_out: o3.Irreps=None,
        # general hyperparameters:
        linear_after_env_embed: bool = False,
        env_embed_multiplicity: int = 32,
        # MLP parameters:
        latent_kwargs={
            "mlp_latent_dimensions": [128, 256, 512, 1024],
            "mlp_nonlinearity": "silu",
            "mlp_initialization": "uniform"
        },
        latent_in: int=1024,
        latent_resnet: bool = True,
        latent_resnet_update_ratios: Optional[List[float]] = None,
        latent_resnet_update_ratios_learnable: bool = False,
    ):
        super().__init__()
        SCALAR = o3.Irrep("0e")
        self.latent_resnet = latent_resnet
        self.avg_num_neighbors = avg_num_neighbors
        self.linear_after_env_embed = linear_after_env_embed
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        assert all(mul==1 for mul, _ in irreps_sh)

        # for normalization of env embed sums
        # one per layer
        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        latent = functools.partial(ScalarMLPFunction, **latent_kwargs)

        self.latents = None
        self.env_embed_mlps = None
        self.tps = None
        self.linears = None
        self.env_linears = None

        # Prune impossible paths
        self.irreps_out = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_out
                    if tp_path_exists(irreps_sh, irreps_in, ir)
                ]
            )

        mul_irreps_sh = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])
        self._env_weighter = Linear(
            irreps_in=irreps_sh,
            irreps_out=mul_irreps_sh,
            internal_weights=False,
            shared_weights=False,
            path_normalization = "element",
        )
        
        # == Remove unneeded paths ==
        #TODO: add the remove unseen paths

        if self.linear_after_env_embed:
            self.env_linears = Linear(
                mul_irreps_sh,
                mul_irreps_sh,
                shared_weights=True,
                internal_weights=True,
            )
        else:
            self.env_linears = torch.nn.Identity()

        # Make TP
        tmp_i_out: int = 0
        instr = []
        n_scalar_outs: int = 0
        n_scalar_mul = []
        full_out_irreps = []
        for i_out, (mul_out, ir_out) in enumerate(self.irreps_out):
            for i_1, (mul1, ir_1) in enumerate(self.irreps_in): # what if feature_irreps_in has mul?
                for i_2, (mul2, ir_2) in enumerate(self._env_weighter.irreps_out):
                    if ir_out in ir_1 * ir_2:
                        if ir_out == SCALAR:
                            n_scalar_outs += 1
                            n_scalar_mul.append(mul2)
                        # assert mul_out == mul1 == mul2
                        instr.append((i_1, i_2, tmp_i_out, 'uvv', True))
                        full_out_irreps.append((mul2, ir_out))
                        assert full_out_irreps[-1][0] == mul2
                        tmp_i_out += 1
        full_out_irreps = o3.Irreps(full_out_irreps)
        assert all(ir == SCALAR for _, ir in full_out_irreps[:n_scalar_outs])
        self.n_scalar_mul = sum(n_scalar_mul)

        self.tp = TensorProduct(
                irreps_in1=o3.Irreps(
                    [(mul, ir) for mul, ir in self.irreps_in]
                ),
                irreps_in2=o3.Irreps(
                    [(mul, ir) for mul, ir in self._env_weighter.irreps_out]
                ),
                irreps_out=o3.Irreps(
                    [(mul, ir) for mul, ir in full_out_irreps]
                ),
                instructions=instr,
                shared_weights=True,
                internal_weights=True,
            )
        
        # build activation
        irreps_scalar = o3.Irreps(str(self.irreps_out[0]))
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0,1)) for mul, _ in irreps_gated]).simplify()
        act={1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates={1: torch.sigmoid, -1: torch.tanh}

        # self.activation = Gate(
        #     irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
        #     irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
        #     irreps_gated  # gated tensors
        # )
        
        # we extract the scalars from the first irrep of the tp
        assert self.irreps_out[0].ir == SCALAR
        self.linears = Linear(
                irreps_in=full_out_irreps,
                irreps_out=irreps_out,
                shared_weights=True,
                internal_weights=True,
            )
        
        # the embedded latent invariants from the previous layer(s)
        # and the invariants extracted from the last layer's TP:
        self.latents = latent(
            mlp_input_dimension=latent_in+self.n_scalar_mul,
            mlp_output_dimension=None,
        )
        
        # the env embed MLP takes the last latent's output as input
        # and outputs enough weights for the env embedder
        self.env_embed_mlps = ScalarMLPFunction(
                mlp_input_dimension=latent_in,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self._env_weighter.weight_numel,
            )
        # - layer resnet update weights -
        if latent_resnet_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            latent_resnet_update_params = torch.zeros(1)
        else:
            latent_resnet_update_ratios = torch.as_tensor(
                latent_resnet_update_ratios, dtype=torch.get_default_dtype()
            )
            assert latent_resnet_update_ratios > 0.0
            assert latent_resnet_update_ratios < 1.0
            latent_resnet_update_params = torch.special.logit(
                latent_resnet_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            latent_resnet_update_params.clamp_(-6.0, 6.0)
        
        if latent_resnet_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                latent_resnet_update_params
            )
        else:
            self.register_buffer(
                "_latent_resnet_update_params", latent_resnet_update_params
            )

    def forward(self, edge_index, edge_sh, atom_type, latents, features, cutoff_coeffs, active_edges):
        # update V
        # update X
        # edge_index: [2, num_edges]
        # irreps_sh: [num_edges, irreps_sh]
        # latents: [num_edges, latent_in]
        # fetures: [num_active_edges, in_irreps]
        # cutoff_coeffs: [num_edges]
        # active_edges: [num_active_edges]

        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        prev_mask = cutoff_coeffs > 0

        # update V
        weights = self.env_embed_mlps(latents[active_edges])

        # Build the local environments
        # This local environment should only be a sum over neighbors
        # who are within the cutoff of the _current_ layer
        # Those are the active edges, which are the only ones we
        # have weights for (env_w) anyway.
        # So we mask out the edges in the sum:
        local_env_per_edge = scatter(
            self._env_weighter(edge_sh[active_edges], weights),
            edge_center[active_edges],
            dim=0,
        )

        # currently, we have a sum over neighbors of constant number for each layer,
        # the env_sum_normalization can be a scalar or list
        # the different cutoff can be added in the future
        
        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        
        local_env_per_edge = local_env_per_edge * norm_const
        local_env_per_edge = self.env_linears(local_env_per_edge)
        
        local_env_per_edge = local_env_per_edge[edge_center[active_edges]]
        
        # Now do the TP
        # recursively tp current features with the environment embeddings
        new_features = self.tp(features, local_env_per_edge) # full_out_irreps
        
        
        # features has shape [N_edge, full_feature_out.dim]
        # we know scalars are first
        scalars = new_features[:, :self.n_scalar_mul]
        assert len(scalars.shape) == 2

        # do the linear
        new_features = self.linears(new_features)
        # new_features = self.activation(new_features)

        if self.latent_resnet:
            update_coefficients = self._latent_resnet_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            features = coefficient_new * new_features + coefficient_old * features
        else:
            features = new_features 

        # update X
        latent_inputs_to_cat = [
                latents[active_edges],
                scalars,
            ]
        
        new_latents = self.latents(torch.cat(latent_inputs_to_cat, dim=-1))
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        # At init, we assume new and old to be approximately uncorrelated
        # Thus their variances add
        # we always want the latent space to be normalized to variance = 1.0,
        # because it is critical for learnability. Still, we want to preserve
        # the _relative_ magnitudes of the current latent and the residual update
        # to be controled by `this_layer_update_coeff`
        # Solving the simple system for the two coefficients:
        #   a^2 + b^2 = 1  (variances add)   &    a * this_layer_update_coeff = b
        # gives
        #   a = 1 / sqrt(1 + this_layer_update_coeff^2)  &  b = this_layer_update_coeff / sqrt(1 + this_layer_update_coeff^2)
        # rsqrt is reciprocal sqrt
        if self.latent_resnet:
            update_coefficients = self._latent_resnet_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            latents = torch.index_add(
                coefficient_old * latents,
                0,
                active_edges,
                coefficient_new * new_latents,
            )
        else:
            latents = torch.index_copy(latents, 0, active_edges, new_latents)
        
        return latents, features, cutoff_coeffs, active_edges
        
        