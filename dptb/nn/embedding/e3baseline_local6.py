from typing import Optional, List, Union, Dict
import math
import functools
import torch
from torch_runstats.scatter import scatter
from torch import fx
from e3nn.util.codegen import CodeGenMixin
from e3nn import o3
from e3nn.nn import Gate
from torch_scatter import scatter_mean
from e3nn.o3 import Linear, SphericalHarmonics
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from dptb.nn.embedding.from_deephe3.deephe3 import tp_path_exists
from dptb.data import _keys
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product import SO2_Linear
import math
from dptb.data.transforms import OrbitalMapper
import torch_geometric
from ..type_encode.one_hot import OneHotAtomEncoding
from dptb.nn.norm import SeperableLayerNorm
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch

from math import ceil

@Embedding.register("e3baseline_6")
class E3BaseLineModel6(torch.nn.Module):
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
            env_embed_multiplicity: int = 32,
            sh_normalized: bool = True,
            sh_normalization: str = "component",
            # tp parameters:
            tp_radial_emb: bool=False,
            tp_radial_channels: list=[128, 128],
            # MLP parameters:
            latent_channels: list=[128, 128],
            latent_dim: int=128,
            res_update: bool = True,
            res_update_ratios: Optional[List[float]] = None,
            res_update_ratios_learnable: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            ):
        
        super(E3BaseLineModel6, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        latent_kwargs={
                "mlp_latent_dimensions": latent_channels+[latent_dim],
                "mlp_nonlinearity": "silu",
                "mlp_initialization": "uniform"
            },
            
        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        self.n_atom = n_atom

        irreps_sh=o3.Irreps([(1, (i, (-1) ** i)) for i in range(lmax + 1)])
        orbpair_irreps = self.idp.orbpair_irreps.sort()[0].simplify()

        # check if the irreps setting satisfied the requirement of idp
        irreps_out = []
        for mul, ir1 in irreps_hidden:
            for _, ir2 in orbpair_irreps:
                irreps_out += [o3.Irrep(str(irr)) for irr in ir1*ir2]
        irreps_out = o3.Irreps(irreps_out).sort()[0].simplify()

        assert all(ir in irreps_out for _, ir in orbpair_irreps), "hidden irreps should at least cover all the reqired irreps in the hamiltonian data {}".format(orbpair_irreps)
        
        # TODO: check if the tp in first layer can produce the required irreps for hidden states

        self.sh = SphericalHarmonics(
            irreps_sh, sh_normalized, sh_normalization
        )
        self.onehot = OneHotAtomEncoding(num_types=n_atom, set_features=False)

        self.init_layer = InitLayer(
            idp=self.idp,
            num_types=n_atom,
            n_radial_basis=n_radial_basis,
            r_max=r_max,
            irreps_sh=irreps_sh,
            avg_num_neighbors=avg_num_neighbors,
            env_embed_multiplicity=env_embed_multiplicity,
            # MLP parameters:
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            # cutoffs
            r_start_cos_ratio=r_start_cos_ratio,
            PolynomialCutoff_p=PolynomialCutoff_p,
            cutoff_type=cutoff_type,
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList()
        # actually, we can derive the least required irreps_in and out from the idp's node and pair irreps
        last_layer = False
        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
                irreps_hidden_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden
                irreps_hidden_in = irreps_hidden
            
            if i == n_layers - 1:
                irreps_out = orbpair_irreps.sort()[0].simplify()
            else:
                irreps_out = irreps_hidden

            self.layers.append(Layer(
                num_types=n_atom,
                # required params
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                irreps_hidden_in=irreps_hidden_in,
                irreps_hidden_out=irreps_hidden,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                # MLP parameters:
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                dtype=dtype,
                device=device,
                )
            )

        # initilize output_layer
        self.out_edge = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True, internal_weights=True, biases=True)
        self.out_node = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True, internal_weights=True, biases=True)

    @property
    def out_edge_irreps(self):
        return self.idp.orbpair_irreps

    @property
    def out_node_irreps(self):
        return self.idp.orbpair_irreps
    
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)
        # data = with_env_vectors(data, with_lengths=True)
        data = with_batch(data)

        edge_index = data[_keys.EDGE_INDEX_KEY]
        edge_vector = data[_keys.EDGE_VECTORS_KEY]
        edge_sh = self.sh(data[_keys.EDGE_VECTORS_KEY][:,[1,2,0]])
        edge_length = data[_keys.EDGE_LENGTH_KEY]

        data = self.onehot(data)
        node_one_hot = data[_keys.NODE_ATTRS_KEY]
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        latents, node_features, edge_features, hidden_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type, bond_type, edge_sh, edge_length, node_one_hot)
        for layer in self.layers:
            hidden_features, node_features, edge_features = \
                layer(
                    latents, 
                    node_features, 
                    hidden_features, 
                    edge_features, 
                    node_one_hot, 
                    edge_index, 
                    edge_vector, 
                    atom_type, 
                    cutoff_coeffs, 
                    active_edges
                )

        data[_keys.NODE_FEATURES_KEY] = self.out_node(node_features)
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype, device=self.device)
        data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(data[_keys.EDGE_FEATURES_KEY], 0, active_edges, self.out_edge(edge_features))

        return data
    
@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
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
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: float,
            avg_num_neighbors: Optional[float] = None,
            irreps_sh: o3.Irreps=None,
            env_embed_multiplicity: int = 32,
            # MLP parameters:
            two_body_latent_channels: list=[128, 128],
            latent_dim: int=128,
            # cutoffs
            r_start_cos_ratio: float = 0.8,
            PolynomialCutoff_p: float = 6,
            cutoff_type: str = "polynomial",
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super(InitLayer, self).__init__()
        SCALAR = o3.Irrep("0e")
        self.num_types = num_types
        if isinstance(r_max, float) or isinstance(r_max, int):
            self.r_max = torch.tensor(r_max, device=device, dtype=dtype)
            self.r_max_dict = None
        elif isinstance(r_max, dict):
            c_set = set(list(r_max.values()))
            self.r_max = torch.tensor(max(list(r_max.values())), device=device, dtype=dtype)
            if len(r_max) == 1 or len(c_set) == 1:
                self.r_max_dict = None
            else:
                self.r_max_dict = {}
                for k,v in r_max.items():
                    self.r_max_dict[k] = torch.tensor(v, device=device, dtype=dtype)
        else:
            raise TypeError("r_max should be either float, int or dict")
                  
        self.idp = idp
        self.r_start_cos_ratio = r_start_cos_ratio
        self.polynomial_cutoff_p = PolynomialCutoff_p
        self.cutoff_type = cutoff_type
        self.device = device
        self.dtype = dtype
        self.irreps_out = o3.Irreps([(env_embed_multiplicity, ir) for _, ir in irreps_sh])

        assert all(mul==1 for mul, _ in irreps_sh)
        # env_embed_irreps = o3.Irreps([(1, ir) for _, ir in irreps_sh])
        assert (
            irreps_sh[0].ir == SCALAR
        ), "env_embed_irreps must start with scalars"

        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )

        # Node invariants for center and neighbor (chemistry)
        # Plus edge invariants for the edge (radius).
        self.two_body_latent = ScalarMLPFunction(
                        mlp_input_dimension=(2 * num_types + n_radial_basis),
                        mlp_output_dimension=latent_dim,
                        mlp_latent_dimensions=two_body_latent_channels,
                        mlp_nonlinearity="silu",
                        mlp_initialization="uniform",
                    )

        self.sln_n = SeperableLayerNorm(
            irreps=self.irreps_out,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_h = SeperableLayerNorm(
            irreps=self.irreps_out,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        self._env_weighter = Linear(
            irreps_in=irreps_sh,
            irreps_out=self.irreps_out,
            internal_weights=False,
            shared_weights=False,
            path_normalization = "element", # if path normalization is element and input irreps has 1 mul, it should not have effect ! 
        )

        self.env_embed_mlp = ScalarMLPFunction(
                        mlp_input_dimension=self.two_body_latent.out_features,
                        mlp_output_dimension=self._env_weighter.weight_numel,
                        mlp_latent_dimensions=[],
                        mlp_nonlinearity=None,
                        mlp_initialization="uniform",
                    )
        
        self.bessel = BesselBasis(r_max=self.r_max, num_basis=n_radial_basis, trainable=True)


    def forward(self, edge_index, atom_type, bond_type, edge_sh, edge_length, node_one_hot):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        edge_invariants = self.bessel(edge_length)
        node_invariants = node_one_hot

        # Vectorized precompute per layer cutoffs
        if self.r_max_dict is None:
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
        else:
            cutoff_coeffs = torch.zeros(edge_index.shape[1], dtype=self.dtype, device=self.device)

            for bond, ty in self.idp.bond_to_type.items():
                mask = bond_type == ty
                index = mask.nonzero().squeeze(-1)

                if mask.any():
                    iatom, jatom = bond.split("-")
                    if self.cutoff_type == "cosine":
                        c_coeff = cosine_cutoff(
                            edge_length[mask],
                            0.5*(self.r_max_dict[iatom]+self.r_max_dict[jatom]),
                            r_start_cos_ratio=self.r_start_cos_ratio,
                        ).flatten()
                    elif self.cutoff_type == "polynomial":
                        c_coeff = polynomial_cutoff(
                            edge_length[mask],
                            0.5*(self.r_max_dict[iatom]+self.r_max_dict[jatom]),
                            p=self.polynomial_cutoff_p
                        ).flatten()

                    else:
                        # This branch is unreachable (cutoff type is checked in __init__)
                        # But TorchScript doesn't know that, so we need to make it explicitly
                        # impossible to make it past so it doesn't throw
                        # "cutoff_coeffs_all is not defined in the false branch"
                        assert False, "Invalid cutoff type"

                    cutoff_coeffs = torch.index_copy(cutoff_coeffs, 0, index, c_coeff)

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
        latents = torch.index_copy(
            latents, 0, active_edges, 
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
            )
        
        weights_h = self.env_embed_mlp(new_latents)
        weights_e = self.env_embed_mlp(latents)

        # embed initial edge
        hidden_features = self._env_weighter(
            edge_sh[prev_mask], weights_h
        )  # features is edge_attr
        # features = self.bn(features)

        edge_features = self._env_weighter(
            edge_sh[prev_mask], weights_e
        )

        node_features = scatter(
            edge_features,
            edge_center[active_edges],
            dim=0,
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        
        node_features = node_features * norm_const

        node_features = self.sln_n(node_features)
        hidden_features = self.sln_h(hidden_features)

        return latents, node_features, edge_features, hidden_features, cutoff_coeffs, active_edges # the radial embedding x and the sperical hidden V

class UpdateNode(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        avg_num_neighbors: Optional[float] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(UpdateNode, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_hidden = irreps_hidden
        self.dtype = dtype
        self.device = device
        self.res_update = res_update

        self.register_buffer(
            "env_sum_normalizations",
            # dividing by sqrt(N)
            torch.as_tensor(avg_num_neighbors).rsqrt(),
        )
        

        self._env_weighter = E3ElementLinear(
            irreps_in=irreps_out,
            dtype=dtype,
            device=device,
        )

        # self.sln_n = SeperableLayerNorm(
        #     irreps=self.irreps_in,
        #     eps=1e-5, 
        #     affine=True, 
        #     normalization='component', 
        #     std_balance_degrees=True,
        #     dtype=self.dtype,
        #     device=self.device
        # )

        self.sln = SeperableLayerNorm(
            irreps=self.irreps_in,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        # self.sln_h = SeperableLayerNorm(
        #     irreps=self.irreps_hidden,
        #     eps=1e-5, 
        #     affine=True, 
        #     normalization='component', 
        #     std_balance_degrees=True,
        #     dtype=self.dtype,
        #     device=self.device
        # )

        assert irreps_out[0].ir.l == 0

        # here we adopt the graph attention's idea to generate the weights as the attention scores
        # self.latent_act = torch.nn.LeakyReLU()
        self.env_embed_mlps = ScalarMLPFunction(
                mlp_input_dimension=latent_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self._env_weighter.weight_numel,
            )
        
        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0,1)) for mul, _ in irreps_gated]).simplify()
        act={1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates={1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in+self.irreps_hidden,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True, 
            internal_weights=True,
            biases=True,
        )

        if res_update:
            self.linear_res = Linear(
                self.irreps_in,
                self.irreps_out,
                shared_weights=True, 
                internal_weights=True,
                biases=True,
            )

        # - layer resnet update weights -
        if res_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            res_update_params.clamp_(-6.0, 6.0)
        
        if res_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )

    def forward(self, latents, node_features, hidden_features, atom_type, node_onehot, edge_index, edge_vector, active_edges):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = self.sln(node_features)
        message = self.tp(
            torch.cat(
                [new_node_features[edge_center[active_edges]], hidden_features]
                , dim=-1), edge_vector[active_edges], latents) # full_out_irreps
        
        message = self.activation(message)
        message = self.lin_post(message)
        scalars = message[:, :self.irreps_out[0].dim]

        # get the attention scores
        # weights = self.env_embed_mlps(self.latent_act(latents[active_edges]))
        # weights = torch_geometric.utils.softmax(weights, edge_center[active_edges], num_nodes=node_features.shape[0])
        weights = self.env_embed_mlps(latents[active_edges])
        new_node_features = scatter(
            self._env_weighter(message, weights),
            edge_center[active_edges],
            dim=0,
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        assert len(scalars.shape) == 2

        new_node_features = new_node_features * norm_const

        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            node_features = coefficient_new * new_node_features + coefficient_old * self.linear_res(node_features)
        else:
            node_features = new_node_features

        return node_features
    
class UpdateEdge(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_hidden: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(UpdateEdge, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_hidden = irreps_hidden
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        
        self._edge_weighter = E3ElementLinear(
                irreps_in=irreps_out,
                dtype=dtype,
                device=device,
            )

        self.edge_embed_mlps = ScalarMLPFunction(
                mlp_input_dimension=latent_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self._edge_weighter.weight_numel,
            )
        
        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0,1)) for mul, _ in irreps_gated]).simplify()
        act={1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates={1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        # self.sln_n = SeperableLayerNorm(
        #     irreps=self.irreps_in,
        #     eps=1e-5, 
        #     affine=True, 
        #     normalization='component', 
        #     std_balance_degrees=True,
        #     dtype=self.dtype,
        #     device=self.device
        # )

        # self.sln_h = SeperableLayerNorm(
        #     irreps=self.irreps_hidden,
        #     eps=1e-5, 
        #     affine=True, 
        #     normalization='component', 
        #     std_balance_degrees=True,
        #     dtype=self.dtype,
        #     device=self.device
        # )

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in+self.irreps_hidden+self.irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
        )

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_out,
            shared_weights=True, 
            internal_weights=True,
            biases=True,
        )

        if res_update:
            self.linear_res = Linear(
                self.irreps_in,
                self.irreps_out,
                shared_weights=True, 
                internal_weights=True,
                biases=True,
            )

        # - layer resnet update weights -
        if res_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            res_update_params.clamp_(-6.0, 6.0)
        
        if res_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )
    
    def forward(self, latents, node_features, hidden_features, edge_features, edge_index, edge_vector, active_edges):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_edge_features = self.tp(
            torch.cat(
                [
                    node_features[edge_center[active_edges]],
                    hidden_features,
                    node_features[edge_neighbor[active_edges]]
                    ]
                , dim=-1), edge_vector[active_edges], latents) # full_out_irreps
        
        scalars = new_edge_features[:, :self.tp.irreps_out[0].dim]
        assert len(scalars.shape) == 2
        new_edge_features = self.activation(new_edge_features)
        new_edge_features = self.lin_post(new_edge_features)

        weights = self.edge_embed_mlps(latents[active_edges])
        new_edge_features = self._edge_weighter(new_edge_features, weights)
        
        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            edge_features = coefficient_new * new_edge_features + coefficient_old * self.linear_res(edge_features)
        else:
            edge_features = new_edge_features

        return edge_features
    
class UpdateHidden(torch.nn.Module):
    def __init__(
        self,
        num_types: int,
        irreps_fea: o3.Irreps,
        irreps_hidden_in: o3.Irreps,
        irreps_hidden_out: o3.Irreps,
        latent_dim: int,
        latent_channels: list=[128, 128],
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        
        super(UpdateHidden, self).__init__()
        self.irreps_fea = irreps_fea
        self.irreps_hidden_in = irreps_hidden_in
        self.irreps_hidden_out = irreps_hidden_out
        self.dtype = dtype
        self.device = device
        self.res_update = res_update
        
        irreps_scalar = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden_out if ir.l == 0]).simplify()
        irreps_gated = o3.Irreps([(mul, ir) for mul, ir in self.irreps_hidden_out if ir.l > 0]).simplify()
        irreps_gates = o3.Irreps([(mul, (0,1)) for mul, _ in irreps_gated]).simplify()
        act={1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates={1: torch.sigmoid, -1: torch.tanh}

        self.activation = Gate(
            irreps_scalar, [act[ir.p] for _, ir in irreps_scalar],  # scalar
            irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
            irreps_gated  # gated tensors
        )

        self.ln = torch.nn.LayerNorm(latent_dim)
        self.ln_o = torch.nn.LayerNorm(latent_dim)

        self.sln = SeperableLayerNorm(
            irreps=self.irreps_hidden_in,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        self.tp = SO2_Linear(
            irreps_in=self.irreps_fea+self.irreps_hidden_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
        )

        all_tp_scalar = o3.Irreps([(mul, ir) for mul, ir in self.tp.irreps_out if ir.l == 0]).simplify()
        assert all_tp_scalar.dim == self.tp.irreps_out[0].dim

        self.lin_post = Linear(
            self.activation.irreps_out,
            self.irreps_hidden_out,
            shared_weights=True, 
            internal_weights=True,
            biases=True,
        )

        if res_update:
            self.linear_res = Linear(
                self.irreps_hidden_in,
                self.irreps_hidden_out,
                shared_weights=True, 
                internal_weights=True,
                biases=True,
            )

        self.latents = ScalarMLPFunction(
            mlp_input_dimension=latent_dim+self.irreps_hidden_out[0].dim+2*num_types,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self._hid_weighter = E3ElementLinear(
                irreps_in=irreps_hidden_out,
                dtype=dtype,
                device=device,
            )

        self.hid_embed_mlps = ScalarMLPFunction(
                mlp_input_dimension=latent_dim,
                mlp_latent_dimensions=[],
                mlp_output_dimension=self._hid_weighter.weight_numel,
            )
        
        

        # - layer resnet update weights -
        if res_update_ratios is None:
            # We initialize to zeros, which under the sigmoid() become 0.5
            # so 1/2 * layer_1 + 1/4 * layer_2 + ...
            # note that the sigmoid of these are the factor _between_ layers
            # so the first entry is the ratio for the latent resnet of the first and second layers, etc.
            # e.g. if there are 3 layers, there are 2 ratios: l1:l2, l2:l3
            res_update_params = torch.zeros(1)
        else:
            res_update_ratios = torch.as_tensor(
                res_update_ratios, dtype=torch.get_default_dtype()
            )
            assert res_update_ratios > 0.0
            assert res_update_ratios < 1.0
            res_update_params = torch.special.logit(
                res_update_ratios
            )
            # The sigmoid is mostly saturated at ±6, keep it in a reasonable range
            res_update_params.clamp_(-6.0, 6.0)
        
        if res_update_ratios_learnable:
            self._latent_resnet_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )

    def forward(self, latents, node_features, hidden_features, node_onehot, edge_index, edge_vector, cutoff_coeffs, active_edges):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_hidden_features = self.sln(hidden_features)
        new_hidden_features = self.tp(
            torch.cat(
                [
                    node_features[edge_center[active_edges]],
                    new_hidden_features
                    ]
                , dim=-1), edge_vector[active_edges], latents)
        
        
        new_hidden_features = self.activation(new_hidden_features)
        new_hidden_features = self.lin_post(new_hidden_features)

        scalars = new_hidden_features[:, :self.irreps_hidden_out[0].dim]
        assert len(scalars.shape) == 2

        weights = self.hid_embed_mlps(self.ln(latents[active_edges]))
        new_hidden_features = self._hid_weighter(new_hidden_features, weights)
        
        # update latent
        latent_inputs_to_cat = [
            node_onehot[edge_center[active_edges]],
            self.ln_o(latents[active_edges]),
            scalars,
            node_onehot[edge_neighbor[active_edges]],
        ]
        
        new_latents = self.latents(torch.cat(latent_inputs_to_cat, dim=-1))
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents

        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            hidden_features = coefficient_new * new_hidden_features + coefficient_old * self.linear_res(hidden_features)
            latents = coefficient_new * new_latents + coefficient_old * latents
        else:
            hidden_features = new_hidden_features
            latents = new_latents
        
        return hidden_features, latents

class Layer(torch.nn.Module):
    def __init__(
        self,
        num_types: int,
        # required params
        avg_num_neighbors: Optional[float] = None,
        irreps_in: o3.Irreps=None,
        irreps_out: o3.Irreps=None,
        irreps_hidden_in: o3.Irreps=None,
        irreps_hidden_out: o3.Irreps=None,
        tp_radial_emb: bool=False,
        tp_radial_channels: list=[128, 128],
        # MLP parameters:
        latent_channels: list=[128, 128],
        latent_dim: int=128,
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Layer, self).__init__()

        self.res_update = res_update
        self.avg_num_neighbors = avg_num_neighbors
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.irreps_hidden_in = irreps_hidden_in
        self.irreps_hidden_out = irreps_hidden_out
        self.dtype = dtype
        self.device = device
        self.num_types = num_types

        # 1. update hidden
        # 2. update edge
        # 3. update node

        self.hidden_update = UpdateHidden(
            num_types=num_types,
            irreps_fea=self.irreps_in,
            irreps_hidden_in=irreps_hidden_in,
            irreps_hidden_out=irreps_hidden_out,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            dtype=dtype,
            device=device,
        )

        self.edge_update = UpdateEdge(
            irreps_in=self.irreps_in,
            irreps_hidden=self.irreps_hidden_out,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            dtype=dtype,
            device=device,
        )

        self.node_update = UpdateNode(
            irreps_in=self.irreps_in,
            irreps_hidden=self.irreps_hidden_out,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            avg_num_neighbors=avg_num_neighbors,
            dtype=dtype,
            device=device,
        )

        self.sln_n = SeperableLayerNorm(
            irreps=self.irreps_in,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_h = SeperableLayerNorm(
            irreps=self.irreps_hidden_out,
            eps=1e-5, 
            affine=True, 
            normalization='component', 
            std_balance_degrees=True,
            dtype=self.dtype,
            device=self.device
        )

    def forward(self, latents, node_features, hidden_features, edge_features, node_onehot, edge_index, edge_vector, atom_type, cutoff_coeffs, active_edges):
        
        n_node_features = self.sln_n(node_features)
        hidden_features, latents = self.hidden_update(latents, n_node_features, hidden_features, node_onehot, edge_index, edge_vector, cutoff_coeffs, active_edges)

        n_hidden_features = self.sln_h(hidden_features)
        edge_features = self.edge_update(latents, n_node_features, n_hidden_features, edge_features, edge_index, edge_vector, active_edges)

        node_features = self.node_update(latents, node_features, n_hidden_features, atom_type, node_onehot, edge_index, edge_vector, active_edges)

        return hidden_features, node_features, edge_features
    