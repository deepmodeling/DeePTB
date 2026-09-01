from typing import Optional, List, Union, Dict
import math
import functools
import torch
from torch_runstats.scatter import scatter
from torch import fx
from e3nn import o3
from e3nn.nn import Gate
from torch_scatter import scatter_mean
from e3nn.o3 import Linear, SphericalHarmonics
from e3nn.math import normalize2mom
from e3nn.util.jit import compile_mode
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding
from ..radial_basis import BesselBasis
from ..base import ScalarMLPFunction
from dptb.data import _keys
from dptb.nn.cutoff import cosine_cutoff, polynomial_cutoff
from dptb.nn.rescale import E3ElementLinear
from dptb.nn.tensor_product import SO2_Linear, tp_path_exists
import math
from dptb.data.transforms import OrbitalMapper
from ..type_encode.one_hot import OneHotAtomEncoding
from dptb.nn.norm import SeperableLayerNorm, PerLGain
from dptb.data.AtomicDataDict import with_edge_vectors, with_batch
from dptb.nn.threecenter import ThreeCenterFactorized, EDGE_THREECENTER_KEY, NODE_THREECENTER_KEY
from dptb.utils.distributed import (
    column_atom_index,
    node_column_values,
    reverse_edge_values,
)
from math import ceil

@Embedding.register("trinity")
class Trinity(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            # required params
            mode: str="full",
            n_layers: int=3,
            n_radial_basis: int=10,
            r_max: float=5.0,
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
            universal: Optional[bool] = False,
            three_center: Optional[dict] = None,
            freeze: Optional[Union[str, List[str]]] = None,
            so2_gate: bool = False,
            hermitian: bool = True,
            spectral_balance: bool = False,
            **kwargs,
            ):

        super(Trinity, self).__init__()

        irreps_hidden = o3.Irreps(irreps_hidden)
        lmax = irreps_hidden.lmax
        

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
        self.latent_dim = latent_dim
        # a single `mode` (Trinity-only; replaces only2b/exclusive) selects the output channels.
        # Following the original convention the two-body base is *always* produced; the mode toggles
        # the extra channels on top of it:
        #   "2b"     : two-body base only
        #   "3b"/"2b+3b" : add the three-body term
        #   "full"   : add three-body + env (message passing)
        assert mode in ("2b", "3b", "2b+3b", "full"), f"unknown trinity mode {mode!r}"
        self.mode = mode
        self.use_3b = mode in ("3b", "2b+3b", "full")
        self.use_env = mode == "full"

        # `freeze` selectively holds trainable blocks fixed (requires_grad=False), ORTHOGONAL to `mode`
        # (which only controls which channels are *applied* in forward). Any subset of:
        #   "2b"  : two-body SK term (two_ness)
        #   "3b"  : three-center term (three_center, if configured)
        #   "env" : many-body message-passing pathway (init_layer, layers, out_edge, out_node)
        # This enables progressive training, e.g. train 2b, then freeze ["2b"] and train 3b, then
        # freeze ["2b","3b"] and train env. If unset (None), keep the historical default: "full"
        # freezes 2b+3b (train only env), other modes freeze nothing; pass freeze=[] to train all.
        _FREEZE_CHOICES = ("2b", "3b", "env")
        if freeze is None:
            freeze = ["2b", "3b"] if mode == "full" else []
        elif isinstance(freeze, str):
            freeze = [freeze]
        self.freeze = set(freeze)
        assert self.freeze <= set(_FREEZE_CHOICES), \
            f"trinity `freeze` must be a subset of {_FREEZE_CHOICES}, got {sorted(self.freeze)}"
            
        self.basis = self.idp.basis
        self.idp.get_irreps(no_parity=False)
        if universal:
            self.n_atom = 95
        else:
            self.n_atom = len(self.basis.keys())

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

        self.onehot = OneHotAtomEncoding(num_types=self.n_atom, set_features=False, idp=self.idp, universal=universal)

        self.init_layer = InitLayer(
            idp=self.idp,
            num_types=self.n_atom,
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
            spectral_balance=spectral_balance,
            device=device,
            dtype=dtype,
        )

        self.two_ness = Twoness(
            idp=self.idp,
            num_types=self.n_atom,
            r_max=r_max,
            n_radial_basis=n_radial_basis,
            # MLP parameters:
            two_body_latent_channels=latent_channels,
            latent_dim=latent_dim,
            # cutoffs
            device=device,
            dtype=dtype,
        )

        self.layers = torch.nn.ModuleList()
        # actually, we can derive the least required irreps_in and out from the idp's node and pair irreps
        last_layer = False
        for i in range(n_layers):
            if i == 0:
                irreps_in = self.init_layer.irreps_out
            else:
                irreps_in = irreps_hidden
            
            if i == n_layers - 1:
                irreps_out = orbpair_irreps.sort()[0].simplify()
            else:
                irreps_out = irreps_hidden

            self.layers.append(Layer(
                num_types=self.n_atom,
                # required params
                avg_num_neighbors=avg_num_neighbors,
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                tp_radial_emb=tp_radial_emb,
                tp_radial_channels=tp_radial_channels,
                # MLP parameters:
                latent_channels=latent_channels,
                latent_dim=latent_dim,
                res_update=res_update,
                res_update_ratios=res_update_ratios,
                res_update_ratios_learnable=res_update_ratios_learnable,
                so2_gate=so2_gate,
                spectral_balance=spectral_balance,
                dtype=dtype,
                device=device,
                )
            )

        # Wigner-D cache: every SO2_Linear in every layer rotates with the SAME per-edge Wigner
        # matrices (they all act on the shared active-edge set); precompute them once per forward
        # and pass down (2 x n_layers redundant computations saved).
        self._wigner_lmax = max(irreps_hidden.lmax, orbpair_irreps.lmax)

        # Hermitian symmetrization of the env edge channel. LCAO H is hermitian: for the reversed
        # edge (j,i,-S) the reduced coefficients of the SAME-shell-pair (io==jo) channels satisfy
        # r_ji = (-1)^J r_ij exactly (verified numerically against E3Hamiltonian). The two-body SK
        # and three-center terms satisfy this by construction; the env (message-passing) output does
        # not, so we enforce it by averaging: r <- (r + (-1)^J r[rev])/2. Channels with io != jo
        # carry independent physics on the two directions (only io<=jo pairs are stored per directed
        # edge) and are left untouched. Zero-parameter, exact constraint -> halves the function
        # space in the constrained subspace. Buffers are non-persistent (derived from idp).
        self.hermitian = hermitian
        herm_mask = torch.zeros(self.idp.reduced_matrix_element, dtype=torch.bool)
        herm_sign = []
        from dptb.utils.constants import anglrMId
        self.idp.get_orbpair_maps()
        for pair, sl in self.idp.orbpair_maps.items():
            io, jo = pair.split("-")
            if io != jo:
                continue
            l = anglrMId[io[1]]
            off = sl.start
            for J in range(0, 2 * l + 1):
                d = 2 * J + 1
                herm_mask[off:off + d] = True
                herm_sign += [float((-1) ** J)] * d
                off += d
        self.register_buffer("herm_mask", herm_mask, persistent=False)
        self.register_buffer("herm_sign", torch.tensor(herm_sign, dtype=self.dtype), persistent=False)

        # initilize output_layer
        self.out_edge = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True, internal_weights=True, biases=True)
        self.out_node = Linear(self.layers[-1].irreps_out, self.idp.orbpair_irreps, shared_weights=True, internal_weights=True, biases=True)

        # additive three-center factorized term H^(3) = sum_C P_AC D_C P_CB, owned entirely by Trinity.
        # The module (and hence its parameters) is built whenever a `three_center` config is given,
        # *independent of* `mode`, so a checkpoint always carries the 2b+3b parameters and one can train
        # progressively 2b -> 2b+3b -> full by reloading and only switching `mode` (the state_dict is
        # identical across modes). `mode` (via use_3b) merely gates whether the term is *applied* in
        # forward, and `freeze` whether it trains. Its blocks are added in reduced-equivariant space
        # (see to_reduced) so the existing NNENV transform picks them up.
        self.three_center = None
        if three_center is not None:
            self.three_center = ThreeCenterFactorized(
                basis=self.basis,
                projectors=three_center["projectors"],
                idp=self.idp,
                er_max=three_center.get("er_max", 5.0),
                n_radial_basis=three_center.get("n_radial_basis", 8),
                latent_channels=three_center.get("latent_channels", [64, 64]),
                coupling_mode=three_center.get("coupling", "block_diag"),
                dtype=dtype,
                device=device,
            )
        else:
            assert not self.use_3b, \
                f"trinity mode {mode!r} needs a `three_center` config (projectors, er_max, ...)."

        # apply the selective freezing now that every sub-module exists. `mode` still governs which
        # channels are applied in forward; this only sets requires_grad on the chosen blocks.
        def _freeze(module):
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
        if "2b" in self.freeze:
            _freeze(self.two_ness)
        if "3b" in self.freeze:
            _freeze(self.three_center)
        if "env" in self.freeze:
            for m in [self.init_layer, self.out_edge, self.out_node, *self.layers]:
                _freeze(m)

    @property
    def out_edge_irreps(self):
        return self.idp.orbpair_irreps

    @property
    def out_node_irreps(self):
        return self.idp.orbpair_irreps

    def _reversed_edges(self, data: AtomicDataDict.Type) -> torch.Tensor:
        """Return local reverse-edge indices for diagnostics and compatibility.

        A reversed edge owned by another rank has no local row index and maps
        to itself here.  Distributed forward evaluation must use
        :func:`reverse_edge_values`, as it does below.
        """
        edge = data[_keys.EDGE_INDEX_KEY]
        shift = data.get(_keys.EDGE_CELL_SHIFT_KEY)
        if shift is None:
            shift = torch.zeros((edge.shape[1], 3), dtype=torch.long, device=edge.device)
        else:
            shift = shift.round().long()
        atom_ids = column_atom_index(data).to(edge.device)
        entries = {}
        for index in range(edge.shape[1]):
            key = (
                int(atom_ids[edge[0, index]]), int(atom_ids[edge[1, index]]),
                int(shift[index, 0]), int(shift[index, 1]), int(shift[index, 2]),
            )
            entries[key] = index
        reversed_indices = []
        for index in range(edge.shape[1]):
            key = (
                int(atom_ids[edge[1, index]]), int(atom_ids[edge[0, index]]),
                -int(shift[index, 0]), -int(shift[index, 1]), -int(shift[index, 2]),
            )
            reversed_indices.append(entries.get(key, index))
        return torch.tensor(reversed_indices, dtype=torch.long, device=edge.device)

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
        node_one_hot_columns = node_column_values(data, node_one_hot)
        atom_type = data[_keys.ATOM_TYPE_KEY].flatten()
        bond_type = data[_keys.EDGE_TYPE_KEY].flatten()
        latents, node_features, edge_features, cutoff_coeffs, active_edges = self.init_layer(edge_index, atom_type, bond_type, edge_sh, edge_length, node_one_hot_columns)

        # get the twobody part of the param.
        node_twoness, edge_twoness = self.two_ness(edge_index, atom_type, cutoff_coeffs, edge_length, node_one_hot_columns)
        data[_keys.EDGE_OVERLAP_KEY] = latents
        data[_keys.NODE_ATTRS_KEY] = node_twoness
        data[_keys.EDGE_ATTRS_KEY] = edge_twoness
        data[_keys.EDGE_FEATURES_KEY] = torch.zeros(edge_index.shape[1], self.idp.orbpair_irreps.dim, dtype=self.dtype, device=self.device)
        data[_keys.NODE_FEATURES_KEY] = torch.zeros(atom_type.shape[0], self.idp.orbpair_irreps.dim, dtype=self.dtype, device=self.device)
        if self.use_env:   # env / many-body message passing (only in "full" mode)
            # precompute the per-edge Wigner-D matrices once and share them across all SO2_Linear
            # calls of all layers (they rotate with the same active-edge directions).
            from dptb.nn.tensor_product import batch_wigner_D, _Jd
            from e3nn.o3 import xyz_to_angles
            _R = edge_vector[active_edges]
            _ang = xyz_to_angles(_R[:, [1, 2, 0]])
            wigner = batch_wigner_D(self._wigner_lmax, _ang[0], _ang[1], torch.zeros_like(_ang[0]), _Jd)
            # smooth boundary envelope for the env pathway (see boundary_envelope): exactly 1 below
            # 0.95*r_max, C2-smooth to 0 at r_max, so the env prediction is continuous in r at the
            # cutoff and float32 cutoff-coefficient noise cannot reach the output.
            benv = boundary_envelope(edge_length[active_edges], self.init_layer.r_max)
            for layer in self.layers:
                node_feature_columns = node_column_values(data, node_features)
                latents, node_features, edge_features = \
                    layer(
                        latents,
                        node_features,
                        node_feature_columns,
                        edge_features,
                        node_one_hot_columns,
                        edge_index,
                        edge_vector,
                        atom_type,
                        cutoff_coeffs,
                        active_edges,
                        wigner=wigner,
                        boundary_env=benv,
                    )


            data[_keys.NODE_FEATURES_KEY] = self.out_node(node_features)
            data[_keys.EDGE_FEATURES_KEY] = torch.index_copy(
                data[_keys.EDGE_FEATURES_KEY], 0, active_edges,
                self.out_edge(edge_features) * benv.unsqueeze(-1))

        # additive three-center factorized term (3b/2b+3b/full). It produces Hamiltonian blocks, but
        # Trinity outputs reduced-equivariant features, so we add its CG-decomposition (to_reduced):
        # NNENV's existing E3Hamiltonian transform then turns (env + three-center) into blocks together,
        # while the always-present two-body base is added by NNENV from EDGE_ATTRS (original convention).
        if self.use_3b:
            assert _keys.ENV_INDEX_KEY in data, \
                "trinity three_center needs the environment neighbour list; set er_max in the data options."
            data = self.three_center(data)
            data[_keys.EDGE_FEATURES_KEY] = data[_keys.EDGE_FEATURES_KEY] + self.three_center.to_reduced(data[EDGE_THREECENTER_KEY])
            data[_keys.NODE_FEATURES_KEY] = data[_keys.NODE_FEATURES_KEY] + self.three_center.to_reduced(data[NODE_THREECENTER_KEY])

        # enforce hermiticity of the env edge channel exactly: average each same-shell-pair channel
        # with (-1)^J times its reversed edge. The SK 2b and three-center terms already satisfy the
        # relation (the sym leaves them invariant), so this only constrains the env output; skipped
        # in the env-free modes where the features are hermitian by construction.
        if self.hermitian and self.use_env:
            ef = data[_keys.EDGE_FEATURES_KEY]
            reversed_ef = reverse_edge_values(data, ef)
            sym = 0.5 * (ef[:, self.herm_mask] + self.herm_sign * reversed_ef[:, self.herm_mask])
            ef = ef.clone()
            ef[:, self.herm_mask] = sym
            data[_keys.EDGE_FEATURES_KEY] = ef

        return data
    
@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor):
    return torch.nn.functional.softplus(x) - math.log(2.0)


# boundary_envelope moved to dptb/nn/cutoff.py (shared with the prediction heads in rescale.py);
# re-exported here for backwards compatibility. It makes the env pathway continuous at the cutoff:
# an active edge with cutoff coefficient eps -> 0 would otherwise still receive full-magnitude env
# features/messages (downstream MLP biases + O(1) spherical harmonics) while an inactive edge gets
# none (measured: 74/14682 edges on Al66O36 within 25 mA of r_max flipped between float32/float64
# with up to 0.7 eV output noise).
from dptb.nn.cutoff import boundary_envelope

class Twoness(torch.nn.Module):
    def __init__(
            self,
            # required params
            idp,
            r_max: Union[float, int, dict],
            num_types: int,
            n_radial_basis: int,
            # MLP parameters:
            two_body_latent_channels: list=[128, 128],
            latent_dim: int=128,
            # cutoffs
            device: Union[str, torch.device] = torch.device("cpu"),
            dtype: Union[str, torch.dtype] = torch.float32,
    ):
        super(Twoness, self).__init__()
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
        self.device = device
        self.dtype = dtype

        self.idp_sk = OrbitalMapper(self.idp.basis, method="sktb", device=self.device)
        self.idp_sk.get_skonsite_maps()
        onsite_param = torch.ones([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, 1], dtype=self.dtype, device=self.device)
        self.onsite_param = torch.nn.Parameter(onsite_param)

        # Node invariants for center and neighbor (chemistry)
        # Plus edge invariants for the edge (radius).
        self.two_body_latent = ScalarMLPFunction(
                        mlp_input_dimension=(2 * num_types + n_radial_basis),
                        mlp_output_dimension=latent_dim,
                        mlp_latent_dimensions=two_body_latent_channels,
                        mlp_nonlinearity="silu",
                        mlp_initialization="uniform",
                    )
        
        self.bessel = BesselBasis(r_max=self.r_max, num_basis=n_radial_basis, trainable=True)


    def forward(self, edge_index, atom_type, cutoff_coeffs, edge_length, node_one_hot):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        edge_invariants = self.bessel(edge_length)
        node_invariants = node_one_hot

        # Determine which edges are still in play
        prev_mask = cutoff_coeffs > 0
        active_edges = (cutoff_coeffs > 0).nonzero().squeeze(-1)

        # Compute latents
        edge_twoness = torch.zeros(
            (edge_index.shape[1], self.two_body_latent.out_features),
            dtype=self.dtype,
            device=self.device,
        )
        
        new_latents = self.two_body_latent(torch.cat([
            node_invariants[edge_center],
            node_invariants[edge_neighbor],
            edge_invariants,
        ], dim=-1)[prev_mask])

        # Apply cutoff, which propagates through to everything else
        edge_twoness = torch.index_copy(
            edge_twoness, 0, active_edges, 
            cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
            )

        node_twoness = self.onsite_param[atom_type.flatten()]

        return node_twoness, edge_twoness # the radial embedding x and the sperical hidden V

class InitLayer(torch.nn.Module):
    def __init__(
            self,
            # required params
            idp,
            num_types: int,
            n_radial_basis: int,
            r_max: Union[float, int, dict],
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
            spectral_balance: bool = False,
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
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_e = SeperableLayerNorm(
            irreps=self.irreps_out,
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
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
        
        weights_e = self.env_embed_mlp(latents[prev_mask])
        # features = self.bn(features)

        edge_features = self._env_weighter(
            edge_sh[prev_mask], weights_e
        )

        # boundary envelope: the MLP biases make edge_features O(1) even when the cutoff coefficient
        # (hence latents) vanishes, so without this the features are discontinuous at the cutoff
        # activation boundary (and float32 cutoff noise there turns into O(1) feature noise).
        # NOTE with a per-bond r_max dict this uses the global max; the residual discontinuity for
        # smaller per-bond cutoffs is suppressed by their own cutoff_coeffs going to zero well
        # inside the envelope's support.
        edge_features = edge_features * boundary_envelope(edge_length[prev_mask], self.r_max).unsqueeze(-1)

        node_features = scatter(
            edge_features,
            edge_center[active_edges],
            dim=0,
            dim_size=atom_type.numel(),
        )

        if self.env_sum_normalizations.ndim < 1:
            norm_const = self.env_sum_normalizations
        else:
            norm_const = self.env_sum_normalizations[atom_type.flatten()].unsqueeze(-1)
        
        node_features = node_features * norm_const
        node_features = self.sln_n(node_features)

        return latents, node_features, edge_features, cutoff_coeffs, active_edges # the radial embedding x and the sperical hidden V

class UpdateNode(torch.nn.Module):
    def __init__(
        self,
        edge_irreps_in: o3.Irreps,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        avg_num_neighbors: Optional[float] = None,
        so2_gate: bool = False,
        spectral_balance: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(UpdateNode, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.edge_irreps_in = edge_irreps_in
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

        self.sln = SeperableLayerNorm(
            irreps=self.irreps_in,
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_e = SeperableLayerNorm(
            irreps=self.edge_irreps_in,
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
            dtype=self.dtype,
            device=self.device
        )

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

        # P1: learnable per-l gain to counteract the Gate's l>0 attenuation. Init identity (1.0) so it
        # does NOT disturb the data-based statistics head calibration at init (see PerLGain); off if None.
        self.gain = PerLGain(self.activation.irreps_out, dtype=dtype, device=device) if spectral_balance else None

        self.tp = SO2_Linear(
            irreps_in=self.irreps_in+self.edge_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            so2_gate=so2_gate,
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
            self._res_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )

    def forward(self, latents, node_features, edge_features, atom_type, node_onehot, edge_index, edge_vector, active_edges, wigner=None, boundary_env=None):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = self.sln(node_features)
        message = self.tp(
            torch.cat(
                [new_node_features[edge_center[active_edges]], self.sln_e(edge_features)]
                , dim=-1), edge_vector[active_edges], latents[active_edges], wigner=wigner) # full_out_irreps

        message = self.activation(message)
        if self.gain is not None:
            message = self.gain(message)
        message = self.lin_post(message)
        scalars = message[:, :self.irreps_out[0].dim]

        # get the attention scores
        # weights = self.env_embed_mlps(self.latent_act(latents[active_edges]))
        # weights = torch_geometric.utils.softmax(weights, edge_center[active_edges], num_nodes=node_features.shape[0])
        weights = self.env_embed_mlps(latents[active_edges])
        weighted = self._env_weighter(message, weights)
        if boundary_env is not None:
            # smooth boundary envelope: env_embed_mlps has a bias, so messages from edges whose
            # cutoff coefficient (hence latents) vanish would otherwise contribute O(1) to the node
            # sum discontinuously (see boundary_envelope).
            weighted = weighted * boundary_env.unsqueeze(-1)
        new_node_features = scatter(
            weighted,
            edge_center[active_edges],
            dim=0,
            dim_size=node_features.shape[0],
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
        num_types,
        node_irreps_in: o3.Irreps,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        latent_dim: int,
        latent_channels: list=[128, 128],
        radial_emb: bool=False,
        radial_channels: list=[128, 128],
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        so2_gate: bool = False,
        spectral_balance: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(UpdateEdge, self).__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.node_irreps_in = node_irreps_in
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

        self.ln = torch.nn.LayerNorm(latent_dim)
        
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

        # P1: learnable per-l gain to counteract the Gate's l>0 attenuation. Init identity (1.0) so it
        # does NOT disturb the data-based statistics head calibration at init (see PerLGain); off if None.
        self.gain = PerLGain(self.activation.irreps_out, dtype=dtype, device=device) if spectral_balance else None

        self.tp = SO2_Linear(
            irreps_in=self.node_irreps_in+self.irreps_in+self.node_irreps_in,
            irreps_out=self.activation.irreps_in,
            latent_dim=latent_dim,
            radial_emb=radial_emb,
            radial_channels=radial_channels,
            so2_gate=so2_gate,
        )

        self.latents = ScalarMLPFunction(
            mlp_input_dimension=latent_dim+self.irreps_out[0].dim+2*num_types,
            mlp_output_dimension=latent_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )

        self.sln_e = SeperableLayerNorm(
            irreps=self.irreps_in,
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
            dtype=self.dtype,
            device=self.device
        )

        self.sln_n = SeperableLayerNorm(
            irreps=self.irreps_in,
            eps=5e-3,
            affine=True,
            normalization='component',
            std_balance_degrees=True,
            per_l=spectral_balance,
            dtype=self.dtype,
            device=self.device
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
            self._res_update_params = torch.nn.Parameter(
                res_update_params
            )
        else:
            self.register_buffer(
                "_res_update_params", res_update_params
            )
    
    def forward(self, latents, node_features, node_onehot, edge_features, edge_index, edge_vector, cutoff_coeffs, active_edges, wigner=None):
        edge_center = edge_index[0]
        edge_neighbor = edge_index[1]

        new_node_features = self.sln_n(node_features)
        new_edge_features = self.tp(
            torch.cat(
                [
                    new_node_features[edge_center[active_edges]],
                    self.sln_e(edge_features),
                    new_node_features[edge_neighbor[active_edges]]
                    ]
                , dim=-1), edge_vector[active_edges], latents[active_edges], wigner=wigner) # full_out_irreps
        
        scalars = new_edge_features[:, :self.tp.irreps_out[0].dim]
        assert len(scalars.shape) == 2
        new_edge_features = self.activation(new_edge_features)
        if self.gain is not None:
            new_edge_features = self.gain(new_edge_features)
        new_edge_features = self.lin_post(new_edge_features)

        scalars = new_edge_features[:, :self.irreps_out[0].dim]
        assert len(scalars.shape) == 2

        weights = self.edge_embed_mlps(latents[active_edges])
        new_edge_features = self._edge_weighter(new_edge_features, weights)

        # update latent
        latent_inputs_to_cat = [
            node_onehot[edge_center[active_edges]],
            self.ln(latents[active_edges]),
            scalars,
            node_onehot[edge_neighbor[active_edges]],
        ]

        new_latents = self.latents(torch.cat(latent_inputs_to_cat, dim=-1))
        new_latents = cutoff_coeffs[active_edges].unsqueeze(-1) * new_latents
        
        if self.res_update:
            update_coefficients = self._res_update_params.sigmoid()
            coefficient_old = torch.rsqrt(update_coefficients.square() + 1)
            coefficient_new = update_coefficients * coefficient_old
            edge_features = coefficient_new * new_edge_features + coefficient_old * self.linear_res(edge_features)

            latents = torch.index_copy(
                latents, 0, active_edges, 
                coefficient_new * new_latents + coefficient_old * latents[active_edges]
            )
        else:
            edge_features = new_edge_features
            latents = torch.index_copy(
                latents, 0, active_edges, 
                new_latents
            )

        return edge_features, latents
    

class Layer(torch.nn.Module):
    def __init__(
        self,
        num_types: int,
        # required params
        avg_num_neighbors: Optional[float] = None,
        irreps_in: o3.Irreps=None,
        irreps_out: o3.Irreps=None,
        tp_radial_emb: bool=False,
        tp_radial_channels: list=[128, 128],
        # MLP parameters:
        latent_channels: list=[128, 128],
        latent_dim: int=128,
        res_update: bool = True,
        res_update_ratios: Optional[List[float]] = None,
        res_update_ratios_learnable: bool = False,
        so2_gate: bool = False,
        spectral_balance: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Layer, self).__init__()

        self.res_update = res_update
        self.avg_num_neighbors = avg_num_neighbors
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.dtype = dtype
        self.device = device
        self.num_types = num_types

        # 1. update hidden
        # 2. update edge
        # 3. update node

        self.edge_update = UpdateEdge(
            node_irreps_in=self.irreps_in,
            num_types=num_types,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            latent_channels=latent_channels,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            so2_gate=so2_gate,
            spectral_balance=spectral_balance,
            dtype=dtype,
            device=device,
        )

        self.node_update = UpdateNode(
            edge_irreps_in=self.edge_update.irreps_out,
            irreps_in=self.irreps_in,
            irreps_out=self.irreps_out,
            latent_dim=latent_dim,
            radial_emb=tp_radial_emb,
            radial_channels=tp_radial_channels,
            res_update=res_update,
            res_update_ratios=res_update_ratios,
            res_update_ratios_learnable=res_update_ratios_learnable,
            avg_num_neighbors=avg_num_neighbors,
            so2_gate=so2_gate,
            spectral_balance=spectral_balance,
            dtype=dtype,
            device=device,
        )

    def forward(self, latents, node_features, node_feature_columns, edge_features, node_onehot_columns, edge_index, edge_vector, atom_type, cutoff_coeffs, active_edges, wigner=None, boundary_env=None):

        edge_features, latents = self.edge_update(latents, node_feature_columns, node_onehot_columns, edge_features, edge_index, edge_vector, cutoff_coeffs, active_edges, wigner=wigner)
        node_features = self.node_update(latents, node_features, edge_features, atom_type, node_onehot_columns, edge_index, edge_vector, active_edges, wigner=wigner, boundary_env=boundary_env)

        return latents, node_features, edge_features
