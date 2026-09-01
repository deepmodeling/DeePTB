"""Two-center-factorized three-center Hamiltonian term for DeePTB.

Implements  H^(3)_{AB} = sum_{C != A,B} P_{AC} D_C P_{CB}  as a DeePTB module: it consumes the
graph already built in the ``AtomicDataDict`` (edge list, edge vectors) and uses ``OrbitalMapper``
for all orbital / projector indexing, exactly like ``Trinity``/``Slem``. Nothing is integrated:
mirroring the two-body path, an MLP emits the *reduced* two-body values of ``P_{AC}=<phi_A|beta_C>``
and the matrix ``D_C``; ``wigner_D`` (the "CG basis" rotation) turns them into blocks.

The projector set on each center is just another ``OrbitalMapper`` basis (``idp_proj``), so no new
basis/neighbour/angular bookkeeping is introduced. See ``docs/adr/0004-three-center-factorization.md``.
"""

import re
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_runstats.scatter import scatter

from dptb.data import AtomicDataDict
from dptb.data.AtomicDataDict import with_edge_vectors, with_env_vectors
from dptb.data.transforms import OrbitalMapper
from dptb.data import _keys
from dptb.utils.constants import anglrMId
from dptb.nn.tensor_product import wigner_D
from dptb.nn.radial_basis import BesselBasis
from dptb.nn.base import ScalarMLPFunction
from dptb.nn.cutoff import polynomial_cutoff
from dptb.utils.distributed import column_atom_index, route_relation_contributions
from e3nn.o3 import xyz_to_angles, wigner_3j

# block keys are defined canonically in dptb.data._keys; re-export for importers of this module.
EDGE_THREECENTER_KEY = _keys.EDGE_THREECENTER_KEY   # hopping correction H^(3)_AB, per basis bond
NODE_THREECENTER_KEY = _keys.NODE_THREECENTER_KEY   # onsite correction H^(3)_AA, per atom


def _l_list(idp: OrbitalMapper, species: str) -> List[int]:
    """Ordered angular momenta of a species' basis, in ``OrbitalMapper`` block order."""
    return [anglrMId[re.findall(r"[a-z]", o)[0]] for o in idp.basis[species]]


def _offsets(l_list: List[int]):
    off, out = 0, []
    for l in l_list:
        out.append(off)
        off += 2 * l + 1
    return out, off


def _skbasis(l1: int, l2: int, dtype, device) -> torch.Tensor:
    """|m|-selection matrices coupling ``l1`` (left) and ``l2`` (right) in the local bond frame.
    Same construction as ``SKHamiltonian._initialize_basis``; shape ``[2l1+1, 2l2+1, min+1]``."""
    mats = []
    for im in range(min(l1, l2) + 1):
        m = torch.zeros(2 * l1 + 1, 2 * l2 + 1, dtype=dtype, device=device)
        if im == 0:
            m[l1, l2] = 1.0
        else:
            m[l1 + im, l2 + im] = 1.0
            m[l1 - im, l2 - im] = 1.0
        mats.append(m)
    return torch.stack(mats, dim=-1)


class CenterCouplingMatrix(nn.Module):
    """Center-local coupling ``D_C`` in the projector space of each species (spec point 4).

    Each species' coupling is built compactly over *its own* projector shells (block-diagonal in
    ``l``, identity in ``m`` -> rotation invariant; symmetric -> Hermitian) and then placed into the
    shared full-projector space via ``idp_proj.mask_to_basis`` so ``expand`` always returns the same
    ``[nproj_full, nproj_full]`` shape (zeros where the species has no projector). ``dense`` is a
    full symmetric block for tests only."""

    def __init__(self, idp_proj: OrbitalMapper, mode: str = "block_diag", init_std: float = 0.1,
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        assert mode in ("diag", "block_diag", "dense")
        self.mode, self.idp_proj, self.dtype, self.device = mode, idp_proj, dtype, device
        self.nproj_full = idp_proj.full_basis_norb
        self.channels_of_l: Dict[str, Dict[int, List[int]]] = {}
        self.params = nn.ParameterDict()
        for s in idp_proj.type_names:
            l_list = _l_list(idp_proj, s)
            groups: Dict[int, List[int]] = {}
            for ch, l in enumerate(l_list):
                groups.setdefault(l, []).append(ch)
            self.channels_of_l[s] = groups
            # full-basis positions of this species' (expanded) projector channels.
            t = idp_proj.chemical_symbol_to_type[s]
            self.register_buffer(f"idx_{s}", idp_proj.mask_to_basis[t].nonzero().flatten())
            if mode == "dense":
                n = _offsets(l_list)[1]
                self.params[s] = nn.Parameter(init_std * torch.randn(n, n, dtype=dtype, device=device))
            else:
                for l, chans in groups.items():
                    shape = (len(chans),) if mode == "diag" else (len(chans), len(chans))
                    self.params[f"{s}__l{l}"] = nn.Parameter(init_std * torch.randn(*shape, dtype=dtype, device=device))

    def _compact(self, species: str) -> torch.Tensor:
        l_list = _l_list(self.idp_proj, species)
        offs, n = _offsets(l_list)
        if self.mode == "dense":
            M = self.params[species]
            return 0.5 * (M + M.t())
        D = torch.zeros(n, n, dtype=self.dtype, device=self.device)
        for l, chans in self.channels_of_l[species].items():
            p = self.params[f"{species}__l{l}"]
            Ml = torch.diag(p) if self.mode == "diag" else 0.5 * (p + p.t())
            eye = torch.eye(2 * l + 1, dtype=self.dtype, device=self.device)
            for ri, ci in enumerate(chans):
                si = slice(offs[ci], offs[ci] + 2 * l + 1)
                for rj, cj in enumerate(chans):
                    D[si, slice(offs[cj], offs[cj] + 2 * l + 1)] = Ml[ri, rj] * eye
        return D

    def expand(self, species: str) -> torch.Tensor:
        """``D_C`` in the shared full-projector space: ``[nproj_full, nproj_full]``."""
        idx = getattr(self, f"idx_{species}")
        D = torch.zeros(self.nproj_full, self.nproj_full, dtype=self.dtype, device=self.device)
        D[idx[:, None], idx[None, :]] = self._compact(species)
        return D


class ThreeCenterFactorized(nn.Module):
    """Assemble the third-center correction ``H^(3)_{AB} = sum_C P_{AC} D_C P_{CB}`` on the graph.

    Works for heterogeneous species: everything lives in ``OrbitalMapper``'s ``full_basis`` (the union
    of all species' shells), so orbital/projector tensors are uniform (enabling the batched GEMM), and
    ``mask_to_basis`` zeros the channels a species does not actually have. For a single shared layout
    the masks are all-ones and this reduces to the plain dense case. Output blocks are in full_basis
    orbital order (``[norb_full, norb_full]``), matching ``E3Hamiltonian``'s convention.

    Two cutoffs (matching the physics):
      * the target blocks ``H^(3)_{AB}`` are produced *only* for the basis-cutoff bonds
        ``EDGE_INDEX`` (the user's ``r_max`` -- same block set as the two-body ``<A|B>``);
      * the projector reach ``A-C`` / ``C-B`` uses the separate, tunable environment cutoff
        ``er_max``: it is both the ``ENV_INDEX`` neighbour cutoff (built by
        ``AtomicData.from_points(er_max=...)``) and the smooth radial cutoff of the projector
        overlaps -- a single value."""

    def __init__(self, basis: Dict[str, Union[str, list]], projectors: Dict[str, Union[str, list]],
                 idp: Optional[OrbitalMapper] = None, er_max: float = 5.0, n_radial_basis: int = 8,
                 latent_channels: List[int] = [64, 64], coupling_mode: str = "block_diag",
                 dtype=torch.float32, device="cpu"):
        super().__init__()
        self.dtype, self.device = dtype, device
        self.idp = idp or OrbitalMapper(basis, method="e3tb", device=device)
        self.idp_proj = OrbitalMapper(projectors, method="e3tb", device=device)
        assert self.idp_proj.chemical_symbol_to_type == self.idp.chemical_symbol_to_type, \
            "projectors must be defined for the same species as the basis"
        self.num_types = len(self.idp.type_names)

        # work in the shared full_basis (union of all species' shells); mask_to_basis selects the
        # channels each species actually has. Shell l-lists drive the CG selection + rotation.
        self.orb_l = [anglrMId[re.findall(r"[a-z]", o)[0]] for o in self.idp.full_basis]
        self.proj_l = [anglrMId[re.findall(r"[a-z]", o)[0]] for o in self.idp_proj.full_basis]
        self.orb_off, self.norb = _offsets(self.orb_l)
        self.proj_off, self.nproj = _offsets(self.proj_l)
        # per-species presence masks over the expanded full_basis (orbital) / full projector space.
        self.register_buffer("orb_mask", self.idp.mask_to_basis.to(dtype))       # [n_types, norb]
        self.register_buffer("proj_mask", self.idp_proj.mask_to_basis.to(dtype))  # [n_types, nproj]

        # reduced-value layout: for P_{XC}, left = projector (center), right = orbital (neighbour).
        self._slices, off = {}, 0
        for a, lp in enumerate(self.proj_l):
            for i, lo in enumerate(self.orb_l):
                nm = min(lp, lo) + 1
                self._slices[(a, i)] = slice(off, off + nm)
                off += nm
                key = f"sel_{lp}_{lo}"
                if not hasattr(self, key):
                    self.register_buffer(key, _skbasis(lp, lo, dtype, device))
        self.reduced_dim = off

        # reduced values from two-body invariants [onehot(center), onehot(neighbour), bessel(R_AC)],
        # smoothly cut off at the environment cutoff er_max (the ENV neighbour-list cutoff).
        self.bessel = BesselBasis(r_max=torch.tensor(er_max, dtype=dtype, device=device), num_basis=n_radial_basis, trainable=True)
        # register as a buffer (with device) so model.to(device) moves it; a plain tensor attribute
        # stays on CPU and breaks the polynomial_cutoff below when the model runs on GPU. Non-persistent:
        # it is fully determined by the `er_max` config, so it stays out of the checkpoint (no layout change).
        self.register_buffer("er_max", torch.tensor(er_max, dtype=dtype, device=device), persistent=False)
        self.reduced_mlp = ScalarMLPFunction(
            mlp_input_dimension=2 * self.num_types + n_radial_basis,
            mlp_output_dimension=self.reduced_dim,
            mlp_latent_dimensions=latent_channels,
            mlp_nonlinearity="silu",
            mlp_initialization="uniform",
        )
        self.coupling = CenterCouplingMatrix(self.idp_proj, coupling_mode, dtype=dtype, device=device)

        # dense-block -> reduced edge-feature layout (for adding into E3TB EDGE_FEATURES, exactly like
        # ``block_to_feature``): each full_basis shell pair io<=jo maps its [2li+1,2lj+1] sub-block,
        # row-major, into idp.orbpair_maps[io-jo]. The reflective (B,A) edge reconstructs the lower
        # part via Hermiticity, which the assembled H^(3) satisfies.
        self.reduced_matrix_element = self.idp.reduced_matrix_element
        opm = self.idp.get_orbpair_maps()
        fb = self.idp.full_basis
        self._feat_slices = []
        for a, io in enumerate(fb):
            for b in range(a, len(fb)):
                sl = opm[f"{io}-{fb[b]}"]
                self._feat_slices.append((sl, self.orb_off[a], self.orb_off[a] + 2 * self.orb_l[a] + 1,
                                          self.orb_off[b], self.orb_off[b] + 2 * self.orb_l[b] + 1))

        # Clebsch-Gordan basis per orbpairtype, identical to E3Hamiltonian._initialize_CG_basis, so
        # to_reduced() is the exact inverse of the E3Hamiltonian(decompose=False) transform that NNENV
        # applies -- letting Trinity add the block correction in the reduced-equivariant space.
        self.idp.get_orbpairtype_maps()
        self._cg = {}
        for opt in self.idp.orbpairtype_maps.keys():
            l1, l2 = anglrMId[opt[0]], anglrMId[opt[2]]
            self._cg[opt] = torch.cat(
                [wigner_3j(int(l1), int(l2), int(lird), dtype=dtype, device=device) * (2 * lird + 1) ** 0.5
                 for lird in range(abs(l2 - l1), l1 + l2 + 1)], dim=-1)

    def to_feature(self, dense: torch.Tensor) -> torch.Tensor:
        """Pack dense ``[n, norb, norb]`` blocks into the reduced E3TB *block* feature layout
        ``[n, idp.reduced_matrix_element]`` (same extraction as ``block_to_feature``; the onsite block
        is symmetric so ``feature_to_block`` reconstructs it)."""
        n = dense.shape[0]
        feat = torch.zeros(n, self.reduced_matrix_element, dtype=self.dtype, device=self.device)
        for sl, r0, r1, c0, c1 in self._feat_slices:
            feat[:, sl] = dense[:, r0:r1, c0:c1].reshape(n, (r1 - r0) * (c1 - c0))
        return feat

    def to_reduced(self, dense: torch.Tensor) -> torch.Tensor:
        """Convert dense ``[n, norb, norb]`` blocks into the reduced-*equivariant* orbpair features
        that Trinity's ``out_edge``/``out_node`` produce (the input to E3Hamiltonian). This is the CG
        decomposition, i.e. the exact inverse of E3Hamiltonian(decompose=False), so the correction can
        be added pre-transform and picked up by the existing NNENV transform."""
        block = self.to_feature(dense)
        n = dense.shape[0]
        reduced = torch.zeros(n, self.reduced_matrix_element, dtype=self.dtype, device=self.device)
        for opt, sl in self.idp.orbpairtype_maps.items():
            l1, l2 = anglrMId[opt[0]], anglrMId[opt[2]]
            n_rme = (2 * l1 + 1) * (2 * l2 + 1)
            hr = block[:, sl].reshape(n, (sl.stop - sl.start) // n_rme, 2 * l1 + 1, 2 * l2 + 1) \
                .permute(0, 2, 3, 1)  # (n,nL,nR,n_pair)
            rme = torch.sum(self._cg[opt][None, :, :, :, None] * hr[:, :, :, None, :], dim=(1, 2))
            reduced[:, sl] = rme.transpose(1, 2).flatten(1)
        return reduced

    # ---------------------------------------------------------------- reduced values & P blocks
    def reduced_values(self, env_index: torch.Tensor, env_length: torch.Tensor,
                       atom_type: torch.Tensor) -> torch.Tensor:
        """Reduced two-body values of P_{XC} for the projector (ENV) edges C -> X."""
        onehot = F.one_hot(atom_type, self.num_types).to(self.dtype)
        r = env_length.to(self.dtype)
        feat = torch.cat([onehot[env_index[0]], onehot[env_index[1]], self.bessel(r)], dim=-1)
        cut = polynomial_cutoff(r, self.er_max.reshape(-1)).flatten().unsqueeze(-1)
        return self.reduced_mlp(feat) * cut

    def compute_P(self, vec: torch.Tensor, reduced: torch.Tensor) -> torch.Tensor:
        """Global-frame ``P_{XC}`` blocks ``[n_edge, nproj, norb]`` for edges (center C -> neighbour X):
        left index = projector on the center, right index = orbital on the neighbour."""
        vec = vec.to(self.dtype)
        n = vec.shape[0]
        P = torch.zeros(n, self.nproj, self.norb, dtype=self.dtype, device=self.device)
        if n == 0:
            return P
        angle = xyz_to_angles(vec[:, [1, 2, 0]])   # same convention as SKHamiltonian.forward
        zeros = torch.zeros_like(angle[0])
        Dl = {l: wigner_D(int(l), angle[0], angle[1], zeros) for l in set(self.proj_l + self.orb_l)}
        for a, lp in enumerate(self.proj_l):
            for i, lo in enumerate(self.orb_l):
                sel = getattr(self, f"sel_{lp}_{lo}")
                p = reduced[:, self._slices[(a, i)]]
                local = torch.einsum("mno,eo->emn", sel, p)
                glob = torch.einsum("elm,emo,eko->elk", Dl[lp], local, Dl[lo])
                P[:, self.proj_off[a]:self.proj_off[a] + 2 * lp + 1,
                     self.orb_off[i]:self.orb_off[i] + 2 * lo + 1] = glob
        return P

    def _D_per_edge(self, centers: torch.Tensor, atom_type: torch.Tensor):
        """``D_C`` for every edge, indexed by the *center's* species. Returns ``[nproj, nproj]`` when
        there is a single species (broadcast) or ``[n_edge, nproj, nproj]`` otherwise -- vectorized."""
        if self.num_types == 1:
            return self.coupling.expand(self.idp.type_names[0])
        Dstack = torch.stack([self.coupling.expand(s) for s in self.idp.type_names], dim=0)  # [T,np,np]
        return Dstack[atom_type[centers]]

    @staticmethod
    def _cocenter_edge_pairs(centers: torch.Tensor):
        """All ordered edge pairs (e1, e2), e1 != e2, that share the same center C.

        Fully vectorized (no Python loop over centers): sort edges by center, then for a center with
        degree d emit its d*d local pairs via cumulative offsets + ``repeat_interleave``. This is the
        triplet index for the three-center sum; the batched GEMM then runs over *all* triangles at
        once. Total triplets T = sum_C deg(C)^2, which is inherent to the three-center term."""
        device = centers.device
        E = centers.shape[0]
        if E == 0:
            z = torch.empty(0, dtype=torch.long, device=device)
            return z, z
        order = torch.argsort(centers)
        counts = torch.unique_consecutive(centers[order], return_counts=True)[1]  # degree per center
        nc = counts.shape[0]
        group_off = torch.cumsum(counts, 0) - counts        # start of each center's block in `order`
        pair_counts = counts * counts                       # d^2 per center
        pair_off = torch.cumsum(pair_counts, 0) - pair_counts
        T = int(pair_counts.sum())
        if T == 0:
            z = torch.empty(0, dtype=torch.long, device=device)
            return z, z
        t = torch.arange(T, device=device)
        cop = torch.repeat_interleave(torch.arange(nc, device=device), pair_counts)  # center of pair
        local = t - pair_off[cop]                           # flat index within the d*d block
        d = counts[cop]
        i = torch.div(local, d, rounding_mode="floor")
        j = local - i * d
        base = group_off[cop]
        e1, e2 = order[base + i], order[base + j]
        keep = i != j                                       # drop C-A-A (single-neighbour "pairs")
        return e1[keep], e2[keep]

    def env_P(self, data: AtomicDataDict.Type):
        """Masked global-frame ``P_{XC}`` for every projector (ENV) edge C -> X: zero the projector
        rows the center species lacks and the orbital columns the neighbour species lacks."""
        env = data[AtomicDataDict.ENV_INDEX_KEY]
        atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        column_type = atom_type
        ghost_type = data.get(AtomicDataDict.GHOST_ATOM_TYPE_KEY)
        if ghost_type is not None:
            column_type = torch.cat((atom_type, ghost_type.flatten().to(atom_type.device)))
        reduced = self.reduced_values(env, data[AtomicDataDict.ENV_LENGTH_KEY], column_type)
        P = self.compute_P(data[AtomicDataDict.ENV_VECTORS_KEY], reduced)
        P = P * self.proj_mask[column_type[env[0]]].unsqueeze(-1) * self.orb_mask[column_type[env[1]]].unsqueeze(1)
        return P, env, atom_type

    # ---------------------------------------------------------------- assembly
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        data = with_edge_vectors(data, with_lengths=True)     # basis bonds (A, B) -> target blocks
        data = with_env_vectors(data, with_lengths=True)      # projector reach A-C, C-B (er_max)
        edge = data[AtomicDataDict.EDGE_INDEX_KEY]
        P, env, atom_type = self.env_P(data)
        n_atoms = atom_type.shape[0]
        n_edge = edge.shape[1]

        G = torch.matmul(self._D_per_edge(env[0], atom_type), P)         # D_C P_{CX}, [n_env, nproj, norb]

        out = torch.zeros(n_edge, self.norb, self.norb, dtype=self.dtype, device=self.device)
        column_ids = column_atom_index(data).to(edge.device)
        edge_shift = data.get(AtomicDataDict.EDGE_CELL_SHIFT_KEY)
        if edge_shift is None:
            edge_shift = torch.zeros((n_edge, 3), dtype=torch.long, device=edge.device)
        env_shift = data.get(AtomicDataDict.ENV_CELL_SHIFT_KEY)
        if env_shift is None:
            env_shift = torch.zeros((env.shape[1], 3), dtype=torch.long, device=env.device)
        edge_keys = torch.cat((
            column_ids[edge[0]].reshape(-1, 1),
            column_ids[edge[1]].reshape(-1, 1),
            edge_shift.round().long(),
        ), dim=1)
        edge_lookup = {
            tuple(key): idx for idx, key in enumerate(edge_keys.detach().cpu().tolist())
        }
        e1, e2 = self._cocenter_edge_pairs(env[0])                       # triangles over ENV, no py loop
        triangle_keys = torch.cat((
            column_ids[env[1][e1]].reshape(-1, 1),
            column_ids[env[1][e2]].reshape(-1, 1),
            (env_shift[e2] - env_shift[e1]).round().long(),
        ), dim=1)
        triangle_values = torch.bmm(P[e1].transpose(-1, -2), G[e2])
        routed_keys, routed_values = route_relation_contributions(
            data, triangle_keys, triangle_values
        )
        self._index_add_matched(out, edge_lookup, routed_keys, routed_values)
        data[_keys.EDGE_THREECENTER_KEY] = out

        # onsite correction H^(3)_AA = sum_{C != A} P_AC D_C P_CA: the A==B "triangle" is just each
        # env edge C->A on its own, so scatter P[e]^T D_C P[e] to its neighbour atom A (fully batched).
        node = torch.zeros(n_atoms, self.norb, self.norb, dtype=self.dtype, device=self.device)
        onsite_keys = column_ids[env[1]].reshape(-1, 1)
        onsite_values = torch.bmm(P.transpose(-1, -2), G)
        routed_keys, routed_values = route_relation_contributions(
            data, onsite_keys, onsite_values
        )
        node_lookup = {
            (int(identity),): index
            for index, identity in enumerate(column_ids[:n_atoms].detach().cpu().tolist())
        }
        self._index_add_matched(node, node_lookup, routed_keys, routed_values)
        data[_keys.NODE_THREECENTER_KEY] = node
        return data

    @staticmethod
    def _index_add_matched(out, lookup, routed_keys, routed_values):
        """Accumulate routed contributions into the local rows their key names.

        Unmatched keys are dropped: those are triangles whose target bond
        (A, B, R) is not a basis bond, so H^(3) fills exactly the same blocks
        as <A|B>."""
        targets, selected = [], []
        for index, key in enumerate(routed_keys.detach().cpu().tolist()):
            target = lookup.get(tuple(key))
            if target is not None:
                targets.append(target)
                selected.append(index)
        if selected:
            out.index_add_(
                0,
                torch.as_tensor(targets, dtype=torch.long, device=out.device),
                routed_values[torch.as_tensor(selected, dtype=torch.long, device=out.device)],
            )

    def reference(self, data: AtomicDataDict.Type) -> torch.Tensor:
        """Naive per-triangle assembly consuming the same masked ``P``; validation only."""
        data = with_edge_vectors(data, with_lengths=True)
        data = with_env_vectors(data, with_lengths=True)
        edge = data[AtomicDataDict.EDGE_INDEX_KEY]
        P, env, atom_type = self.env_P(data)
        n_atoms, n_edge = atom_type.shape[0], edge.shape[1]
        env_of = {(int(env[0, e]), int(env[1, e])): e for e in range(env.shape[1])}
        out = torch.zeros(n_edge, self.norb, self.norb, dtype=self.dtype, device=self.device)
        for e in range(n_edge):
            A, B = int(edge[0, e]), int(edge[1, e])
            H = torch.zeros(self.norb, self.norb, dtype=self.dtype, device=self.device)
            for C in range(n_atoms):
                if C in (A, B) or (C, A) not in env_of or (C, B) not in env_of:
                    continue
                D = self.coupling.expand(self.idp.type_names[int(atom_type[C])])
                H = H + P[env_of[(C, A)]].t() @ D @ P[env_of[(C, B)]]     # P_AC D_C P_CB
            out[e] = H
        return out

    def reference_node(self, data: AtomicDataDict.Type) -> torch.Tensor:
        """Naive onsite reference H^(3)_AA = sum_{C != A} P_AC D_C P_CA; validation only."""
        data = with_env_vectors(data, with_lengths=True)
        P, env, atom_type = self.env_P(data)
        node = torch.zeros(atom_type.shape[0], self.norb, self.norb, dtype=self.dtype, device=self.device)
        for e in range(env.shape[1]):
            C, A = int(env[0, e]), int(env[1, e])
            D = self.coupling.expand(self.idp.type_names[int(atom_type[C])])
            node[A] = node[A] + P[e].t() @ D @ P[e]
        return node
