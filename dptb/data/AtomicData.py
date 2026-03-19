"""AtomicData: neighbor graphs in (periodic) real space.

Authors: Albert Musaelian
"""

import warnings
from copy import deepcopy
from typing import Union, Tuple, Dict, Optional, List, Set, Sequence
from collections.abc import Mapping
import os

import numpy as np
import ase.neighborlist
import ase
from ase.calculators.singlepoint import SinglePointCalculator, SinglePointDFTCalculator
from ase.calculators.calculator import all_properties as ase_all_properties
from ase.stress import voigt_6_to_full_3x3_stress, full_3x3_to_voigt_6_stress
from ase.data import chemical_symbols
import itertools

import torch
import e3nn.o3

from . import AtomicDataDict
from .util import _TORCH_INTEGER_DTYPES
from dptb.utils.torch_geometric.data import Data
from dptb.utils.constants import atomic_num_dict
import logging

log = logging.getLogger(__name__)

# A type representing ASE-style periodic boundary condtions, which can be partial (the tuple case)
PBC = Union[bool, Tuple[bool, bool, bool]]


_DEFAULT_NESTED_FIELDS : Set[str] = {
    # AtomicDataDict.HAMILTONIAN_KEY,
    # AtomicDataDict.OVERLAP_KEY, not support nested type in this two since nested format does not support complex dtype
    AtomicDataDict.ENERGY_EIGENVALUE_KEY,
    AtomicDataDict.KPOINT_KEY,
}


_DEFAULT_LONG_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_INDEX_KEY,
    AtomicDataDict.ENV_INDEX_KEY, # new
    AtomicDataDict.ONSITENV_INDEX_KEY, # new
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.BATCH_KEY,
}

_DEFAULT_NODE_FIELDS: Set[str] = {
    AtomicDataDict.POSITIONS_KEY,
    AtomicDataDict.NODE_FEATURES_KEY,
    AtomicDataDict.NODE_ATTRS_KEY,
    AtomicDataDict.ATOMIC_NUMBERS_KEY,
    AtomicDataDict.ATOM_TYPE_KEY,
    AtomicDataDict.FORCE_KEY,
    AtomicDataDict.PER_ATOM_ENERGY_KEY,
    AtomicDataDict.NODE_HAMILTONIAN_KEY,
    AtomicDataDict.NODE_OVERLAP_KEY,
    AtomicDataDict.BATCH_KEY,
    AtomicDataDict.NODE_SOC_KEY # new
}

_DEFAULT_EDGE_FIELDS: Set[str] = {
    AtomicDataDict.EDGE_CELL_SHIFT_KEY,
    AtomicDataDict.EDGE_VECTORS_KEY,
    AtomicDataDict.EDGE_LENGTH_KEY,
    AtomicDataDict.EDGE_ATTRS_KEY,
    AtomicDataDict.EDGE_EMBEDDING_KEY,
    AtomicDataDict.EDGE_FEATURES_KEY,
    AtomicDataDict.EDGE_CUTOFF_KEY,
    AtomicDataDict.EDGE_ENERGY_KEY,
    AtomicDataDict.EDGE_OVERLAP_KEY,
    AtomicDataDict.EDGE_HAMILTONIAN_KEY,
    AtomicDataDict.EDGE_TYPE_KEY,
}

_DEFAULT_ENV_FIELDS: Set[str] = {
    AtomicDataDict.ENV_CELL_SHIFT_KEY,
    AtomicDataDict.ENV_VECTORS_KEY,
    AtomicDataDict.ENV_LENGTH_KEY,
    AtomicDataDict.ENV_ATTRS_KEY,
    AtomicDataDict.ENV_EMBEDDING_KEY,
    AtomicDataDict.ENV_FEATURES_KEY,
    AtomicDataDict.ENV_CUTOFF_KEY,
}

_DEFAULT_ONSITENV_FIELDS: Set[str] = {
    AtomicDataDict.ONSITENV_CELL_SHIFT_KEY,
    AtomicDataDict.ONSITENV_VECTORS_KEY,
    AtomicDataDict.ONSITENV_LENGTH_KEY,
    AtomicDataDict.ONSITENV_ATTRS_KEY,
    AtomicDataDict.ONSITENV_EMBEDDING_KEY,
    AtomicDataDict.ONSITENV_FEATURES_KEY,
    AtomicDataDict.ONSITENV_CUTOFF_KEY,
}

_DEFAULT_GRAPH_FIELDS: Set[str] = {
    AtomicDataDict.TOTAL_ENERGY_KEY,
    AtomicDataDict.STRESS_KEY,
    AtomicDataDict.VIRIAL_KEY,
    AtomicDataDict.PBC_KEY,
    AtomicDataDict.CELL_KEY,
    AtomicDataDict.BATCH_PTR_KEY,
    AtomicDataDict.KPOINT_KEY, # new # should be nested
    AtomicDataDict.HAMILTONIAN_KEY, # new # should be nested
    AtomicDataDict.OVERLAP_KEY, # new # should be nested
    AtomicDataDict.ENERGY_EIGENVALUE_KEY, # new # should be nested
    AtomicDataDict.EIGENVECTOR_KEY, # new # should be nested
    AtomicDataDict.ENERGY_WINDOWS_KEY, # new,
    AtomicDataDict.BAND_WINDOW_KEY, # new,
    AtomicDataDict.NODE_SOC_SWITCH_KEY # new
}

_NODE_FIELDS: Set[str] = set(_DEFAULT_NODE_FIELDS)
_EDGE_FIELDS: Set[str] = set(_DEFAULT_EDGE_FIELDS)
_ENV_FIELDS: Set[str] = set(_DEFAULT_ENV_FIELDS)
_ONSITENV_FIELDS: Set[str] = set(_DEFAULT_ONSITENV_FIELDS)
_GRAPH_FIELDS: Set[str] = set(_DEFAULT_GRAPH_FIELDS)
_LONG_FIELDS: Set[str] = set(_DEFAULT_LONG_FIELDS)
_NESTED_FIELDS: Set[str] = set(_DEFAULT_NESTED_FIELDS)


def register_fields(
    node_fields: Sequence[str] = [],
    edge_fields: Sequence[str] = [],
    env_fields: Sequence[str] = [],
    onsitenv_fields: Sequence[str] = [],
    graph_fields: Sequence[str] = [],
    long_fields: Sequence[str] = [],
) -> None:
    
    r"""Register fields as being per-atom, per-edge, or per-frame.

    Args:
        node_permute_fields: fields that are equivariant to node permutations.
        edge_permute_fields: fields that are equivariant to edge permutations.
    """
    
    node_fields: set = set(node_fields)
    edge_fields: set = set(edge_fields)
    env_fields: set = set(env_fields)
    onsitenv_fields: set = set(onsitenv_fields)
    graph_fields: set = set(graph_fields)
    long_fields: set = set(long_fields)
    allfields = node_fields.union(edge_fields, graph_fields, env_fields, onsitenv_fields)
    assert len(allfields) == len(node_fields) + len(edge_fields) + len(graph_fields)
    _NODE_FIELDS.update(node_fields)
    _EDGE_FIELDS.update(edge_fields)
    _ENV_FIELDS.update(env_fields)
    _ONSITENV_FIELDS.update(onsitenv_fields)
    _GRAPH_FIELDS.update(graph_fields)
    _LONG_FIELDS.update(long_fields)
    if len(set.union(_NODE_FIELDS, _EDGE_FIELDS, _GRAPH_FIELDS)) < (
        len(_NODE_FIELDS) + len(_EDGE_FIELDS) + len(_GRAPH_FIELDS)
    ):
        raise ValueError(
            "At least one key was registered as more than one of node, edge, or graph!"
        )


def deregister_fields(*fields: Sequence[str]) -> None:
    r"""Deregister a field registered with ``register_fields``.

    Silently ignores fields that were never registered to begin with.

    Args:
        *fields: fields to deregister.
    """
    for f in fields:
        assert f not in _DEFAULT_NODE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_EDGE_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_GRAPH_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_ENV_FIELDS, "Cannot deregister built-in field"
        assert f not in _DEFAULT_ONSITENV_FIELDS, "Cannot deregister built-in field"
        _NODE_FIELDS.discard(f)
        _EDGE_FIELDS.discard(f)
        _ENV_FIELDS.discard(f)
        _ONSITENV_FIELDS.discard(f)
        _GRAPH_FIELDS.discard(f)


def _register_field_prefix(prefix: str) -> None:
    """Re-register all registered fields as the same type, but with `prefix` added on."""
    assert prefix.endswith("_")
    register_fields(
        node_fields=[prefix + e for e in _NODE_FIELDS],
        edge_fields=[prefix + e for e in _EDGE_FIELDS],
        env_fields=[prefix + e for e in _ENV_FIELDS],
        onsitenv_fields=[prefix + e for e in _ONSITENV_FIELDS],
        graph_fields=[prefix + e for e in _GRAPH_FIELDS],
        long_fields=[prefix + e for e in _LONG_FIELDS],
    )


def _process_dict(kwargs, ignore_fields=[]):
    """Convert a dict of data into correct dtypes/shapes according to key"""
    # Deal with _some_ dtype issues

    # assert all nested_field is also graph_field
    for field in _NESTED_FIELDS:
        assert field in _GRAPH_FIELDS

    if AtomicDataDict.BATCH_KEY in kwargs:
        num_frames = kwargs[AtomicDataDict.BATCH_KEY].max() + 1
    else:
        num_frames = 1

    for k, v in kwargs.items():
        if k in ignore_fields:
            continue

        if k in _LONG_FIELDS:
            # Any property used as an index must be long (or byte or bool, but those are not relevant for atomic scale systems)
            # int32 would pass later checks, but is actually disallowed by torch
            kwargs[k] = torch.as_tensor(v, dtype=torch.long)
        elif k in _NESTED_FIELDS: # we need to transform the input array or tensors into nested tensors
            if num_frames > 1:
                if isinstance(v, np.ndarray): # this suggest that the dimension for each features are the same
                    v = torch.nested.as_nested_tensor(list(torch.as_tensor(v)), dtype=torch.get_default_dtype())
                elif isinstance(v, torch.Tensor) and not getattr(v, "is_nested"): # this also suggest the dimenion are the same
                    v = torch.nested.as_nested_tensor(list(v), dtype=torch.get_default_dtype())
                elif isinstance(v, list):
                    v = torch.nested.as_nested_tensor(v, dtype=torch.get_default_dtype())
                else:
                    assert v.is_nested # this assert that v is a tensor and is nested
            else:
                if isinstance(v, np.ndarray):
                    v = torch.as_tensor(v, dtype=torch.get_default_dtype())
                    v = torch.nested.as_nested_tensor([v], dtype=torch.get_default_dtype())
                elif isinstance(v, torch.Tensor) and not getattr(v, "is_nested"):
                    v = torch.nested.as_nested_tensor([v], dtype=torch.get_default_dtype())
                else:
                    assert v.is_nested

            kwargs[k] = v
        elif isinstance(v, bool):
            kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, np.ndarray):
            if np.issubdtype(v.dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif isinstance(v, list):
            ele_dtype = np.array(v).dtype
            if np.issubdtype(ele_dtype, np.floating):
                kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
            else:
                kwargs[k] = torch.as_tensor(v)
        elif np.issubdtype(type(v), np.floating):
            # Force scalars to be tensors with a data dimension
            # This makes them play well with irreps
            kwargs[k] = torch.as_tensor(v, dtype=torch.get_default_dtype())
        elif isinstance(v, torch.Tensor) and len(v.shape) == 0:
            # ^ this tensor is a scalar; we need to give it
            # a data dimension to play nice with irreps
            kwargs[k] = v

    for k, v in kwargs.items():
        if k in ignore_fields:
            continue

        if k not in _NESTED_FIELDS:
            if len(v.shape) == 0:
                kwargs[k] = v.unsqueeze(-1)
                v = kwargs[k]

            if k in set.union(_NODE_FIELDS, _EDGE_FIELDS) and len(v.shape) == 1:
                kwargs[k] = v.unsqueeze(-1)
                v = kwargs[k]

            if (
                k in _NODE_FIELDS
                and AtomicDataDict.POSITIONS_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.POSITIONS_KEY].shape[0]
            ):
                raise ValueError(
                    f"{k} is a node field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _EDGE_FIELDS
                and AtomicDataDict.EDGE_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.EDGE_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a edge field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _ENV_FIELDS
                and AtomicDataDict.ENV_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.ENV_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a env field but has the wrong dimension {v.shape}"
                )
            elif (
                k in _ONSITENV_FIELDS
                and AtomicDataDict.ONSITENV_INDEX_KEY in kwargs
                and v.shape[0] != kwargs[AtomicDataDict.ONSITENV_INDEX_KEY].shape[1]
            ):
                raise ValueError(
                    f"{k} is a env field but has the wrong dimension {v.shape}"
                )
            elif k in _GRAPH_FIELDS:
                if num_frames > 1 and v.shape[0] != num_frames:
                    raise ValueError(f"Wrong shape for graph property {k}")
        else: # when k in NESTED_FIELD, and also k must be in GRAPH_FIELDS
            if num_frames > 1 and v.size(0) != num_frames:
                raise ValueError(f"Wrong shape for NESTED property {k}")


class AtomicData(Data):
    """A neighbor graph for points in (periodic triclinic) real space.

    For typical cases either ``from_points`` or ``from_ase`` should be used to
    construct a AtomicData; they also standardize and check their input much more
    thoroughly.

    In general, ``node_features`` are features or input information on the nodes that will be fed through and transformed by the network, while ``node_attrs`` are _encodings_ fixed, inherant attributes of the atoms themselves that remain constant through the network.
    For example, a one-hot _encoding_ of atomic species is a node attribute, while some observed instantaneous property of that atom (current partial charge, for example), would be a feature.

    In general, ``torch.Tensor`` arguments should be of consistant dtype. Numpy arrays will be converted to ``torch.Tensor``s; those of floating point dtype will be converted to ``torch.get_current_dtype()`` regardless of their original precision. Scalar values (Python scalars or ``torch.Tensor``s of shape ``()``) a resized to tensors of shape ``[1]``. Per-atom scalar values should be given with shape ``[N_at, 1]``.

    ``AtomicData`` should be used for all data creation and manipulation outside of the model; inside of the model ``AtomicDataDict.Type`` is used.

    Args:
        pos (Tensor [n_nodes, 3]): Positions of the nodes.
        edge_index (LongTensor [2, n_edges]): ``edge_index[0]`` is the per-edge
            index of the source node and ``edge_index[1]`` is the target node.
        edge_cell_shift (Tensor [n_edges, 3], optional): which periodic image
            of the target point each edge goes to, relative to the source point.
        cell (Tensor [1, 3, 3], optional): the periodic cell for
            ``edge_cell_shift`` as the three triclinic cell vectors.
        node_features (Tensor [n_atom, ...]): the input features of the nodes, optional
        node_attrs (Tensor [n_atom, ...]): the attributes of the nodes, for instance the atom type, optional
        batch (Tensor [n_atom]): the graph to which the node belongs, optional
        atomic_numbers (Tensor [n_atom]): optional.
        atom_type (Tensor [n_atom]): optional.
        **kwargs: other data, optional.
    """

    def __init__(
        self, irreps: Dict[str, e3nn.o3.Irreps] = {}, _validate: bool = True, **kwargs
    ):

        # empty init needed by get_example
        if len(kwargs) == 0 and len(irreps) == 0:
            super().__init__()
            return

        # Check the keys
        if _validate:
            AtomicDataDict.validate_keys(kwargs)
            _process_dict(kwargs)

        super().__init__(num_nodes=len(kwargs["pos"]), **kwargs)

        if _validate:
            # Validate shapes
            assert self.pos.dim() == 2 and self.pos.shape[1] == 3
            assert self.edge_index.dim() == 2 and self.edge_index.shape[0] == 2
            if "edge_cell_shift" in self and self.edge_cell_shift is not None:
                assert self.edge_cell_shift.shape == (self.num_edges, 3)
                assert self.edge_cell_shift.dtype == self.pos.dtype
            # TODO: should we add checks for env too ?
            if "cell" in self and self.cell is not None:
                assert (self.cell.shape == (3, 3)) or (
                    self.cell.dim() == 3 and self.cell.shape[1:] == (3, 3)
                )
                assert self.cell.dtype == self.pos.dtype
            if "node_features" in self and self.node_features is not None:
                assert self.node_features.shape[0] == self.num_nodes
                assert self.node_features.dtype == self.pos.dtype
            if "node_attrs" in self and self.node_attrs is not None:
                assert self.node_attrs.shape[0] == self.num_nodes
                assert self.node_attrs.dtype == self.pos.dtype

            if (
                AtomicDataDict.ATOMIC_NUMBERS_KEY in self
                and self.atomic_numbers is not None
            ):
                assert self.atomic_numbers.dtype in _TORCH_INTEGER_DTYPES
            if "batch" in self and self.batch is not None:
                assert self.batch.dim() == 2 and self.batch.shape[0] == self.num_nodes
                # Check that there are the right number of cells
                if "cell" in self and self.cell is not None:
                    cell = self.cell.view(-1, 3, 3)
                    assert cell.shape[0] == self.batch.max() + 1

            # Validate irreps
            # __*__ is the only way to hide from torch_geometric
            self.__irreps__ = AtomicDataDict._fix_irreps_dict(irreps)
            for field, irreps in self.__irreps__:
                if irreps is not None:
                    assert self[field].shape[-1] == irreps.dim

    @classmethod
    def from_points(
        cls,
        pos=None,
        r_max: Union[float, int, dict] = None,
        self_interaction: bool = False,
        cell=None,
        pbc: Optional[PBC] = None,
        er_max: Optional[float] = None,
        oer_max: Optional[float] = None,
        **kwargs,
    ):
        """Build neighbor graph from points, optionally with PBC.

        Args:
            pos (np.ndarray/torch.Tensor shape [N, 3]): node positions. If Tensor, must be on the CPU.
            r_max (float): neighbor cutoff radius.
            cell (ase.Cell/ndarray [3,3], optional): periodic cell for the points. Defaults to ``None``.
            pbc (bool or 3-tuple of bool, optional): whether to apply periodic boundary conditions to all or each of
            the three cell vector directions. Defaults to ``False``.
            self_interaction (bool, optional): whether to include self edges for points. Defaults to ``False``. Note
            that edges between the same atom in different periodic images are still included. (See
            ``strict_self_interaction`` to control this behaviour.)
            strict_self_interaction (bool): Whether to include *any* self interaction edges in the graph, even if the
            two instances of the atom are in different periodic images. Defaults to True, should be True for most
            applications.
            **kwargs (optional): other fields to add. Keys listed in ``AtomicDataDict.*_KEY` will be treated specially.
        """
        if pos is None or r_max is None:
            raise ValueError("pos and r_max must be given.")

        if pbc is None:
            if cell is not None:
                raise ValueError(
                    "A cell was provided, but pbc weren't. Please explicitly probide PBC."
                )
            # there are no PBC if cell and pbc are not provided
            pbc = False

        if isinstance(pbc, bool):
            pbc = (pbc,) * 3
        else:
            assert len(pbc) == 3

        # TODO: We can only compute the edge vector one times with the largest radial distance among [r_max, er_max, oer_max]

        pos = torch.as_tensor(pos, dtype=torch.get_default_dtype())

        edge_index, edge_cell_shift, cell = neighbor_list_and_relative_vec(
            pos=pos,
            r_max=r_max,
            self_interaction=self_interaction,
            cell=cell,
            reduce=False,
            atomic_numbers=kwargs.get("atomic_numbers", None),
            pbc=pbc,
        )

        # Make torch tensors for data:
        if cell is not None:
            kwargs[AtomicDataDict.CELL_KEY] = cell.view(3, 3)
            kwargs[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = edge_cell_shift
        if pbc is not None:
            kwargs[AtomicDataDict.PBC_KEY] = torch.as_tensor(
                pbc, dtype=torch.bool
            ).view(3)

        # add env index
        if er_max is not None:
            env_index, env_cell_shift, _ = neighbor_list_and_relative_vec(
                pos=pos,
                r_max=er_max,
                self_interaction=self_interaction,
                cell=cell,
                reduce=False,
                atomic_numbers=kwargs.get("atomic_numbers", None),
                pbc=pbc,
            )

            if cell is not None:
                kwargs[AtomicDataDict.ENV_CELL_SHIFT_KEY] = env_cell_shift
            kwargs[AtomicDataDict.ENV_INDEX_KEY] = env_index
        
        # add onsitenv index
        if oer_max is not None:
            onsitenv_index, onsitenv_cell_shift, _ = neighbor_list_and_relative_vec(
                pos=pos,
                r_max=oer_max,
                self_interaction=self_interaction,
                cell=cell,
                reduce=False,
                atomic_numbers=kwargs.get("atomic_numbers", None),
                pbc=pbc
            )

            if cell is not None:
                kwargs[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = onsitenv_cell_shift
            kwargs[AtomicDataDict.ONSITENV_INDEX_KEY] = onsitenv_index
            
        return cls(edge_index=edge_index, pos=torch.as_tensor(pos), **kwargs)

    @classmethod
    def from_ase(
        cls,
        atoms,
        r_max: Union[float, int, dict],
        er_max: Optional[float] = None,
        oer_max: Optional[float] = None,
        key_mapping: Optional[Dict[str, str]] = {},
        include_keys: Optional[list] = [],
        **kwargs,
    ):
        """Build a ``AtomicData`` from an ``ase.Atoms`` object.

        Respects ``atoms``'s ``pbc`` and ``cell``.

        First tries to extract energies and forces from a single-point calculator associated with the ``Atoms`` if one is present and has those fields.
        If either is not found, the method will look for ``energy``/``energies`` and ``force``/``forces`` in ``atoms.arrays``.

        `get_atomic_numbers()` will be stored as the atomic_numbers attribute.

        Args:
            atoms (ase.Atoms): the input.
            r_max (float): neighbor cutoff radius.
            features (torch.Tensor shape [N, M], optional): per-atom M-dimensional feature vectors. If ``None`` (the
             default), uses a one-hot encoding of the species present in ``atoms``.
            include_keys (list): list of additional keys to include in AtomicData aside from the ones defined in
                 ase.calculators.calculator.all_properties. Optional
            key_mapping (dict): rename ase property name to a new string name. Optional
            **kwargs (optional): other arguments for the ``AtomicData`` constructor.

        Returns:
            A ``AtomicData``.
        """
        # from nequip.ase import NequIPCalculator

        assert "pos" not in kwargs

        default_args = set(
            [
                "numbers",
                "positions",
            ]  # ase internal names for position and atomic_numbers
            + ["pbc", "cell", "pos", "r_max", "er_max", "oer_max"]  # arguments for from_points method
            + list(kwargs.keys())
        )
        # the keys that are duplicated in kwargs are removed from the include_keys
        include_keys = list(
            set(include_keys + ase_all_properties + list(key_mapping.keys()))
            - default_args
        )

        km = {
            "forces": AtomicDataDict.FORCE_KEY,
            "energy": AtomicDataDict.TOTAL_ENERGY_KEY,
        }
        km.update(key_mapping)
        key_mapping = km

        add_fields = {}

        # Get info from atoms.arrays; lowest priority. copy first
        add_fields = {
            key_mapping.get(k, k): v
            for k, v in atoms.arrays.items()
            if k in include_keys
        }

        # Get info from atoms.info; second lowest priority.
        add_fields.update(
            {
                key_mapping.get(k, k): v
                for k, v in atoms.info.items()
                if k in include_keys
            }
        )

        # if atoms.calc is not None:

        #     if isinstance(
        #         atoms.calc, (SinglePointCalculator, SinglePointDFTCalculator)
        #     ):
        #         add_fields.update(
        #             {
        #                 key_mapping.get(k, k): deepcopy(v)
        #                 for k, v in atoms.calc.results.items()
        #                 if k in include_keys
        #             }
        #         )
        #     elif isinstance(atoms.calc, NequIPCalculator):
        #         pass  # otherwise the calculator breaks
        #     else:
        #         raise NotImplementedError(
        #             f"`from_ase` does not support calculator {atoms.calc}"
        #         )

        add_fields[AtomicDataDict.ATOMIC_NUMBERS_KEY] = atoms.get_atomic_numbers()

        # cell and pbc in kwargs can override the ones stored in atoms
        cell = kwargs.pop("cell", atoms.get_cell())
        pbc = kwargs.pop("pbc", atoms.pbc)

        # handle ASE-style 6 element Voigt order stress
        for key in (AtomicDataDict.STRESS_KEY, AtomicDataDict.VIRIAL_KEY):
            if key in add_fields:
                if add_fields[key].shape == (3, 3):
                    # it's already 3x3, do nothing else
                    pass
                elif add_fields[key].shape == (6,):
                    # it's Voigt order
                    add_fields[key] = voigt_6_to_full_3x3_stress(add_fields[key])
                else:
                    raise RuntimeError(f"bad shape for {key}")

        return cls.from_points(
            pos=atoms.positions,
            r_max=r_max,
            er_max=er_max,
            oer_max=oer_max,
            cell=cell,
            pbc=pbc,
            **kwargs,
            **add_fields,
        )

    def to_ase(
        self,
        type_mapper=None,
        extra_fields: List[str] = [],
    ) -> Union[List[ase.Atoms], ase.Atoms]:
        """Build a (list of) ``ase.Atoms`` object(s) from an ``AtomicData`` object.

        For each unique batch number provided in ``AtomicDataDict.BATCH_KEY``,
        an ``ase.Atoms`` object is created. If ``AtomicDataDict.BATCH_KEY`` does not
        exist in self, a single ``ase.Atoms`` object is created.

        Args:
            type_mapper: if provided, will be used to map ``ATOM_TYPES`` back into
                elements, if the configuration of the ``type_mapper`` allows.
            extra_fields: fields other than those handled explicitly (currently
                those defining the structure as well as energy, per-atom energy,
                and forces) to include in the output object. Per-atom (per-node)
                quantities will be included in ``arrays``; per-graph and per-edge
                quantities will be included in ``info``.

        Returns:
            A list of ``ase.Atoms`` objects if ``AtomicDataDict.BATCH_KEY`` is in self
            and is not None. Otherwise, a single ``ase.Atoms`` object is returned.
        """
        positions = self.pos
        edge_index = self[AtomicDataDict.EDGE_INDEX_KEY]
        if positions.device != torch.device("cpu"):
            raise TypeError(
                "Explicitly move this `AtomicData` to CPU using `.to()` before calling `to_ase()`."
            )
        if AtomicDataDict.ATOMIC_NUMBERS_KEY in self:
            atomic_nums = self.atomic_numbers
        elif type_mapper is not None and type_mapper.has_chemical_symbols:
            atomic_nums = type_mapper.untransform(self[AtomicDataDict.ATOM_TYPE_KEY])
        else:
            warnings.warn(
                "AtomicData.to_ase(): self didn't contain atomic numbers... using atom_type as atomic numbers instead, but this means the chemical symbols in ASE (outputs) will be wrong"
            )
            atomic_nums = self[AtomicDataDict.ATOM_TYPE_KEY]
        pbc = getattr(self, AtomicDataDict.PBC_KEY, None)
        cell = getattr(self, AtomicDataDict.CELL_KEY, None)
        batch = getattr(self, AtomicDataDict.BATCH_KEY, None)
        energy = getattr(self, AtomicDataDict.TOTAL_ENERGY_KEY, None)
        energies = getattr(self, AtomicDataDict.PER_ATOM_ENERGY_KEY, None)
        force = getattr(self, AtomicDataDict.FORCE_KEY, None)
        do_calc = any(
            k in self
            for k in [
                AtomicDataDict.TOTAL_ENERGY_KEY,
                AtomicDataDict.FORCE_KEY,
                AtomicDataDict.PER_ATOM_ENERGY_KEY,
                AtomicDataDict.STRESS_KEY,
            ]
        )

        # exclude those that are special for ASE and that we process seperately
        special_handling_keys = [
            AtomicDataDict.POSITIONS_KEY,
            AtomicDataDict.CELL_KEY,
            AtomicDataDict.PBC_KEY,
            AtomicDataDict.ATOMIC_NUMBERS_KEY,
            AtomicDataDict.TOTAL_ENERGY_KEY,
            AtomicDataDict.FORCE_KEY,
            AtomicDataDict.PER_ATOM_ENERGY_KEY,
            AtomicDataDict.STRESS_KEY,
        ]
        assert (
            len(set(extra_fields).intersection(special_handling_keys)) == 0
        ), f"Cannot specify keys handled in special ways ({special_handling_keys}) as `extra_fields` for atoms output--- they are output by default"

        if cell is not None:
            cell = cell.view(-1, 3, 3)
        if pbc is not None:
            pbc = pbc.view(-1, 3)

        if batch is not None:
            n_batches = batch.max() + 1
            cell = cell.expand(n_batches, 3, 3) if cell is not None else None
            pbc = pbc.expand(n_batches, 3) if pbc is not None else None
        else:
            n_batches = 1

        batch_atoms = []
        for batch_idx in range(n_batches):
            if batch is not None:
                mask = batch == batch_idx
                mask = mask.view(-1)
                # if both ends of the edge are in the batch, the edge is in the batch
                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
            else:
                mask = slice(None)
                edge_mask = slice(None)

            mol = ase.Atoms(
                numbers=atomic_nums[mask].view(-1),  # must be flat for ASE
                positions=positions[mask],
                cell=cell[batch_idx] if cell is not None else None,
                pbc=pbc[batch_idx] if pbc is not None else None,
            )

            if do_calc:
                fields = {}
                if energies is not None:
                    fields["energies"] = energies[mask].cpu().numpy()
                if energy is not None:
                    fields["energy"] = energy[batch_idx].cpu().numpy()
                if force is not None:
                    fields["forces"] = force[mask].cpu().numpy()
                if AtomicDataDict.STRESS_KEY in self:
                    fields["stress"] = full_3x3_to_voigt_6_stress(
                        self["stress"].view(-1, 3, 3)[batch_idx].cpu().numpy()
                    )
                mol.calc = SinglePointCalculator(mol, **fields)

            # add other information
            for key in extra_fields:
                if key in _NODE_FIELDS:
                    # mask it
                    mol.arrays[key] = (
                        self[key][mask].cpu().numpy().reshape(mask.sum(), -1)
                    )
                elif key in _EDGE_FIELDS:
                    mol.info[key] = (
                        self[key][edge_mask].cpu().numpy().reshape(edge_mask.sum(), -1)
                    )
                elif key == AtomicDataDict.EDGE_INDEX_KEY:
                    mol.info[key] = self[key][:, edge_mask].cpu().numpy()
                elif key in _GRAPH_FIELDS:
                    mol.info[key] = self[key][batch_idx].cpu().numpy().reshape(-1)
                else:
                    raise RuntimeError(
                        f"Extra field `{key}` isn't registered as node/edge/graph"
                    )

            batch_atoms.append(mol)

        if batch is not None:
            return batch_atoms
        else:
            assert len(batch_atoms) == 1
            return batch_atoms[0]

    def get_edge_vectors(data: Data) -> torch.Tensor:
        data = AtomicDataDict.with_edge_vectors(AtomicData.to_AtomicDataDict(data))
        return data[AtomicDataDict.EDGE_VECTORS_KEY]
    
    def get_env_vectors(data: Data) -> torch.Tensor:
        data = AtomicDataDict.with_env_vectors(AtomicData.to_AtomicDataDict(data))
        return data[AtomicDataDict.ENV_VECTORS_KEY]

    @staticmethod
    def to_AtomicDataDict(
        data: Union[Data, Mapping], exclude_keys=tuple()
    ) -> AtomicDataDict.Type:
        if isinstance(data, Data):
            keys = data.keys
        elif isinstance(data, Mapping):
            keys = data.keys()
        else:
            raise ValueError(f"Invalid data `{repr(data)}`")

        return {
            k: data[k]
            for k in keys
            if (
                k not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }

    @classmethod
    def from_AtomicDataDict(cls, data: AtomicDataDict.Type):
        # it's an AtomicDataDict, so don't validate-- assume valid:
        return cls(_validate=False, **data)

    @property
    def irreps(self):
        return self.__irreps__

    def __cat_dim__(self, key, value):
        if key == AtomicDataDict.EDGE_INDEX_KEY or key == AtomicDataDict.ENV_INDEX_KEY or key == AtomicDataDict.ONSITENV_INDEX_KEY:
            return 1  # always cat in the edge dimension
        elif key in _GRAPH_FIELDS:
            # graph-level properties and so need a new batch dimension
            if key in _NESTED_FIELDS:
                return 0
            else:
                return None
        else:
            return 0  # cat along node/edge dimension

    def without_nodes(self, which_nodes):
        """Return a copy of ``self`` with ``which_nodes`` removed.
        The returned object may share references to some underlying data tensors with ``self``.
        Args:
            which_nodes (index tensor or boolean mask)
        Returns:
            A new data object.
        """
        which_nodes = torch.as_tensor(which_nodes)
        if which_nodes.dtype == torch.bool:
            mask = ~which_nodes
        else:
            mask = torch.ones(self.num_nodes, dtype=torch.bool)
            mask[which_nodes] = False
        assert mask.shape == (self.num_nodes,)
        n_keeping = mask.sum()

        # Only keep edges where both from and to are kept
        edge_mask = mask[self.edge_index[0]] & mask[self.edge_index[1]]
        if hasattr(self, AtomicDataDict.ENV_INDEX_KEY):
            env_mask = mask[self.env_index[0]] & mask[self.env_index[1]]
        if hasattr(self, AtomicDataDict.ONSITENV_INDEX_KEY):
            onsitenv_mask = mask[self.onsitenv_index[0]] & mask[self.onsitenv_index[1]]
        # Create an index mapping:
        new_index = torch.full((self.num_nodes,), -1, dtype=torch.long)
        new_index[mask] = torch.arange(n_keeping, dtype=torch.long)

        new_dict = {}
        for k in self.keys:
            if k == AtomicDataDict.EDGE_INDEX_KEY:
                new_dict[AtomicDataDict.EDGE_INDEX_KEY] = new_index[
                    self.edge_index[:, edge_mask]
                ]
            elif k == AtomicDataDict.EDGE_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.EDGE_CELL_SHIFT_KEY] = self.edge_cell_shift[
                    edge_mask
                ]
            elif k == AtomicDataDict.CELL_KEY:
                new_dict[k] = self[k]
            elif k == AtomicDataDict.ENV_INDEX_KEY:
                new_dict[AtomicDataDict.ENV_INDEX_KEY] = new_index[
                    self.env_index[:, env_mask]
                ]
            elif k == AtomicDataDict.ENV_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.ENV_CELL_SHIFT_KEY] = self.env_cell_shift[
                    env_mask
                ]
            elif k == AtomicDataDict.ONSITENV_INDEX_KEY:
                new_dict[AtomicDataDict.ONSITENV_INDEX_KEY] = new_index[
                    self.onsitenv_index[:, onsitenv_mask]
                ]
            elif k == AtomicDataDict.ONSITENV_CELL_SHIFT_KEY:
                new_dict[AtomicDataDict.ONSITENV_CELL_SHIFT_KEY] = self.onsitenv_cell_shift[
                    onsitenv_mask
                ]
            else:
                if isinstance(self[k], torch.Tensor) and len(self[k]) == self.num_nodes:
                    new_dict[k] = self[k][mask]
                else:
                    new_dict[k] = self[k]

        new_dict["irreps"] = self.__irreps__

        return type(self)(**new_dict)


_ERROR_ON_NO_EDGES = os.environ.get("NEQUIP_ERROR_ON_NO_EDGES", "true").lower()
assert _ERROR_ON_NO_EDGES in ("true", "false"), "NEQUIP_ERROR_ON_NO_EDGES must be 'true' or 'false'"
_ERROR_ON_NO_EDGES = _ERROR_ON_NO_EDGES == "true"


def neighbor_list_and_relative_vec(
        pos,
        r_max,
        self_interaction=False,
        reduce=True,
        atomic_numbers=None,
        cell=None,
        pbc=False,
):
    """Create neighbor list and neighbor vectors based on radial cutoff.

    Create neighbor list (``edge_index``) and relative vectors
    (``edge_attr``) based on radial cutoff.

    Edges are given by the following convention:
    - ``edge_index[0]`` is the *source* (convolution center).
    - ``edge_index[1]`` is the *target* (neighbor).

    Thus, ``edge_index`` has the same convention as the relative vectors:
    :math:`\\vec{r}_{source, target}`

    Args:
        pos (shape [N, 3]): Positional coordinate; Tensor or numpy array. If Tensor, must be on CPU.
        r_max (float, dict): Radial cutoff distance. Can be a global float, or a dictionary.
        cell (numpy shape [3, 3]): Cell for periodic boundary conditions. Ignored if ``pbc == False``.
        pbc (bool or 3-tuple of bool): Periodic boundary conditions.
        self_interaction (bool): Whether to include same periodic image self-edges.
        reduce (bool): If True, returns an undirected graph (half edges). If False, returns a full directed graph.
        atomic_numbers (array-like): Atomic numbers of the atoms, required if r_max is a dict.

    Returns:
        edge_index (torch.tensor shape [2, num_edges])
        shifts (torch.tensor shape [num_edges, 3])
        cell (torch.Tensor [3, 3])
    """
    if isinstance(pbc, bool):
        pbc = (pbc,) * 3

    # 【优化】：确保如果输入的是 GPU Tensor，能够安全转换
    if atomic_numbers is not None:
        if isinstance(atomic_numbers, torch.Tensor):
            atomic_numbers_np = atomic_numbers.detach().cpu().numpy()
        else:
            atomic_numbers_np = np.asarray(atomic_numbers)
    else:
        atomic_numbers_np = None

    # -------------------------------------------------------------------------
    # 1. Parse r_max to ASE-compatible format for native pair-cutoff filtering
    # -------------------------------------------------------------------------
    if isinstance(r_max, dict):
        if atomic_numbers_np is None:
            raise ValueError("atomic_numbers must be provided when r_max is a dict.")

        if len(r_max) < len(set(atomic_numbers_np)):
            raise ValueError("The number of r_max is less than the number of required atom species.")

        first_key = next(iter(r_max.keys()))
        key_parts = str(first_key).split("-")

        if len(key_parts) == 1:
            # Atom-wise cutoffs: ASE naturally handles array input as R[i] + R[j] < cutoff
            r_map = get_r_map(r_max, atomic_numbers_np)
            r_map_np = r_map.detach().cpu().numpy() if isinstance(r_map, torch.Tensor) else np.asarray(r_map)
            user_cutoff = 0.5 * r_map_np[atomic_numbers_np - 1]

        elif len(key_parts) == 2:
            # Pair-wise cutoffs: Convert user string keys to ASE tuple keys
            r_map = get_r_map_bondwise(r_max, atomic_numbers_np)
            r_map_np = r_map.detach().cpu().numpy() if isinstance(r_map, torch.Tensor) else np.asarray(r_map)
            user_cutoff = {}
            unique_nums = np.unique(atomic_numbers_np)
            for z1 in unique_nums:
                for z2 in unique_nums:
                    user_cutoff[(int(z1), int(z2))] = float(r_map_np[int(z1) - 1, int(z2) - 1])
        else:
            raise ValueError("The r_max keys should be either atomic number or atomic number pair.")
    else:
        assert isinstance(r_max, (float, int))
        user_cutoff = float(r_max)

    # -------------------------------------------------------------------------
    # 2. Setup Device, Tensors, and Geometry
    # -------------------------------------------------------------------------
    if isinstance(pos, torch.Tensor):
        temp_pos = pos.detach().cpu().numpy()
        out_device = pos.device
        out_dtype = pos.dtype
    else:
        temp_pos = np.asarray(pos)
        out_device = torch.device("cpu")
        out_dtype = torch.get_default_dtype()

    if out_device.type != "cpu":
        warnings.warn(
            "Currently, neighborlists require a round trip to the CPU. Please pass CPU tensors if possible."
        )

    # 获取初始 cell 数据
    if isinstance(cell, torch.Tensor):
        temp_cell = cell.detach().cpu().numpy()
    elif cell is not None:
        temp_cell = np.asarray(cell)
    else:
        temp_cell = np.zeros((3, 3), dtype=temp_pos.dtype)

    # ASE 补全缺失的晶格向量
    temp_cell = ase.geometry.complete_cell(temp_cell)

    # 【修复2】：在此处（补全后）生成 cell_tensor，保证输出与 shifts 处于同一坐标系参考标准下
    cell_tensor = torch.as_tensor(temp_cell, device=out_device, dtype=out_dtype)

    # -------------------------------------------------------------------------
    # 3. Call core O(N) neighbor search algorithm
    # -------------------------------------------------------------------------
    # By default, primitive_neighbor_list returns a fully directed graph representing
    # both (i, j, S) and (j, i, -S). It also automatically removes self-edges (i=i, S=0)
    # if self_interaction=False.
    first_idx, second_idx, shifts = ase.neighborlist.primitive_neighbor_list(
        "ijS",
        pbc,
        temp_cell,
        temp_pos,
        cutoff=user_cutoff,
        numbers=atomic_numbers_np,
        self_interaction=self_interaction,
        use_scaled_positions=False,
    )
    # -------------------------------------------------------------------------
    # 4. Handle graph reduction state
    # -------------------------------------------------------------------------
    if reduce:
        # Convert full directed graph to undirected half-graph
        mask_lt = first_idx < second_idx
        mask_eq = first_idx == second_idx

        # Deduplicate mirrored periodic boundaries for i == j
        eq_first = first_idx[mask_eq]
        eq_shifts = shifts[mask_eq]
        eq_keep = np.zeros(len(eq_first), dtype=bool)

        rev_dict = {}
        for i in range(len(eq_first)):
            key = f"{eq_first[i]}_{eq_shifts[i]}"
            key_rev = f"{eq_first[i]}_{-eq_shifts[i]}"
            rev_dict[key] = True

            if not (rev_dict.get(key_rev, False) and rev_dict.get(key, False)):
                eq_keep[i] = True

            if self_interaction and (eq_shifts[i] == 0).all():
                eq_keep[i] = True

        # Combine reduction masks
        final_mask = mask_lt.copy()
        final_mask[mask_eq] = eq_keep

        first_idx = first_idx[final_mask]
        second_idx = second_idx[final_mask]
        shifts = shifts[final_mask]

    # Note: If `reduce=False`, the output of primitive_neighbor_list is exactly the
    # full bidirectional graph structure required, so no post-processing is needed.

    # -------------------------------------------------------------------------
    # 5. Build output tensors
    # -------------------------------------------------------------------------
    first_idx_t = torch.as_tensor(first_idx, dtype=torch.long, device=out_device)
    second_idx_t = torch.as_tensor(second_idx, dtype=torch.long, device=out_device)
    shifts_t = torch.as_tensor(shifts, dtype=out_dtype, device=out_device)

    edge_index = torch.vstack((first_idx_t, second_idx_t))

    return edge_index, shifts_t, cell_tensor


def get_r_map(r_max: dict, atomic_numbers=None):
    """
    Returns a torch tensor representing the mapping of atomic species to their maximum distances.

    Args:
        r_max (dict): A dictionary mapping atomic species to their maximum distances.
        atomic_numbers (list, optional): A list of atomic numbers to validate against the atomic species. Defaults to None.

    Returns:
        torch.Tensor: A torch tensor representing the mapping of atomic species to their maximum distances.
    """
    atom_species_num = [atomic_num_dict[k] for k in r_max.keys()]
    if atomic_numbers is not None:
        for i in atomic_numbers:
            assert i in atom_species_num
    r_map = torch.zeros(max(atom_species_num))
    for k, v in r_max.items():
        r_map[atomic_num_dict[k]-1] = v
    return r_map

def get_r_map_bondwise(r_max:dict, atomic_numbers=None):
    """
    Calculate the bondwise distance map based on the maximum bond length dictionary.

    Args:
        r_max (dict): A dictionary containing the maximum bond lengths for different atom pairs.
        atomic_numbers (list, optional): A list of atomic numbers. Defaults to None.

    Returns:
        torch.Tensor: A torch tensor representing the bondwise distance map.
    """
    atom_species_num = []
    for k in r_max.keys():
        assert len(k.split('-')) == 2
        atom_a, atom_b = k.split('-')
        if atomic_num_dict[atom_a] not in atom_species_num:
            atom_species_num.append(atomic_num_dict[atom_a])    
        if atomic_num_dict[atom_b] not in atom_species_num:
            atom_species_num.append(atomic_num_dict[atom_b])    
    
    if atomic_numbers is not None:
        for i in atomic_numbers:
            assert i in atom_species_num

    r_map = torch.zeros(max(atom_species_num), max(atom_species_num))
    for k, v in r_max.items():
        atom_a, atom_b = k.split('-')
        
        inv_value = r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1]
        if inv_value == 0:
            r_map[atomic_num_dict[atom_a]-1, atomic_num_dict[atom_b]-1] = v
            r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1] = v
        else:
            mean_val = (v + inv_value) / 2
            r_map[atomic_num_dict[atom_a]-1, atomic_num_dict[atom_b]-1] = mean_val
            r_map[atomic_num_dict[atom_b]-1, atomic_num_dict[atom_a]-1] = mean_val

    return r_map