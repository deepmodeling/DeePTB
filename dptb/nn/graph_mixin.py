import random
from typing import Dict, Tuple, Callable, Any, Sequence, Union, Mapping, Optional
from collections import OrderedDict

import torch

from e3nn import o3

from dptb.data import AtomicDataDict
from dptb.utils import instantiate


class GraphModuleMixin:
    r"""Mixin parent class for ``torch.nn.Module``s that act on and return ``AtomicDataDict.Type`` graph data.

    All such classes should call ``_init_irreps`` in their ``__init__`` functions with information on the data fields they expect, require, and produce, as well as their corresponding irreps.
    """

    def _init_irreps(
        self,
        irreps_in: Dict[str, Any] = {},
        my_irreps_in: Dict[str, Any] = {},
        required_irreps_in: Sequence[str] = [],
        irreps_out: Dict[str, Any] = {},
    ):
        """Setup the expected data fields and their irreps for this graph module.

        ``None`` is a valid irreps in the context for anything that is invariant but not well described by an ``e3nn.o3.Irreps``. An example are edge indexes in a graph, which are invariant but are integers, not ``0e`` scalars.

        Args:
            irreps_in (dict): maps names of all input fields from previous modules or
                data to their corresponding irreps
            my_irreps_in (dict): maps names of fields to the irreps they must have for
                this graph module. Will be checked for consistancy with ``irreps_in``
            required_irreps_in: sequence of names of fields that must be present in
                ``irreps_in``, but that can have any irreps.
            irreps_out (dict): mapping names of fields that are modified/output by
                this graph module to their irreps.
        """
        # Coerce
        irreps_in = {} if irreps_in is None else irreps_in
        irreps_in = AtomicDataDict._fix_irreps_dict(irreps_in)
        # positions are *always* 1o, and always present
        if AtomicDataDict.POSITIONS_KEY in irreps_in:
            if irreps_in[AtomicDataDict.POSITIONS_KEY] != o3.Irreps("1x1o"):
                raise ValueError(
                    f"Positions must have irreps 1o, got instead `{irreps_in[AtomicDataDict.POSITIONS_KEY]}`"
                )
        irreps_in[AtomicDataDict.POSITIONS_KEY] = o3.Irreps("1o")
        # edges are also always present
        if AtomicDataDict.EDGE_INDEX_KEY in irreps_in:
            if irreps_in[AtomicDataDict.EDGE_INDEX_KEY] is not None:
                raise ValueError(
                    f"Edge indexes must have irreps None, got instead `{irreps_in[AtomicDataDict.EDGE_INDEX_KEY]}`"
                )
        irreps_in[AtomicDataDict.EDGE_INDEX_KEY] = None

        my_irreps_in = AtomicDataDict._fix_irreps_dict(my_irreps_in)

        irreps_out = AtomicDataDict._fix_irreps_dict(irreps_out)
        # Confirm compatibility:
        # with my_irreps_in
        for k in my_irreps_in:
            if k in irreps_in and irreps_in[k] != my_irreps_in[k]:
                raise ValueError(
                    f"The given input irreps {irreps_in[k]} for field '{k}' is incompatible with this configuration {type(self)}; should have been {my_irreps_in[k]}"
                )
        # with required_irreps_in
        for k in required_irreps_in:
            if k not in irreps_in:
                raise ValueError(
                    f"This {type(self)} requires field '{k}' to be in irreps_in"
                )
        # Save stuff
        self.irreps_in = irreps_in
        # The output irreps of any graph module are whatever inputs it has, overwritten with whatever outputs it has.
        new_out = irreps_in.copy()
        new_out.update(irreps_out)
        self.irreps_out = new_out

    def _add_independent_irreps(self, irreps: Dict[str, Any]):
        """
        Insert some independent irreps that need to be exposed to the self.irreps_in and self.irreps_out.
        The terms that have already appeared in the irreps_in will be removed.

        Args:
            irreps (dict): maps names of all new fields
        """

        irreps = {
            key: irrep for key, irrep in irreps.items() if key not in self.irreps_in
        }
        irreps_in = AtomicDataDict._fix_irreps_dict(irreps)
        irreps_out = AtomicDataDict._fix_irreps_dict(
            {key: irrep for key, irrep in irreps.items() if key not in self.irreps_out}
        )
        self.irreps_in.update(irreps_in)
        self.irreps_out.update(irreps_out)

    def _make_tracing_inputs(self, n):
        # We impliment this to be able to trace graph modules
        out = []
        for _ in range(n):
            batch = random.randint(1, 4)
            # TODO: handle None case
            # TODO: do only required inputs
            # TODO: dummy input if empty?
            out.append(
                {
                    "forward": (
                        {
                            k: i.randn(batch, -1)
                            for k, i in self.irreps_in.items()
                            if i is not None
                        },
                    )
                }
            )
        return out