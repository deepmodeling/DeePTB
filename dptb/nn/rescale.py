import math
import torch
from torch_runstats.scatter import scatter
from dptb.data import _keys
import logging
from typing import Optional, List
import torch.nn.functional
from e3nn.o3 import Linear
from dptb.data import AtomicDataDict

class PerSpeciesScaleShift(torch.nn.Module):
    """Scale and/or shift a predicted per-atom property based on (learnable) per-species/type parameters.

    Args:
        field: the per-atom field to scale/shift.
        num_types: the number of types in the model.
        shifts: the initial shifts to use, one per atom type.
        scales: the initial scales to use, one per atom type.
        arguments_in_dataset_units: if ``True``, says that the provided shifts/scales are in dataset
            units (in which case they will be rescaled appropriately by any global rescaling later
            applied to the model); if ``False``, the provided shifts/scales will be used without modification.

            For example, if identity shifts/scales of zeros and ones are provided, this should be ``False``.
            But if scales/shifts computed from the training data are used, and are thus in dataset units,
            this should be ``True``.
        out_field: the output field; defaults to ``field``.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types) * shifts
            assert shifts.shape == (num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types) * scales
            assert scales.shape == (num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[species_idx].view(-1, 1) + in_field
        data[self.out_field] = in_field
        return data

    # def update_for_rescale(self, rescale_module):
    #     if hasattr(rescale_module, "related_scale_keys"):
    #         if self.out_field not in rescale_module.related_scale_keys:
    #             return
    #     if self.arguments_in_dataset_units and rescale_module.has_scale:
    #         logging.debug(
    #             f"PerSpeciesScaleShift's arguments were in dataset units; rescaling:\n  "
    #             f"Original scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
    #             f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
    #         )
    #         with torch.no_grad():
    #             if self.has_scales:
    #                 self.scales.div_(rescale_module.scale_by)
    #             if self.has_shifts:
    #                 self.shifts.div_(rescale_module.scale_by)
    #         logging.debug(
    #             f"  New scales: {TypeMapper.format(self.scales, self.type_names) if self.has_scales else 'n/a'} "
    #             f"shifts: {TypeMapper.format(self.shifts, self.type_names) if self.has_shifts else 'n/a'}"
    #         )

class PerEdgeSpeciesScaleShift(torch.nn.Module):
    """Sum edgewise energies.

    Includes optional per-species-pair edgewise energy scales.
    """

    field: str
    out_field: str
    scales_trainble: bool
    shifts_trainable: bool
    has_scales: bool
    has_shifts: bool

    def __init__(
        self,
        field: str,
        num_types: int,
        shifts: Optional[List[float]],
        scales: Optional[List[float]],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        **kwargs,
    ):
        """Sum edges into nodes."""
        super(PerEdgeSpeciesScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field

        self.has_shifts = shifts is not None

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types, num_types) * scales
            assert scales.shape == (num_types, num_types,), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=torch.get_default_dtype())
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types, num_types) * shifts
            assert shifts.shape == (num_types, num_types,), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)



    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]
        edge_neighbor = data[AtomicDataDict.EDGE_INDEX_KEY][1]

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        center_species = species_idx[edge_center]
        neighbor_species = species_idx[edge_neighbor]
        in_field = data[self.field]

        assert len(in_field) == len(
            edge_center
        ), "in_field doesnt seem to have correct per-edge shape"


        if self.has_scales:
            in_field = self.scales[center_species, neighbor_species].view(-1, 1) * in_field
        if self.has_shifts:
            in_field = self.shifts[center_species, neighbor_species].view(-1, 1) + in_field
        
        data[self.out_field] = in_field

        return data