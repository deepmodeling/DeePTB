import math
import torch
from torch_runstats.scatter import scatter
from dptb.data import _keys
import logging
from typing import Optional, List, Union
import torch.nn.functional
from e3nn.o3 import Linear
from e3nn.util.jit import compile_mode
from dptb.data import AtomicDataDict
import e3nn.o3 as o3

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
    
class E3PerEdgeSpeciesScaleShift(torch.nn.Module):
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
        irreps_in,
        shifts: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        r_max: Optional[float] = None,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        """Sum edges into nodes."""
        super(E3PerEdgeSpeciesScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
        # when r_max is known, the additive shifts are enveloped by boundary_envelope(r, r_max)^2.
        # The shift is otherwise DISCONTINUOUS at the cutoff-activation boundary: this forward skips
        # exactly-zero rows (padded/inactive edges), so an inactive edge gets no shift while an
        # epsilon-active one gets the full per-channel shift (up to ~0.7 eV) — and float32 noise in
        # the polynomial cutoff flips activation for edges near r_max. With the (squared) envelope
        # the shift vanishes smoothly where such flips can occur, restoring continuity in r.
        self.boundary_r_max = float(r_max) if r_max is not None else None
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self.irreps_in = irreps_in
        self.num_scalar = 0
        self.device = device
        self.dtype = dtype
        self.shift_index = []
        self.scale_index = []

        start = 0
        start_scalar = 0
        for mul, ir in irreps_in:
            if str(ir) == "0e":
                self.num_scalar += mul
                self.shift_index += list(range(start_scalar, start_scalar + mul))
                start_scalar += mul
            else:
                self.shift_index += [-1] * mul * ir.dim

            for _ in range(mul):
                self.scale_index += [start] * ir.dim
                start += 1

        self.shift_index = torch.as_tensor(self.shift_index, dtype=torch.long, device=device)
        self.scale_index = torch.as_tensor(self.scale_index, dtype=torch.long, device=device)

        self.has_shifts = shifts is not None
        self.has_scales = scales is not None

        # optional data-driven radial decay envelope: the effective per-channel scale becomes
        # scale_ch * exp(-kappa_ch * (r - r0_ch)). LCAO hopping norms decay near-exponentially with
        # bond length (measured 1e3-1e6x over a 2-7.4 A window), which a constant per-channel scale
        # cannot absorb; with the envelope the network only has to learn O(1) values at every r.
        # kappa=0 (default) makes the envelope exactly 1, i.e. the historical behaviour. Buffers are
        # always registered so the state_dict layout does not depend on whether statistics were set.
        self.register_buffer("decay_kappa", torch.zeros(num_types*num_types, irreps_in.num_irreps, dtype=self.dtype, device=self.device))
        self.register_buffer("decay_r0", torch.zeros(num_types*num_types, irreps_in.num_irreps, dtype=self.dtype, device=self.device))

        if scales is not None:
            scales = torch.as_tensor(scales, dtype=self.dtype, device=device)
            if len(scales.reshape(-1)) == 1:
                scales = scales * torch.ones(num_types*num_types, self.irreps_in.num_irreps, dtype=self.dtype, device=self.device)
            assert scales.shape == (num_types*num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.dtype, device=device)
            if len(shifts.reshape(-1)) == 1:
                shifts = shifts * torch.ones(num_types*num_types, self.num_scalar, dtype=self.dtype, device=self.device)
            assert shifts.shape == (num_types*num_types, self.num_scalar), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

    def set_scale_shift(self, scales: torch.Tensor=None, shifts: torch.Tensor=None):
        self.has_scales = scales is not None or self.has_scales
        if scales is not None:
            assert scales.shape == (self.num_types*self.num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales}"
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        
        self.has_shifts = shifts is not None or self.has_shifts
        if shifts is not None:
            assert shifts.shape == (self.num_types*self.num_types, self.num_scalar), f"Invalid shape of shifts {shifts}"
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

    def set_decay(self, kappa: torch.Tensor, r0: torch.Tensor):
        """Set the per-(bond-type, irrep) radial decay envelope exp(-kappa*(r-r0)); kappa=0 disables
        a channel's envelope. Fitted from dataset statistics (see E3statistics)."""
        assert kappa.shape == (self.num_types*self.num_types, self.irreps_in.num_irreps), f"Invalid shape of decay kappa {kappa.shape}"
        assert r0.shape == kappa.shape, f"Invalid shape of decay r0 {r0.shape}"
        self.decay_kappa.copy_(kappa.to(dtype=self.dtype, device=self.device).clamp_min(0.0))
        self.decay_r0.copy_(r0.to(dtype=self.dtype, device=self.device))

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        mask = data[self.field][:,0] != 0 # strictly zero valued point must come from masked edges
        in_field = data[self.field][mask]
        species_idx = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()[mask]

        

        assert len(in_field) == len(
            edge_center[mask]
        ), "in_field doesnt seem to have correct per-edge shape"

        if self.has_scales:
            scales = self.scales[species_idx]
            if bool((self.decay_kappa > 0).any()):
                # radial decay envelope: scale_ch(r) = scale_ch * exp(-kappa_ch * (r - r0_ch))
                assert AtomicDataDict.EDGE_LENGTH_KEY in data, \
                    "decay envelope needs edge lengths; call with_edge_vectors(with_lengths=True) first"
                r = data[AtomicDataDict.EDGE_LENGTH_KEY][mask].unsqueeze(-1)
                scales = scales * torch.exp(-self.decay_kappa[species_idx] * (r - self.decay_r0[species_idx]))
            in_field = scales[:, self.scale_index].view(-1, self.irreps_in.dim) * in_field
        if self.has_shifts:
            shifts = self.shifts[species_idx][:,self.shift_index[self.shift_index>=0]].view(-1, self.num_scalar)
            if self.boundary_r_max is not None:
                assert AtomicDataDict.EDGE_LENGTH_KEY in data, \
                    "shift boundary envelope needs edge lengths; call with_edge_vectors(with_lengths=True) first"
                from dptb.nn.cutoff import boundary_envelope
                env = boundary_envelope(data[AtomicDataDict.EDGE_LENGTH_KEY][mask], self.boundary_r_max)
                shifts = shifts * (env * env).unsqueeze(-1)
            in_field[:, self.shift_index>=0] = shifts + in_field[:, self.shift_index>=0]

        data[self.out_field][mask] = in_field

        return data

class E3PerSpeciesScaleShift(torch.nn.Module):
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
        irreps_in,
        shifts: Optional[torch.Tensor],
        scales: Optional[torch.Tensor],
        out_field: Optional[str] = None,
        scales_trainable: bool = False,
        shifts_trainable: bool = False,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super().__init__()
        self.num_types = num_types
        self.field = field
        self.out_field = f"shifted_{field}" if out_field is None else out_field
        self.irreps_in = irreps_in
        self.num_scalar = 0
        self.shift_index = []
        self.scale_index = []
        self.dtype = dtype
        self.device = device

        start = 0
        start_scalar = 0
        for mul, ir in irreps_in:
            # only the scalar irreps can be shifted
            # all the irreps can be scaled
            if str(ir) == "0e":
                self.num_scalar += mul
                self.shift_index += list(range(start_scalar, start_scalar + mul))
                start_scalar += mul
            else:
                self.shift_index += [-1] * mul * ir.dim
            for _ in range(mul):
                self.scale_index += [start] * ir.dim
                start += 1

        self.shift_index = torch.as_tensor(self.shift_index, dtype=torch.long, device=device)
        self.scale_index = torch.as_tensor(self.scale_index, dtype=torch.long, device=device)

        self.has_shifts = shifts is not None
        if shifts is not None:
            shifts = torch.as_tensor(shifts, dtype=self.dtype, device=device)
            if len(shifts.reshape([-1])) == 1:
                shifts = torch.ones(num_types, self.num_scalar, dtype=dtype, device=device) * shifts
            assert shifts.shape == (num_types,self.num_scalar), f"Invalid shape of shifts {shifts}"
            self.shifts_trainable = shifts_trainable
            if shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)

        self.has_scales = scales is not None
        if scales is not None:
            scales = torch.as_tensor(scales, dtype=torch.get_default_dtype())
            if len(scales.reshape([-1])) == 1:
                scales = torch.ones(num_types, self.irreps_in.num_irreps, dtype=dtype, device=device) * scales
            assert scales.shape == (num_types,self.irreps_in.num_irreps), f"Invalid shape of scales {scales}"
            self.scales_trainable = scales_trainable
            if scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)

    def set_scale_shift(self, scales: torch.Tensor=None, shifts: torch.Tensor=None):
        self.has_scales = scales is not None or self.has_scales
        if scales is not None:
            assert scales.shape == (self.num_types, self.irreps_in.num_irreps), f"Invalid shape of scales {scales}"
            if self.scales_trainable:
                self.scales = torch.nn.Parameter(scales)
            else:
                self.register_buffer("scales", scales)
        
        self.has_shifts = shifts is not None or self.has_shifts
        if shifts is not None:
            assert shifts.shape == (self.num_types, self.num_scalar), f"Invalid shape of shifts {shifts}"
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)


    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        in_field = data[self.field]
        assert len(in_field) == len(
            species_idx
        ), "in_field doesnt seem to have correct per-atom shape"
        if self.has_scales:
            in_field = self.scales[species_idx][:,self.scale_index].view(-1, self.irreps_in.dim) * in_field
        if self.has_shifts:
            shifts = self.shifts[species_idx][:,self.shift_index[self.shift_index>=0]].view(-1, self.num_scalar)
            in_field[:, self.shift_index>=0] = shifts + in_field[:, self.shift_index>=0]
        data[self.out_field] = in_field
        return data
    

@compile_mode("script")
class E3ElementLinear(torch.nn.Module):
    """Sum edgewise energies.
    Includes optional per-species-pair edgewise energy scales.
    """

    weight_numel: int

    def __init__(
        self,
        irreps_in: o3.Irreps,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        super(E3ElementLinear, self).__init__()
        self.irreps_in = irreps_in
        self.num_scalar = 0
        self.device = device
        self.dtype = dtype
        self.shift_index = []
        self.scale_index = []

        count_scales= 0
        count_shift = 0
        for mul, ir in irreps_in:
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

        self.weight_numel = irreps_in.num_irreps + self.num_scalar
        assert count_scales + count_shift == self.weight_numel
        self.num_scales = count_scales
        self.num_shifts = count_shift

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor]=None):

        if x.shape[0] == 0:
            # Per-row scaling/shifting of an empty batch; reshape(n, -1)
            # below is ambiguous at n == 0.
            return x

        scales = weights[:, :self.num_scales] if weights is not None else None
        if weights is not None:
            if weights.shape[1] > self.num_scales:
                shifts = weights[:, self.num_scales:]
            else:
                shifts = None
        else:
            shifts = None

        if scales is not None:
            assert len(scales) == len(
                x
            ), "in_field doesnt seem to have correct shape as scales"
            x = scales[:,self.scale_index].reshape(x.shape[0], -1) * x
        else:
            x = x

        if shifts is not None:
            assert len(shifts) == len(
                x
            ), "in_field doesnt seem to have correct shape as shifts"
            
            # bias = torch.zeros_like(x)
            # bias[:, self.shift_index.ge(0)] = shifts[:,self.shift_index[self.shift_index.ge(0)]].reshape(-1, self.num_scalar)
            # x = x + bias
            x[:, self.shift_index.ge(0)] = shifts[:,self.shift_index[self.shift_index.ge(0)]].reshape(-1, self.num_scalar) + x[:, self.shift_index.ge(0)]
        else:
            x = x

        return x