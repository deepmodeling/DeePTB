import math
import torch
from torch_runstats.scatter import scatter
from dptb.data import _keys
import logging
from typing import Optional, List, Union
import torch.nn.functional
from e3nn.o3 import Linear
from dptb.nn.sktb import HoppingFormula, bond_length_list
from e3nn.util.jit import compile_mode
from dptb.data import AtomicDataDict
from dptb.utils.constants import atomic_num_dict
import e3nn.o3 as o3

log = logging.getLogger(__name__)

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
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        """Sum edges into nodes."""
        super(E3PerEdgeSpeciesScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
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



    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        species_idx = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
        in_field = data[self.field]

        assert len(in_field) == len(
            edge_center
        ), "in_field doesnt seem to have correct per-edge shape"

        if self.has_scales:
            in_field = self.scales[species_idx][:,self.scale_index].view(-1, self.irreps_in.dim) * in_field
        if self.has_shifts:
            shifts = self.shifts[species_idx][:,self.shift_index[self.shift_index>=0]].view(-1, self.num_scalar)
            in_field[:, self.shift_index>=0] = shifts + in_field[:, self.shift_index>=0]
        
        data[self.out_field] = in_field

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


class E3PerEdgeSpeciesRadialDpdtScaleShift(torch.nn.Module):
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
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
        **kwargs,
    ):
        """Sum edges into nodes."""
        super(E3PerEdgeSpeciesRadialDpdtScaleShift, self).__init__()
        self.num_types = num_types
        self.field = field
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

        self.r0 = [] # initilize r0

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
            assert shifts.shape == (self.num_types*self.num_types, self.num_scalar, 7), f"Invalid shape of shifts {shifts}"
            if self.shifts_trainable:
                self.shifts = torch.nn.Parameter(shifts)
            else:
                self.register_buffer("shifts", shifts)
    
    def fit_radialdpdt_shift(self, decay, idp):
        shifts = torch.randn(self.num_types*self.num_types, self.num_scalar, 7, dtype=self.dtype, device=self.device)
        shifts.requires_grad_()
        optimizer = torch.optim.Adam([shifts], lr=0.01)
        lrsch = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3000, threshold=1e-5, eps=1e-5, verbose=True)
        bond_sym = list(decay.keys())
        bsz = 128

        for sym in idp.type_names:
            self.r0.append(bond_length_list[atomic_num_dict[sym]-1])
        self.r0 = torch.tensor(self.r0, device=self.device, dtype=self.dtype)

        #TODO: check wether exist some bond that does not have eneough values, this may appear in sparse dopping.
        #TODO: check whether there is bond that does not cover the range bwtween equilirbium r0 to r_cut. This may appear in some hetrogenous system.
        n_edge_length = []
        edge_lengths = {}
        scalar_decays = {}
        for bsym in decay:
            n_edge_length.append(len(decay[bsym]["edge_length"]))
            edge_lengths[bsym] = decay[bsym]["edge_length"].type(self.dtype).to(self.device)
            scalar_decays[bsym] = decay[bsym]["scalar_decay"].type(self.dtype).to(self.device)


        if min(n_edge_length) <= bsz: 
            log.warning("There exist edge that does not have enough values for fitting edge decaying behaviour, please use decay == False.")
        
        edge_number = idp._index_to_ZZ.T
        for i in range(40000):
            optimizer.zero_grad()
            rs = [None] * len(bond_sym)
            frs = [None] * len(bond_sym)
            # construct the dataset
            for bsym in decay:
                bt = idp.bond_to_type[bsym]
                random_index = torch.randint(0, len(edge_lengths[bsym]), (bsz,))
                rs[bt] = edge_lengths[bsym][random_index]
                frs[bt] = scalar_decays[bsym][:,random_index].T # [bsz, n_scalar]
            rs = torch.cat(rs, dim=0)
            frs = torch.cat(frs, dim=0)
            r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[edge_number-1].sum(0)
            r0 = r0.unsqueeze(1).repeat(1, bsz).reshape(-1)
            
            paraArray=shifts.reshape(-1, 1, self.num_scalar, 7).repeat(1,bsz,1,1).reshape(-1, self.num_scalar, 7)
            
            fr_ = self.poly5pow(
                rij=rs, 
                paraArray=paraArray,
                r0 = r0,
            )

            loss = (fr_ - frs).pow(2).mean()

            log.info("Decaying function fitting Step {}, loss: {:.4f}, lr: {:.5f}".format(i, loss.item(), lrsch.get_last_lr()[0]))
            loss.backward()
            optimizer.step()
            lrsch.step(loss.item())
        
        return shifts.detach()



    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        if not (self.has_scales or self.has_shifts):
            return data

        edge_center = data[AtomicDataDict.EDGE_INDEX_KEY][0]

        species_idx = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
        edge_atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[data[AtomicDataDict.EDGE_INDEX_KEY]]
        in_field = data[self.field]

        assert len(in_field) == len(
            edge_center
        ), "in_field doesnt seem to have correct per-edge shape"

        if self.has_scales:
            in_field = self.scales[species_idx][:,self.scale_index].view(-1, self.irreps_in.dim) * in_field
        if self.has_shifts:
            shifts = self.shifts[species_idx][:,self.shift_index[self.shift_index>=0]].view(-1, self.num_scalar, 7)
            r0 = self.r0[edge_atom_type].sum(0) * 0.5
            shifts = self.poly5pow(
                rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
                r0=r0,
                paraArray=shifts
            ) # [n_edge, n_scalar]
            in_field[:, self.shift_index>=0] = shifts + in_field[:, self.shift_index>=0]
        
        data[self.out_field] = in_field

        return data

    def poly5pow(self, rij, paraArray, r0:torch.Tensor):
        """> This function calculates SK integrals without the environment dependence of the form of powerlaw

                $$ h(rij) = alpha_1 * (rij / r_ij0)^(lambda + alpha_2) $$
        """

        #alpha1, alpha2, alpha3, alpha4 = paraArray[:, 0], paraArray[:, 1]**2, paraArray[:, 2]**2, paraArray[:, 3]**2
        alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7 = paraArray[..., 0], paraArray[..., 1], paraArray[..., 2], paraArray[..., 3], paraArray[..., 4], paraArray[..., 5], paraArray[..., 6].abs()
        #[N, n_op]
        shape = [-1]+[1] * (len(alpha1.shape)-1)
        # [-1, 1]
        rij = rij.reshape(shape)
        r0 = r0.reshape(shape)

        r0 = r0 / 1.8897259886

        return (alpha1 + alpha2 * (rij-r0) + 0.5 * alpha3 * (rij - r0)**2 + 1/6 * alpha4 * (rij-r0)**3 + 1./24 * alpha5 * (rij-r0)**4 + 1./120 * alpha6 * (rij-r0)**5) * (r0/rij)**(1 + alpha7)
    