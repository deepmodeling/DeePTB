from typing import Optional

import torch
import torch.nn.functional
from ase.data import atomic_numbers
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper

class OneHotAtomEncoding(torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        universal: Optional[bool] = False,
        idp: Optional[OrbitalMapper] = None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        self.idp = idp
        self.universal = universal

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        if self.universal:
            atomic_numbers = self.idp.untransform_atom(type_numbers)
            one_hot = torch.nn.functional.one_hot(
                atomic_numbers, num_classes=95
            ).to(device=atomic_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        else:
            one_hot = torch.nn.functional.one_hot(
                type_numbers, num_classes=self.num_types
            ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data
