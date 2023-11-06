"""The file doing the process from the fitting net output sk formula parameters in node/edge feature to the tight binding two centre integrals parameters in node/edge feature.
in: Data
out Data

basically a map from a matrix parameters to edge/node features, or strain mode's environment edge features
"""
import torch
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.utils.index_mapping import Index_Mapings_e3
from dptb.data import AtomicDataDict

class SKTB(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]],
            onsite: str = "uniform",
            hopping: str = "powerlaw",
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ) -> None:
        
        super(SKTB, self).__init__()

        self.basis = basis
        self.idp = Index_Mapings_e3(basis)
        self.dtype = dtype
        self.device = device
        self.overlap = overlap
        self.onsite = onsite
        self.hopping = hopping

        orbtype_count = self.idp.orbtype_count
        self.n_skintegrals =1 * (orbtype_count["s"] * orbtype_count["s"] + \
                                2*orbtype_count["s"] * orbtype_count["p"] + \
                                2*orbtype_count["s"] * orbtype_count["d"] + \
                                2*orbtype_count["s"] * orbtype_count["f"]) + \
                            2 * (orbtype_count["p"] * orbtype_count["p"] + \
                                2*orbtype_count["p"] * orbtype_count["d"] + \
                                2*orbtype_count["p"] * orbtype_count["f"]) + \
                            3 * (orbtype_count["d"] * orbtype_count["d"] + \
                                2*orbtype_count["d"] * orbtype_count["f"]) + \
                            4 * (orbtype_count["f"] * orbtype_count["f"])
        
        # init_onsite, hopping, overlap formula

        # init_param
        self.hopping_param = torch.nn.Parameter(torch.randn([len(self.idp.bondtype), self.n_skintegrals, n_formula], dtype=self.dtype, device=self.device))
        if overlap:
            self.overlap_param = torch.nn.Parameter(torch.randn([len(self.idp.bondtype), self.n_skintegrals, n_formula], dtype=self.dtype, device=self.device))

        self.onsite_param = []

    def forward(data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get the env and bond from the data
        # calculate the sk integrals
        # calculate the onsite
        # calculate the hopping
        # calculate the overlap
        # return the data with updated edge/node features
        pass
        
