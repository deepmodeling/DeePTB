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
from .sktb.hopping import HoppingFormula
from .sktb.onsite import OnsiteFormula
from .sktb.bondlengthDB import bond_length_list
from dptb.utils.constants import atomic_num_dict_r

class SKTB(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]],
            onsite: str = "uniform",
            hopping: str = "powerlaw",
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            rc: Union[float, torch.Tensor] = 5.0,
            w: Union[float, torch.Tensor] = 1.0,
            ) -> None:
        
        super(SKTB, self).__init__()

        self.basis = basis
        self.idp = Index_Mapings_e3(basis, method="sktb")
        self.dtype = dtype
        self.device = device
        self.onsite = OnsiteFormula(functype=onsite)
        self.hopping = HoppingFormula(functype=hopping)
        self.overlap = HoppingFormula(functype=hopping, overlap=overlap)
        self.rc = rc
        self.w = w

        orbtype_count = self.idp.orbtype_count
        
        # init_onsite, hopping, overlap formula

        # init_param
        self.hopping_param = torch.nn.Parameter(torch.randn([len(self.idp.bondtype), self.idp.edge_reduced_matrix_element, self.hopping.num_paras], dtype=self.dtype, device=self.device))
        if overlap:
            self.overlap_param = torch.nn.Parameter(torch.randn([len(self.idp.bondtype), self.idp.edge_reduced_matrix_element, self.hopping.num_paras], dtype=self.dtype, device=self.device))

        if onsite == "strain":
            self.onsite_param = []
        elif onsite == "none":
            self.onsite_param = None
        else:
            self.onsite_param = torch.nn.Parameter(torch.randn([len(self.idp.atomtype), self.idp.node_reduced_matrix_element, self.onsite.num_paras], dtype=self.dtype, device=self.device))
        
        if onsite == "strain":
            # AB [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # AA [...]
            # but need to map to all pairs and all orbital pairs like AB, AA, BB, BA for [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # with this map: BA[sp, sd] = AB[ps, ds]
            self.strain_param = torch.nn.Parameter(torch.randn([len(self.idp.bondtype), self.idp.edge_reduced_matrix_element], dtype=self.dtype, device=self.device))
    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get the env and bond from the data
        # calculate the sk integrals
        # calculate the onsite
        # calculate the hopping
        # calculate the overlap
        # return the data with updated edge/node features

        # map the parameters to the edge/node/env features
        
        # compute integrals from parameters using hopping and onsite clas
        edge_type = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.EDGE_INDEX_KEY].flatten()].view(2, -1)
        edge_index = [self.idp.bondtype_map[atomic_num_dict_r(edge_type[:,i][0])+"-"+atomic_num_dict_r(edge_type[:,i][1])] for i in range(edge_type.shape[1])]
        edge_params = self.hopping_param[edge_index] # [N_edge, n_pairs, n_paras]
        r0 = 0.5*bond_length_list[data[AtomicDataDict.EDGE_INDEX_KEY].flatten()].view(2,-1).sum(0)
        data[AtomicDataDict.EDGE_FEATURES_KEY] = self.hopping.get_skhij(
            rij=data[AtomicDataDict.EDGE_LENGTH_KEY].unsqueeze(1).repeat(1, self.idp.edge_reduced_matrix_element).view(-1),
            paraArray=edge_params.view(-1, self.hopping.num_paras),
            rcut=self.rc,
            w=self.w,
            r0=r0.unsqueeze(1).repeat(1, self.idp.edge_reduced_matrix_element).view(-1)
            ).reshape(-1, self.idp.edge_reduced_matrix_element)
        
        if hasattr(self, "overlap"):
            edge_params = self.overlap_param[edge_index]
            self.overlap.getsksij()
            equal_orbpair = torch.zeros(self.idp.edge_reduced_matrix_element, dtype=self.dtype, device=self.device).view(1, -1)
            for orbpair_key, slices in self.idp.pair_maps.items():
                if orbpair_key.split("-")[0] == orbpair_key.split("-")[1]:
                    equal_orbpair[slices] = 1.0
            paraconst = edge_type[0].eq(edge_type[1]).float().view(-1, 1) @ equal_orbpair.unsqueeze(0)
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = self.hopping.get_skhij(
                rij=data[AtomicDataDict.EDGE_LENGTH_KEY].unsqueeze(1).repeat(1, self.idp.edge_reduced_matrix_element).view(-1),
                paraArray=edge_params.view(-1, self.hopping.num_paras),
                paraconst=paraconst.view(-1),
                rcut=self.rc,
                w=self.w,
                r0=r0.unsqueeze(1).repeat(1, self.idp.edge_reduced_matrix_element).view(-1)
                ).reshape(-1, self.idp.edge_reduced_matrix_element)

        if self.onsite.functype == "NRL":
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite.get_skEs(
                onsitenv_index=data[AtomicDataDict.ENV_INDEX_KEY], 
                onsitenv_length=data[AtomicDataDict.ENV_LENGTH_KEY], 
                nn_onsite_paras=self.onsite_param, 
                rcut=self.rc, 
                w=self.w
                )
        else:
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite.get_skEs(
                atype_list=data[AtomicDataDict.ATOMIC_NUMBERS_KEY], 
                otype_list=data[AtomicDataDict.ATOMIC_NUMBERS_KEY], 
                nn_onsite_paras=self.onsite_param
                )
        
        # compute strain
        if self.onsite.functype == "strain":
            onsitenv_type = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.ONSITENV_INDEX_KEY].flatten()].view(2, -1)
            onsitenv_index = torch.tensor([self.idp.bondtype_map.get(atomic_num_dict_r(onsitenv_type[:,i][0])+"-"+atomic_num_dict_r(onsitenv_type[:,i][1]), 
                            -self.idp.bondtype_map[atomic_num_dict_r(onsitenv_type[:,i][1])+"-"+atomic_num_dict_r(onsitenv_type[:,i][0])]) 
                            for i in range(onsitenv_type.shape[1])], dtype=torch.long, device=self.device)
            onsitenv_index[onsitenv_index<0] = -onsitenv_index[onsitenv_index<0] + len(self.idp.bondtype)
            onsitenv_params = torch.stack([self.strain_param, 
                self.strain_param.reshape(-1, len(self.idp.full_basis), len(self.idp.full_basis)).transpose(1,2).reshape(len(self.idp.bondtype), -1)], dim=1)
            data[AtomicDataDict.ONSITENV_FEATURES_KEY] = onsitenv_params[onsitenv_index]

        return data

        
