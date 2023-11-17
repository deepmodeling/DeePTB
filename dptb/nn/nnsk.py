"""The file doing the process from the fitting net output sk formula parameters in node/edge feature to the tight binding two centre integrals parameters in node/edge feature.
in: Data
out Data

basically a map from a matrix parameters to edge/node features, or strain mode's environment edge features
"""

import torch
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import numpy as np
from .sktb import OnsiteFormula, bond_length_list, HoppingFormula
from dptb.utils.constants import atomic_num_dict_r
from dptb.nn.hamiltonian import SKHamiltonian

class NNSK(torch.nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            onsite: str = "uniform",
            hopping: str = "powerlaw",
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            rc: Union[float, torch.Tensor] = 5.0,
            w: Union[float, torch.Tensor] = 1.0,
            **kwargs,
            ) -> None:
        
        super(NNSK, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="sktb")
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_node_maps()
        self.idp.get_pair_maps()


        # init_onsite, hopping, overlap formula

        self.onsite_fn = OnsiteFormula(idp=self.idp, functype=onsite, dtype=dtype, device=device)
        self.hopping_fn = HoppingFormula(functype=hopping)
        if overlap:
            self.overlap_fn = HoppingFormula(functype=hopping, overlap=True)
        self.rc = rc
        self.w = w

        print(overlap)
        
        # init_param
        self.hopping_param = torch.nn.Parameter(torch.randn([len(self.idp.reduced_bond_types), self.idp.edge_reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device))
        if overlap:
            self.overlap_param = torch.nn.Parameter(torch.randn([len(self.idp.reduced_bond_types), self.idp.edge_reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device))

        if onsite == "strain":
            self.onsite_param = []
        elif onsite == "none":
            self.onsite_param = None
        else:
            self.onsite_param = torch.nn.Parameter(torch.randn([len(self.idp.type_names), self.idp.node_reduced_matrix_element, self.onsite_fn.num_paras], dtype=self.dtype, device=self.device))
        
        if onsite == "strain":
            # AB [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # AA [...]
            # but need to map to all pairs and all orbital pairs like AB, AA, BB, BA for [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # with this map: BA[sp, sd] = AB[ps, ds]
            self.strain_param = torch.nn.Parameter(torch.randn([len(self.idp.reduced_bond_types), self.idp.edge_reduced_matrix_element], dtype=self.dtype, device=self.device))
        self.hamiltonian = SKHamiltonian(idp=self.idp, dtype=self.dtype, device=self.device)
        if overlap:
            self.overlap = SKHamiltonian(idp=self.idp, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, node_field=AtomicDataDict.NODE_OVERLAP_KEY, dtype=self.dtype, device=self.device)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get the env and bond from the data
        # calculate the sk integrals
        # calculate the onsite
        # calculate the hopping
        # calculate the overlap
        # return the data with updated edge/node features

        # map the parameters to the edge/node/env features
        
        # compute integrals from parameters using hopping and onsite clas
        edge_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.EDGE_INDEX_KEY]].reshape(2, -1)
        edge_index = self.idp.transform_reduced_bond(*edge_number)
        r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[edge_number].sum(0)
        data[AtomicDataDict.EDGE_FEATURES_KEY] = self.hopping_fn.get_skhij(
            rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
            paraArray=self.hopping_param[edge_index], # [N_edge, n_pairs, n_paras],
            rcut=self.rc,
            w=self.w,
            r0=r0
            ) # [N_edge, n_pairs]

        if hasattr(self, "overlap"):
            equal_orbpair = torch.zeros(self.idp.edge_reduced_matrix_element, dtype=self.dtype, device=self.device)
            for orbpair_key, slices in self.idp.pair_maps.items():
                if orbpair_key.split("-")[0] == orbpair_key.split("-")[1]:
                    equal_orbpair[slices] = 1.0
            paraconst = edge_number[0].eq(edge_number[1]).float().view(-1, 1) * equal_orbpair.unsqueeze(0)

            data[AtomicDataDict.EDGE_OVERLAP_KEY] = self.overlap_fn.get_sksij(
                rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
                paraArray=self.overlap_param[edge_index],
                paraconst=paraconst,
                rcut=self.rc,
                w=self.w,
                r0=r0,
                )

        if self.onsite_fn.functype == "NRL":
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                atomic_numbers=data[AtomicDataDict.ATOMIC_NUMBERS_KEY],
                onsitenv_index=data[AtomicDataDict.ONSITENV_INDEX_KEY], 
                onsitenv_length=data[AtomicDataDict.ONSITENV_LENGTH_KEY], 
                nn_onsite_paras=self.onsite_param, 
                rcut=self.rc, 
                w=self.w
                )
        else:
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                atomic_numbers=data[AtomicDataDict.ATOMIC_NUMBERS_KEY], 
                nn_onsite_paras=self.onsite_param
                )
            
        if hasattr(self, "overlap"):
            data[AtomicDataDict.NODE_OVERLAP_KEY] = torch.ones_like(data[AtomicDataDict.NODE_OVERLAP_KEY])
        
        # compute strain
        if self.onsite_fn.functype == "strain":
            onsitenv_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.ONSITENV_INDEX_KEY]].reshape(2,-1)
            onsitenv_index = self.idp.transform_reduced_bond(*onsitenv_number)
            reflect_index = self.idp.transform_reduced_bond(*onsitenv_number.flip(0))
            onsitenv_index[onsitenv_index<0] = reflect_index[onsitenv_index<0] + len(self.idp.reduced_bond_types)

            # be careful here cause I am not so sure about whether this keys of pair maps has the correct order, 
            # but it works for now
            reflect_keys = np.array(list(self.idp.pair_maps.keys()), dtype="str").reshape(len(self.idp.full_basis), len(self.idp.full_basis)).transpose(1,0).reshape(-1)
            reflect_params = torch.cat([self.strain_param[:,self.idp.pair_maps[key]] for key in reflect_keys], dim=-1)
            onsitenv_params = torch.cat([self.strain_param, 
                reflect_params], dim=0)
            
            data[AtomicDataDict.ONSITENV_FEATURES_KEY] = onsitenv_params[onsitenv_index]

        # sk param to hamiltonian and overlap
        data = self.hamiltonian(data)
        if hasattr(self, "overlap"):
            data = self.overlap(data)
        
        return data
    
    @classmethod
    def from_reference(cls, checkpoint, nnsk_options: Dict=None):
        # the mapping from the parameters of the ref_model and the current model can be found using
        # reference model's idp and current idp
        pass

    @classmethod
    def from_model_v1(self, v1_model: torch.nn.Module, nnsk_options):
        # could support json file and .pth file checkpoint of nnsk
        pass

        
