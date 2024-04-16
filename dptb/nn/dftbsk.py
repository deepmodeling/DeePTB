import torch
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
from .sktb import OnsiteFormula
from dptb.nn.dftb.hopping_dptb import HoppingIntp
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.nn.dftb.sk_param import SKParam
import logging

log = logging.getLogger(__name__)


class DFTBSK(torch.nn.Module):
    name = "dftb"
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp_sk: Union[OrbitalMapper, None]=None,
            sk_path: str=None,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            **kwargs,
            ) -> None:
        
        super(DFTBSK, self).__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=self.device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk

        self.transform = transform
        self.basis = self.idp_sk.basis
        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.model_options = {
            "dftb":{
                "sk_path": sk_path         
                }
        }
        skparams = SKParam(basis=self.basis, sk_path=sk_path)
        self.skparams = skparams.skdict

        self.onsite_fn = OnsiteFormula(idp=self.idp_sk, functype='dftb', 
                                        dftb_onsiteE= self.skparams['OnsiteE'], dtype=dtype, device=device)
        self.hopping_fn = HoppingIntp(xdist=self.skparams['Distance'], num_ingrls=self.idp_sk.reduced_matrix_element, method='linear')

        self.hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, dtype=self.dtype, device=self.device, 
                                        strain=False,soc=False)
        if overlap:
            self.overlap = SKHamiltonian(idp_sk=self.idp_sk, onsite=False, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, node_field=AtomicDataDict.NODE_OVERLAP_KEY, dtype=self.dtype, device=self.device)

        self.idp = self.hamiltonian.idp

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
        edge_number = self.idp_sk.untransform_bond(edge_index).T
        rij = data[AtomicDataDict.EDGE_LENGTH_KEY]

        data[AtomicDataDict.EDGE_FEATURES_KEY] = torch.zeros((len(edge_index), self.idp_sk.reduced_matrix_element), dtype=self.dtype, device=self.device)
        
        if hasattr(self, "overlap"):
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = torch.zeros((len(edge_index), self.idp_sk.reduced_matrix_element), dtype=self.dtype, device=self.device)


        for ibtype in self.idp_sk.bond_types:

            mask = edge_index == self.idp_sk.bond_to_type[ibtype]
            
            hoptable = self.skparams['Hopping'][ibtype]
            data[AtomicDataDict.EDGE_FEATURES_KEY][mask] = self.hopping_fn.get_skhij(rij[mask], yy=hoptable)
            
            if hasattr(self, "overlap"):
                overlaptable = self.skparams['Overlap'][ibtype]
                data[AtomicDataDict.EDGE_OVERLAP_KEY][mask] = self.hopping_fn.get_skhij(rij[mask], yy=overlaptable)

        atomic_numbers = self.idp_sk.untransform_atom(data[AtomicDataDict.ATOM_TYPE_KEY].flatten())
        
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                atomic_numbers=atomic_numbers)
        
        if AtomicDataDict.NODE_SOC_SWITCH_KEY not in data:
            data[AtomicDataDict.NODE_SOC_SWITCH_KEY] =  torch.full((data['pbc'].shape[0], 1), False)
        else:
            data[AtomicDataDict.NODE_SOC_SWITCH_KEY].fill_(False)
        
        # sk param to hamiltonian and overlap
        if self.transform:
            data = self.hamiltonian(data)
            if hasattr(self, "overlap"):
                data = self.overlap(data)

        return data

