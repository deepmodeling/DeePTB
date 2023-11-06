"""This file refactor the SK and E3 Rotation in dptb/hamiltonian/transform_se3.py], it will take input of AtomicDataDict.Type
    perform rotation from irreducible matrix element / sk parameters in EDGE/NODE FEATURE, and output the atomwise/ pairwise hamiltonian
    as the new EDGE/NODE FEATURE. The rotation should also be a GNN module and speed uptable by JIT. The HR2HK should also be included here.
    The indexmapping should ne passed here.
    """

import torch
from e3nn.o3 import wigner_3j, Irrep, xyz_to_angles, Irrep
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.utils.index_mapping import Index_Mapings_e3
from dptb.data import AtomicDataDict


class E3Hamiltonian(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]],
            decompose: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ) -> None:
        super(E3Hamiltonian, self).__init__()
        self.dtype = dtype
        self.device = device
        self.idp = Index_Mapings_e3(basis)
        self.basis = self.idp.basis
        self.cgbasis = {}
        self.decompose = decompose

        # initialize the CG basis
        self.idp.get_pairtype_maps()
        pairtypes = self.idp.pairtype_maps.keys()
        for pairtype in pairtypes:
            self._initialize_CG_basis(pairtype)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        assert data[AtomicDataDict.EDGE_FEATURES_KEY].shape[1] == self.idp.bond_reduced_matrix_element

        n_edge = data[AtomicDataDict.EDGE_INDEX_KEY].shape[1]

        if not self.decompose:
            # The EDGE_FEATURES_KEY and NODE_FAETURE_KEY are the reduced matrix elements

            # compute hopping blocks
            for pairtype in self.idp.pairtype_maps.keys():
                # currently, "a-b" and "b-a" orbital pair are computed seperately, it is able to combined further
                # for better performance
                l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]
                n_rme = (2*l1+1) * (2*l2+1) # number of reduced matrix element
                rme = data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[pairtype]]
                rme = rme.reshape(n_edge, -1, n_rme)
                rme = rme.transpose(1,2) # shape (N, n_rme, n_pair)
                
                H_z = torch.sum(self.cgbasis[pairtype][None,:,:,:,None] * \
                    rme[:,None, None, :, :], dim=-2) # shape (N, 2l1+1, 2l2+1, n_pair)
                
                # rotation
                angle = xyz_to_angles(data[AtomicDataDict.EDGE_VECTORS_KEY]) # (tensor(N), tensor(N))
                rot_mat_L = Irrep(int(l1), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l1+1, 2l1+1)
                rot_mat_R = Irrep(int(l2), 1).D_from_angles(angle[0], angle[1], torch.tensor(0., dtype=self.dtype, device=self.device)) # tensor(N, 2l2+1, 2l2+1)
                HR = torch.einsum("nlm, nmoq, nko -> nqlk", rot_mat_L, H_z, rot_mat_R).reshape(n_edge, -1) # shape (N, n_pair * n_rme)
                
                data[AtomicDataDict.EDGE_FEATURES_KEY][:, self.idp.pairtype_maps[pairtype]] = HR
        
        else:
            # The EDGE_FEATURES_KEY and NODE_FAETURE_KEY are the hamiltonian matrix in z-axis direction
            pass

        return data
            
    def _initialize_CG_basis(self, pairtype: str):
        self.cgbasis.setdefault(pairtype, None)

        l1, l2 = anglrMId[pairtype[0]], anglrMId[pairtype[2]]

        cg = []
        for l_ird in range(abs(l2-l1), l2+l1+1):
            cg.append(wigner_3j(int(l1), int(l2), int(l_ird), dtype=self.dtype, device=self.device) * (2*l_ird+1)**0.5)
        
        cg = torch.cat(cg, dim=-1)
        self.cgbasis[pairtype] = cg

        return cg