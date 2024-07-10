#from hr2dhk import Hr2dHk
import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from dptb.utils.tools import float2comlex

class HR2dHk(torch.nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = 'None',
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            ):
        super(HR2dHk, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.overlap = overlap
        self.ctype = float2comlex(dtype)

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            assert idp.method == "e3tb", "The method of idp should be e3tb."
            self.idp = idp
        
        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()
        self.idp.get_orbpair_soc_maps()

        self.edge_field = edge_field
        self.node_field = node_field
        self.out_field = out_field

    def forward(self, data: AtomicDataDict.Type, direction = 'xyz') -> AtomicDataDict.Type:
        dir2ind = {'x':0, 'y':1, 'z':2}

        uniq_direcs = list(set(direction))
        assert len(uniq_direcs) > 0, "direction should be provided."
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)

        bondwise_dh = {}
        for idirec in uniq_direcs:
            assert idirec in dir2ind, "direction should be x, y or z."
            bondwise_dh[idirec] = torch.zeros((len(orbpair_hopping), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.dtype, device=self.device)

        dr_ang = data[AtomicDataDict.EDGE_VECTORS_KEY]
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
        
        ist = 0
        for i,iorb in enumerate(self.idp.full_basis):
            jst = 0
            li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
            for j,jorb in enumerate(self.idp.full_basis):
                orbpair = iorb + "-" + jorb
                lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                
                # constructing hopping blocks
                if iorb == jorb:
                    factor = 0.5
                else:
                    factor = 1.0

                if i <= j:
                    # note: we didnot consider the factor 1.0j here. we will multiply it later.
                    # -1j * R_ij * H_ij(R_ij) * exp(-i2pi k.R_ij)
                    for idirec in uniq_direcs:
                        bondwise_dh[idirec][:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * ( -1 * dr_ang[:,[dir2ind[idirec]]] * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]]).reshape(-1, 2*li+1, 2*lj+1)
                if self.overlap:
                    raise NotImplementedError("Overlap is not implemented for dHk yet.")
                else:
                    if i <= j:
                        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                jst += 2*lj+1
            ist += 2*li+1
        self.onsite_block = onsite_block
        self.bondwise_dh = bondwise_dh
        
        # R2K procedure can be done for all kpoint at once.
        all_norb = self.idp.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum()


        dHdk = {}
        for idirec in uniq_direcs:
            dHdk[idirec] = torch.zeros(data[AtomicDataDict.KPOINT_KEY][0].shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)

        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            for idirec in uniq_direcs:
                dHdk[idirec][:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = masked_oblock.squeeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            ist += masked_oblock.shape[0]
       
        for i in range (len(bondwise_dh[uniq_direcs[0]])):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            iatom_indices = atom_id_to_indices[int(iatom)]
            jatom_indices = atom_id_to_indices[int(jatom)]
            imask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[iatom]]
            jmask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[jatom]]

            
            for idirec in uniq_direcs:        
                hblock = bondwise_dh[idirec][i]
                masked_hblock = hblock[imask][:,jmask]
                dHdk[idirec][:,iatom_indices,jatom_indices] += 1.0j  *masked_hblock.squeeze(0).type_as(dHdk[idirec]) * \
                    torch.exp(-1j * 2 * torch.pi * (data[AtomicDataDict.KPOINT_KEY][0] @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)
                
        for idirec in uniq_direcs:    
            dHdk[idirec] = dHdk[idirec] + dHdk[idirec].conj().transpose(1,2)

        return dHdk
    