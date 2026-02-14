import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from dptb.utils.tools import float2comlex, tdtype2ndtype
import numpy as np
try:
    from vbcsr import ImageContainer
    from vbcsr import AtomicData as AtomicData_vbcsr
except ImportError:
    raise ImportError("Please install vbcsr first: pip install vbcsr")

class Hr2HR:
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu")
            ):

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

    def __call__(self, data: AtomicDataDict.Type):

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i
        if AtomicDataDict.ATOM_TYPE_KEY not in data:
            self.idp(data)
        
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        natom = orbpair_onsite.shape[0]
        nedge = orbpair_hopping.shape[0]
        
        bondwise_hopping = torch.zeros((nedge, self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.dtype, device=self.device)
        onsite_block = torch.zeros((natom, self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)

        atom_type = data[AtomicDataDict.ATOM_TYPE_KEY].flatten()
        edge_shift_vec = data[AtomicDataDict.EDGE_CELL_SHIFT_KEY]

        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        ndtype = np.float64
        if isinstance(soc, torch.Tensor):
            soc = soc.all()

        if soc: 
            # this soc only support sktb.
            orbpair_soc = data[AtomicDataDict.NODE_SOC_KEY]
            soc_upup_block = torch.zeros((natom, self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)
            soc_updn_block = torch.zeros((natom, self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)
            ndtype = np.complex128
        
        spin_factor = 2 if soc else 1

        with torch.no_grad():
            ist = 0
            for i,iorb in enumerate(self.idp.full_basis):
                jst = 0
                li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
                for j,jorb in enumerate(self.idp.full_basis):
                    orbpair = iorb + "-" + jorb
                    lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                    
                    # constructing hopping blocks
                    if iorb == jorb:
                        factor = 1.0
                    else:
                        factor = 2.0

                    if i <= j:
                        bondwise_hopping[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)

                    if soc and i==j and not self.overlap:
                        # For now, The SOC part is only added to Hamiltonian, not overlap matrix.
                        # For now, The SOC only has onsite contribution.
                        soc_updn_tmp = orbpair_soc[:,self.idp.orbpair_soc_maps[orbpair]].reshape(-1, 2*li+1, 2*(2*lj+1))
                        soc_upup_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,:2*lj+1]
                        soc_updn_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,2*lj+1:]
                    
                    jst += 2*lj+1
                ist += 2*li+1
        onsite_block = onsite_block.to("cpu")
        bondwise_hopping = bondwise_hopping.to("cpu")
        if soc and not self.overlap:
            # store for later use
            # for now, soc only contribute to Hamiltonain, thus for overlap not store soc parts.
            soc_upup_block = soc_upup_block.to("cpu")
            soc_updn_block = soc_updn_block.to("cpu")

        adata = AtomicData_vbcsr.from_distributed(
            natom, natom, 0, nedge, nedge,
            list(range(natom)), data[AtomicDataDict.ATOM_TYPE_KEY], data[AtomicDataDict.EDGE_INDEX_KEY].T, self.idp.atom_norb, data[AtomicDataDict.EDGE_CELL_SHIFT_KEY],
            data[AtomicDataDict.CELL_KEY], data[AtomicDataDict.POSITIONS_KEY]
        )
        image_container = ImageContainer(adata, ndtype)

        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            nrol, ncol = masked_oblock.shape
            full_block = np.zeros([nrol*spin_factor, ncol*spin_factor], dtype=ndtype)
            if soc:
                full_block[:nrol,:ncol] = masked_oblock
                full_block[nrol:,ncol:] = masked_oblock
                if not self.overlap:
                    full_block[:nrol,ncol:] = soc_updn_block[i,mask][:,mask]
                    full_block[nrol:,:ncol] = soc_updn_block[i,mask][:,mask].conj()
                    full_block[:nrol,:ncol] += soc_upup_block[i,mask][:,mask]
                    full_block[nrol:,ncol:] += soc_upup_block[i,mask][:,mask].conj()
            else:
                full_block[:nrol,:ncol] = masked_oblock
            full_block = np.ascontiguousarray(full_block)
            
            image_container.add_block(g_row=i, g_col=i, data=full_block, R=None, mode="insert")
    
        for i, hblock in enumerate(bondwise_hopping):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            imask = self.idp.mask_to_basis[atom_type[iatom]]
            jmask = self.idp.mask_to_basis[atom_type[jatom]]
            masked_hblock = hblock[imask][:,jmask]
            nrol, ncol = masked_hblock.shape
            full_block = np.zeros([nrol*spin_factor, ncol*spin_factor], dtype=ndtype)
            if soc:
                full_block[:nrol,:ncol] = masked_hblock
                full_block[nrol:,ncol:] = masked_hblock
            else:
                full_block[:nrol,:ncol] = masked_hblock
            full_block = np.ascontiguousarray(full_block)
            
            image_container.add_block(g_row=iatom, g_col=jatom, data=full_block, R=edge_shift_vec[i], mode="insert")
        
        image_container.assemble()

        return image_container