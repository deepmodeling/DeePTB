import torch
from dptb.utils.constants import h_all_types, anglrMId, atomic_num_dict, atomic_num_dict_r
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import re
from dptb.utils.tools import float2comlex


class HR2HK(torch.nn.Module):
    # this is actually a general FFT from real space hamiltonian/overlap to kspace hamiltonian/overlap
    # the more correct name should be HSR2HSK. But to keep consistent with previous naming convention, we still use HR2HK here.
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            edge_field: str = AtomicDataDict.EDGE_FEATURES_KEY,
            node_field: str = AtomicDataDict.NODE_FEATURES_KEY,
            out_field: str = AtomicDataDict.HAMILTONIAN_KEY,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            derivative:bool = False,
            out_derivative_field: str = AtomicDataDict.HAMILTONIAN_DERIV_KEY,
            gauge: bool = False 
            ):
        # gauge: False -> Tight-binding Convention I:  Wannier90 Gauge 
        # gauge: True  -> Tight-binding Convention II: "Physical Gauge"/"Periodic Gauge"
        super(HR2HK, self).__init__()
    
        if derivative:
            gauge = True
        self.gauge = gauge
        self.derivative = derivative
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
        self.out_derivative_field = out_derivative_field

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:

        # construct bond wise hamiltonian block from obital pair wise node/edge features
        # we assume the edge feature have the similar format as the node feature, which is reduced from orbitals index oj-oi with j>i
        
        # Ensure edge_vectors are computed if using gauge mode
        if self.gauge:
            data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        
        orbpair_hopping = data[self.edge_field]
        orbpair_onsite = data.get(self.node_field)
        bondwise_hopping = torch.zeros((len(orbpair_hopping), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.dtype, device=self.device)
        bondwise_hopping.to(self.device)
        bondwise_hopping.type(self.dtype)
        onsite_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
        kpoints = data[AtomicDataDict.KPOINT_KEY]
        if kpoints.is_nested:
            assert kpoints.size(0) == 1
            kpoints = kpoints[0]

        soc = data.get(AtomicDataDict.NODE_SOC_SWITCH_KEY, False)
        if isinstance(soc, torch.Tensor):
            soc = soc.all()
        if soc: 
            # this soc only support sktb.
            orbpair_soc = data[AtomicDataDict.NODE_SOC_KEY]
            soc_upup_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)
            soc_updn_block = torch.zeros((len(data[AtomicDataDict.ATOM_TYPE_KEY]), self.idp.full_basis_norb, self.idp.full_basis_norb), dtype=self.ctype, device=self.device)

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
                    bondwise_hopping[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_hopping[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                    onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)

                if soc and i==j and not self.overlap:
                        # For now, The SOC part is only added to Hamiltonian, not overlap matrix.
                        # For now, The SOC only has onsite contribution.
                        soc_updn_tmp = orbpair_soc[:,self.idp.orbpair_soc_maps[orbpair]].reshape(-1, 2*li+1, 2*(2*lj+1))
                        soc_upup_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,:2*lj+1]
                        soc_updn_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,2*lj+1:]

                # constructing onsite blocks
                #if self.overlap:
                #    # if iorb == jorb:
                #    #     onsite_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = factor * torch.eye(2*li+1, dtype=self.dtype, device=self.device).reshape(1, 2*li+1, 2*lj+1).repeat(onsite_block.shape[0], 1, 1)
                #    if i <= j:
                #        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                #    if soc and i == j:
                #        soc_updn_tmp = orbpair_soc[:, self.idp.orbpair_soc_maps[orbpair]].reshape(-1, 2*li+1, 2*(2*lj+1))
                #        # j==i -> 2*lj+1 == 2*li+1
                #        soc_upup_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1, :2*lj+1]
                #        soc_updn_block[:, ist:ist+2*li+1, jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1, 2*lj+1:]
                #else:
                #    if i <= j:
                #        onsite_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = factor * orbpair_onsite[:,self.idp.orbpair_maps[orbpair]].reshape(-1, 2*li+1, 2*lj+1)
                #    
                #    if soc and i==j:
                #        soc_updn_tmp = orbpair_soc[:,self.idp.orbpair_soc_maps[orbpair]].reshape(-1, 2*li+1, 2*(2*lj+1))
                #        soc_upup_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,:2*lj+1]
                #        soc_updn_block[:,ist:ist+2*li+1,jst:jst+2*lj+1] = soc_updn_tmp[:, :2*li+1,2*lj+1:]
                
                jst += 2*lj+1
            ist += 2*li+1
        self.onsite_block = onsite_block
        self.bondwise_hopping = bondwise_hopping
        if soc and not self.overlap:
            # store for later use
            # for now, soc only contribute to Hamiltonain, thus for overlap not store soc parts.
            self.soc_upup_block = soc_upup_block
            self.soc_updn_block = soc_updn_block

        # R2K procedure can be done for all kpoint at once.
        all_norb = self.idp.atom_norb[data[AtomicDataDict.ATOM_TYPE_KEY]].sum()
        block = torch.zeros(kpoints.shape[0], all_norb, all_norb, dtype=self.ctype, device=self.device)
        
        # Initialize derivative blocks if needed: dH/dk = [dH/dkx, dH/dky, dH/dkz]
        if self.derivative:
            dblock = torch.zeros(kpoints.shape[0], all_norb, all_norb, 3, dtype=self.ctype, device=self.device)
        
        atom_id_to_indices = {}
        ist = 0
        for i, oblock in enumerate(onsite_block):
            mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
            masked_oblock = oblock[mask][:,mask]
            block[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1]] = masked_oblock.squeeze(0)
            #if self.derivative:
            #    for alpha in range(3):
            #        dblock[:,ist:ist+masked_oblock.shape[0],ist:ist+masked_oblock.shape[1],alpha] = masked_oblock.squeeze(0)
            atom_id_to_indices[i] = slice(ist, ist+masked_oblock.shape[0])
            ist += masked_oblock.shape[0]
    
        for i, hblock in enumerate(bondwise_hopping):
            iatom = data[AtomicDataDict.EDGE_INDEX_KEY][0][i]
            jatom = data[AtomicDataDict.EDGE_INDEX_KEY][1][i]
            iatom_indices = atom_id_to_indices[int(iatom)]
            jatom_indices = atom_id_to_indices[int(jatom)]
            imask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[iatom]]
            jmask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[jatom]]
            masked_hblock = hblock[imask][:,jmask]

            if self.gauge:
                # phase factor according to convention II
                # k and R are in fractional coordinates, need to convert to cartesian
                edge_vec = data[AtomicDataDict.EDGE_VECTORS_KEY][i]  # Cartesian coordinates
                phase_factor = torch.exp(-1j * 2 * torch.pi * (
                    kpoints @ data[AtomicDataDict.CELL_KEY].inverse().T @ edge_vec)).reshape(-1,1,1)
                
                # Compute derivative: dH/dk_alpha = -i * R_alpha * H_R * exp(-i k·R)
                # where R is edge_vec in Cartesian coordinates
                if self.derivative:
                    # derivative_factor shape: [n_kpoints, 1, 1, 3]
                    # - i * R * exp(-i k·R) = -i * R * phase_factor
                    derivative_factor = (-1.0j * edge_vec).reshape(1, 1, 1, 3) * phase_factor.unsqueeze(-1)
            else:
                phase_factor = torch.exp(-1j * 2 * torch.pi * (
                    kpoints @ data[AtomicDataDict.EDGE_CELL_SHIFT_KEY][i])).reshape(-1,1,1)
                
            block[:,iatom_indices,jatom_indices] += masked_hblock.squeeze(0).type_as(block) * phase_factor
            
            if self.derivative and self.gauge:
                # Add derivative contribution
                dblock[:,iatom_indices,jatom_indices,:] += masked_hblock.squeeze(0).type_as(dblock).unsqueeze(-1) * derivative_factor

        block = block + block.transpose(1,2).conj()
        block = block.contiguous()
        
        # Hermitianize derivative blocks: dH/dk should also be Hermitian
        if self.derivative:
            for alpha in range(3):
                dblock[:,:,:,alpha] = dblock[:,:,:,alpha] + dblock[:,:,:,alpha].transpose(1,2).conj()
            dblock = dblock.contiguous()
        
        if soc:
            if self.overlap:
                # ========== S_soc = S ⊗ I₂ : N×N S(k) to 2N×2N kronecker product ==========
                S_soc = torch.zeros(kpoints.shape[0], 2*all_norb, 2*all_norb, dtype=self.ctype, device=self.device)
                S_soc[:, :all_norb, :all_norb] = block
                S_soc[:, all_norb:, all_norb:] = block
                # Enforce strict Hermitian form to avoid non-positive-definite errors during training by torch._C._LinAlgError: linalg.cholesky.
                # This issue only occurs when SOC+overlap is active and "overlap" is not frozen. It can be avoided by setting "freeze": ["overlap"].
                S_soc = 0.5 * (S_soc + S_soc.transpose(1, 2).conj())

                data[self.out_field] = S_soc
            else:
                HK_SOC = torch.zeros(kpoints.shape[0], 2*all_norb, 2*all_norb, dtype=self.ctype, device=self.device)
                ist = 0
                assert len(soc_upup_block) == len(soc_updn_block)
                for i in range(len(soc_upup_block)):
                    assert soc_upup_block[i].shape == soc_updn_block[i].shape
                    mask = self.idp.mask_to_basis[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()[i]]
                    masked_soc_upup_block = soc_upup_block[i][mask][:,mask]
                    masked_soc_updn_block = soc_updn_block[i][mask][:,mask]
                    HK_SOC[:,ist:ist+masked_soc_upup_block.shape[0],ist:ist+masked_soc_upup_block.shape[1]] = masked_soc_upup_block.squeeze(0)
                    HK_SOC[:,ist:ist+masked_soc_updn_block.shape[0],ist+all_norb:ist+all_norb+masked_soc_updn_block.shape[1]] = masked_soc_updn_block.squeeze(0)
                    assert masked_soc_upup_block.shape[0] == masked_soc_upup_block.shape[1]
                    assert masked_soc_upup_block.shape[0] == masked_soc_updn_block.shape[0]
                    
                    ist += masked_soc_upup_block.shape[0]

                HK_SOC[:,all_norb:,:all_norb] = HK_SOC[:,:all_norb,all_norb:].conj()   
                HK_SOC[:,all_norb:,all_norb:] = HK_SOC[:,:all_norb,:all_norb].conj()  + block
                HK_SOC[:,:all_norb,:all_norb] = HK_SOC[:,:all_norb,:all_norb] + block  

                data[self.out_field] = HK_SOC
        else:
            data[self.out_field] = block
        
        # Store derivative if computed
        if self.derivative:
            data[self.out_derivative_field] = dblock

        return data
    