import torch
import torch as th
import numpy as np
import logging
import re
from dptb.hamiltonian.transform_sk_speed import RotationSK
from dptb.hamiltonian.transform_se3 import RotationSE3
from dptb.nnsktb.formula import SKFormula
from dptb.utils.constants import anglrMId, atomic_num_dict
from dptb.hamiltonian.soc import creat_basis_lm, get_soc_matrix_cubic_basis

import matplotlib.pyplot as plt

''' Over use of different index system cause the symbols and type and index kind of object need to be recalculated in different 
Class, this makes entanglement of classes difficult. Need to design an consistent index system to resolve.'''

log = logging.getLogger(__name__)

class HamilEig(RotationSE3):
    """ This module is to build the Hamiltonian from the SK-type bond integral.
    """
    def __init__(self, dtype=torch.float32, device='cpu') -> None:
        super().__init__(rot_type=dtype, device=device)
        self.dtype = dtype
        if self.dtype is th.float32:
            self.cdtype = th.complex64
        elif self.dtype is th.float64:
            self.cdtype = th.complex128
        self.use_orthogonal_basis = False
        self.hamil_blocks = None
        self.overlap_blocks = None
        self.device = device

    def update_hs_list(self, struct, hoppings, onsiteEs, onsiteVs=None, overlaps=None, onsiteSs=None, soc_lambdas=None, **options):
        '''It updates the bond structure, bond type, bond type id, bond hopping, bond onsite, hopping, onsite
        energy, overlap, and onsite spin
        
        Parameters
        ----------
        hoppings
            a list bond integral for hoppings.
        onsiteEs
            a list of onsite energy for each atom and each orbital.
        overlaps
            a list bond integral for overlaps.
        onsiteSs
            a list of onsite overlaps for each atom and each orbital.
        '''
        self.__struct__ = struct
        self.hoppings = hoppings
        self.onsiteEs = onsiteEs
        self.onsiteVs = onsiteVs
        self.soc_lambdas = soc_lambdas
        self.use_orthogonal_basis = False
        if overlaps is None:
            self.use_orthogonal_basis = True
        else:
            self.overlaps = overlaps
            self.onsiteSs = onsiteSs
            self.use_orthogonal_basis = False
        
        if onsiteSs is not None:
            log.info(msg='The onsiteSs is not None, But even for non-orthogonal basis, the onsite S matrix part is still identity.')
            log.info(msg='Therefore the onsiteSs will not be used !!')
            
        if soc_lambdas is None:
            self.soc = False
        else:
            self.soc = True

        self.num_orbs_per_atom = []
        for itype in self.__struct__.proj_atom_symbols:
            norbs = self.__struct__.proj_atomtype_norbs[itype]
            self.num_orbs_per_atom.append(norbs)

    def get_soc_block(self, bonds_onsite = None):
        numOrbs = np.array(self.num_orbs_per_atom)
        totalOrbs = np.sum(numOrbs)
        if bonds_onsite is None:
            _, bonds_onsite = self.__struct__.get_bond()

        soc_upup = torch.zeros((totalOrbs, totalOrbs), device=self.device, dtype=self.cdtype)
        soc_updown = torch.zeros((totalOrbs, totalOrbs), device=self.device, dtype=self.cdtype)

        # compute soc mat for each atom:
        soc_atom_upup = getattr(self.__struct__, "soc_atom_upup") if hasattr(self.__struct__, "soc_atom_upup") else {}
        soc_atom_updown = getattr(self.__struct__, "soc_atom_updown") if hasattr(self.__struct__, "soc_atom_updown") else {}
        if not soc_atom_upup or not soc_atom_updown:
            for iatype in self.__struct__.proj_atomtype:
                total_num_orbs_iatom= self.__struct__.proj_atomtype_norbs[iatype]
                tmp_upup = torch.zeros([total_num_orbs_iatom, total_num_orbs_iatom], dtype=self.cdtype, device=self.device)
                tmp_updown = torch.zeros([total_num_orbs_iatom, total_num_orbs_iatom], dtype=self.cdtype, device=self.device)

                ist = 0
                for ish in self.__struct__.proj_atom_anglr_m[iatype]:
                    ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                    shidi = anglrMId[ishsymbol]          # 0,1,2,...
                    norbi = 2*shidi + 1

                    soc_orb = get_soc_matrix_cubic_basis(orbital=ishsymbol, device=self.device, dtype=self.dtype)
                    if len(soc_orb) != 2*norbi:
                        log.error(msg='The dimension of the soc_orb is not correct!')
                    tmp_upup[ist:ist+norbi, ist:ist+norbi] = soc_orb[:norbi,:norbi]
                    tmp_updown[ist:ist+norbi, ist:ist+norbi] = soc_orb[:norbi, norbi:]
                    ist = ist + norbi

                soc_atom_upup.update({iatype:tmp_upup})
                soc_atom_updown.update({iatype:tmp_updown})
            self.__struct__.soc_atom_upup = soc_atom_upup
            self.__struct__.soc_atom_updown = soc_atom_updown
        
        for ib in range(len(bonds_onsite)):
            ibond = bonds_onsite[ib].int()
            iatom = ibond[1]
            ist = int(np.sum(numOrbs[0:iatom]))
            ied = int(np.sum(numOrbs[0:iatom+1]))
            iatype = self.__struct__.proj_atom_symbols[iatom]

            # get lambdas
            istin = 0
            lambdas = torch.zeros((ied-ist,), device=self.device, dtype=self.cdtype)
            for ish in self.__struct__.proj_atom_anglr_m[iatype]:
                indx = self.__struct__.onsite_index_map[iatype][ish]
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                shidi = anglrMId[ishsymbol]          # 0,1,2,...
                norbi = 2*shidi + 1
                lambdas[istin:istin+norbi] = self.soc_lambdas[ib][indx]
                istin = istin + norbi
                
            soc_upup[ist:ied,ist:ied] = soc_atom_upup[iatype] @ torch.diag(lambdas)
            soc_updown[ist:ied, ist:ied] = soc_atom_updown[iatype] @ torch.diag(lambdas)
        
        soc_upup.contiguous()
        soc_updown.contiguous()

        return soc_upup, soc_updown
    
    def get_hs_onsite(self, bonds_onsite = None, onsite_envs=None):
        if bonds_onsite is None:
            _, bonds_onsite = self.__struct__.get_bond()
        onsiteH_blocks = []
        if not self.use_orthogonal_basis:
            onsiteS_blocks = []
        else:
            onsiteS_blocks = None
        
        iatom_to_onsite_index = {}
        for ib in range(len(bonds_onsite)):
            ibond = bonds_onsite[ib].int()
            iatom = int(ibond[1])
            iatom_to_onsite_index.update({iatom:ib})
            jatom = int(ibond[3])
            iatype = self.__struct__.proj_atom_symbols[iatom]
            jatype = self.__struct__.proj_atom_symbols[jatom]
            assert iatype == jatype, "i type should equal j type."

            sub_hamil_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]], dtype=self.dtype, device=self.device)
            if not self.use_orthogonal_basis:
                # For non - orthogonal basis, the overlap matrix is needed. 
                # but for the onsite, the overlap matrix is identity.
                sub_over_block = th.eye(self.__struct__.proj_atomtype_norbs[iatype], dtype=self.dtype, device=self.device)

            ist = 0
            for ish in self.__struct__.proj_atom_anglr_m[iatype]:     # ['s','p',..]
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                shidi = anglrMId[ishsymbol]          # 0,1,2,...
                norbi = 2*shidi + 1

                indx = self.__struct__.onsite_index_map[iatype][ish] # change onsite index map from {N:{s:}} to {N:{ss:, sp:}}
                sub_hamil_block[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi, dtype=self.dtype, device=self.device) * self.onsiteEs[ib][indx]
                # For non - orthogonal basis, the onsite overlap is identity, we don't need to calculate it.
                #if not self.use_orthogonal_basis:
                #    sub_over_block[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi, dtype=self.dtype, device=self.device) * self.onsiteSs[ib][indx]
                ist = ist + norbi

            onsiteH_blocks.append(sub_hamil_block)
            if not self.use_orthogonal_basis:
                onsiteS_blocks.append(sub_over_block)

        # onsite strain
        if onsite_envs is not None:
            assert self.onsiteVs is not None
            for ib, env in enumerate(onsite_envs):
                
                iatype, iatom, jatype, jatom = self.__struct__.proj_atom_symbols[int(env[1])], int(env[1]), self.__struct__.atom_symbols[int(env[3])], int(env[3])
                direction_vec = env[8:11].float()

                sub_hamil_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[iatype]], dtype=self.dtype, device=self.device)
                envtype = iatype + '-' + jatype

                ist = 0
                for ish in self.__struct__.proj_atom_anglr_m[iatype]:
                    ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                    shidi = anglrMId[ishsymbol]
                    norbi = 2*shidi+1
                    
                    jst = 0
                    for jsh in self.__struct__.proj_atom_anglr_m[iatype]:
                        jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                        shidj = anglrMId[jshsymbol]
                        norbj = 2 * shidj + 1

                        idx = self.__struct__.onsite_strain_index_map[envtype][ish+'-'+jsh]
                        
                        if shidi < shidj:
                            
                            tmpH = self.rot_HS(Htype=ishsymbol+jshsymbol, Hvalue=self.onsiteVs[ib][idx], Angvec=direction_vec)
                            # Hamilblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                            sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                        else:
                            tmpH = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue=self.onsiteVs[ib][idx], Angvec=direction_vec)
                            sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = tmpH
                
                        jst = jst + norbj 
                    ist = ist + norbi
                onsiteH_blocks[iatom_to_onsite_index[iatom]] += sub_hamil_block

        return onsiteH_blocks, onsiteS_blocks, bonds_onsite
    
    def get_hs_hopping(self, bonds_hoppings = None):
        if bonds_hoppings is None:
            bonds_hoppings, _ = self.__struct__.get_bond()

        hoppingH_blocks = []
        if not self.use_orthogonal_basis:
            hoppingS_blocks = []
        else:
            hoppingS_blocks = None
        
        out_bonds = []
        atomtype = self.__struct__.atomtype
        for iatype in atomtype:
            for jatype in atomtype:
                ia = atomic_num_dict[iatype]
                ja = atomic_num_dict[jatype]
                mask = bonds_hoppings[:,0].int().eq(ia) & bonds_hoppings[:,2].int().eq(ja)
                bonds = bonds_hoppings[torch.arange(bonds_hoppings.shape[0])[mask]]

                if len(bonds) == 0:
                    continue
                else:
                    hoppings = torch.stack([self.hoppings[i] for i in torch.arange(bonds_hoppings.shape[0])[mask]]) # might have problems
                    direction_vec = bonds[:,8:11].type(self.dtype)
                    sub_hamil_block = th.zeros([len(bonds), self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]], dtype=self.dtype, device=self.device)
                    if not self.use_orthogonal_basis:
                        sub_over_block = th.zeros([len(bonds), self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]], dtype=self.dtype, device=self.device)
                        overlaps = torch.stack([self.overlaps[i] for i in torch.arange(bonds_hoppings.shape[0])[mask]]) # might have problems
                    ist = 0
                    for ish in self.__struct__.proj_atom_anglr_m[iatype]:
                        ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                        shidi = anglrMId[ishsymbol]
                        norbi = 2*shidi+1
                        jst = 0
                        for jsh in self.__struct__.proj_atom_anglr_m[jatype]:
                            jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                            shidj = anglrMId[jshsymbol]
                            norbj = 2 * shidj + 1
                            idx = self.__struct__.bond_index_map[iatype+'-'+jatype][ish+'-'+jsh]
                            if shidi < shidj:
                                tmpH = self.rot_HS(Htype=ishsymbol+jshsymbol, Hvalue=hoppings[:,idx], Angvec=direction_vec)
                                # Hamilblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                                sub_hamil_block[:,ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpH,dim0=-2,dim1=-1)
                                if not self.use_orthogonal_basis:
                                    tmpS = self.rot_HS(Htype=ishsymbol+jshsymbol, Hvalue=overlaps[:,idx], Angvec=direction_vec)
                                # Soverblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpS,dim0=0,dim1=1)
                                    sub_over_block[:,ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpS,dim0=-2,dim1=-1)
                            else:
                                tmpH = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue=hoppings[:,idx], Angvec=direction_vec)
                                sub_hamil_block[:,ist:ist+norbi, jst:jst+norbj] = tmpH
                                if not self.use_orthogonal_basis:
                                    tmpS = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue = overlaps[:,idx], Angvec = direction_vec)
                                    sub_over_block[:,ist:ist+norbi, jst:jst+norbj] = tmpS
                        
                            jst = jst + norbj 
                        ist = ist + norbi
                    hoppingH_blocks.extend(list(sub_hamil_block))
                    if not self.use_orthogonal_basis:
                        hoppingS_blocks.extend(list(sub_over_block))
                    out_bonds.extend(list(bonds))

        return hoppingH_blocks, hoppingS_blocks, torch.stack(out_bonds)
    
    def get_hs_blocks(self, bonds_onsite = None, bonds_hoppings=None, onsite_envs=None):
        onsiteH, onsiteS, bonds_onsite = self.get_hs_onsite(bonds_onsite=bonds_onsite, onsite_envs=onsite_envs)
        hoppingH, hoppingS, bonds_hoppings = self.get_hs_hopping(bonds_hoppings=bonds_hoppings)

        self.all_bonds = torch.cat([bonds_onsite[:,0:7],bonds_hoppings[:,0:7]],dim=0)
        self.all_bonds = self.all_bonds.int()
        onsiteH.extend(hoppingH)
        self.hamil_blocks = onsiteH
        if not self.use_orthogonal_basis:
            onsiteS.extend(hoppingS)
            self.overlap_blocks = onsiteS
        if self.soc:
            self.soc_upup, self.soc_updown = self.get_soc_block(bonds_onsite=bonds_onsite)

        return True

    def hs_block_R2k(self, kpoints, HorS='H', time_symm=True):
        '''The function takes in a list of Hamiltonian matrices for each bond, and a list of k-points, and
        returns a list of Hamiltonian matrices for each k-point

        Parameters
        ----------
        HorS
            string, 'H' or 'S' to indicate for Hk or Sk calculation.
        kpoints
            the k-points in the path.
        time_symm, optional
            if True, the Hamiltonian is time-reversal symmetric, defaults to True (optional)
        dtype, optional
            'tensor' or 'numpy', defaults to tensor (optional)

        Returns
        -------
            A list of Hamiltonian or Overlap matrices for each k-point.
        ''' 

        numOrbs = np.array(self.num_orbs_per_atom)
        totalOrbs = np.sum(numOrbs)
        if HorS == 'H':
            hijAll = self.hamil_blocks
        elif HorS == 'S':
            hijAll = self.overlap_blocks
        else:
            print("HorS should be 'H' or 'S' !")

        if self.soc:
            Hk = th.zeros([len(kpoints), 2*totalOrbs, 2*totalOrbs], dtype = self.cdtype, device=self.device)
        else:
            Hk = th.zeros([len(kpoints), totalOrbs, totalOrbs], dtype = self.cdtype, device=self.device)

        for ik in range(len(kpoints)):
            k = kpoints[ik]
            hk = th.zeros([totalOrbs,totalOrbs],dtype = self.cdtype, device=self.device)
            for ib in range(len(self.all_bonds)):
                Rlatt = self.all_bonds[ib,4:7].int()
                i = self.all_bonds[ib,1].int()
                j = self.all_bonds[ib,3].int()
                ist = int(np.sum(numOrbs[0:i]))
                ied = int(np.sum(numOrbs[0:i+1]))
                jst = int(np.sum(numOrbs[0:j]))
                jed = int(np.sum(numOrbs[0:j+1]))
                if ib < len(numOrbs): 
                    """
                    len(numOrbs)= numatoms. the first numatoms are onsite energies.
                    if turn on timeSymm when generating the bond list <i,j>. only i>= or <= j are included. 
                    if turn off timeSymm when generating the bond list <i,j>. all the i j are included.
                    for case 1, H = H+H^\dagger to get the full matrix, the the onsite one is doubled.
                    for case 2. no need to do H = H+H^dagger. since the matrix is already full.
                    """
                    if time_symm:
                        hk[ist:ied,jst:jed] += 0.5 * hijAll[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,Rlatt))
                    else:
                        hk[ist:ied,jst:jed] += hijAll[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,Rlatt)) 
                else:
                    hk[ist:ied,jst:jed] += hijAll[ib] * np.exp(-1j * 2 * np.pi* np.dot(k,Rlatt)) 
            if time_symm:
                hk = hk + hk.T.conj()
            if self.soc:
                hk = torch.kron(input=torch.eye(2, device=self.device, dtype=self.dtype), other=hk)
            Hk[ik] = hk
        
        if self.soc:
            Hk[:, :totalOrbs, :totalOrbs] += self.soc_upup.unsqueeze(0)
            Hk[:, totalOrbs:, totalOrbs:] += self.soc_upup.conj().unsqueeze(0)
            Hk[:, :totalOrbs, totalOrbs:] += self.soc_updown.unsqueeze(0)
            Hk[:, totalOrbs:, :totalOrbs] += self.soc_updown.conj().unsqueeze(0)
        
        Hk.contiguous()
            
        return Hk

    def Eigenvalues(self, kpoints, time_symm=True, unit="Hartree",if_eigvec=False):
        """ using the tight-binding H and S matrix calculate eigenvalues at kpoints.
        
        Args:
            kpoints: the k-kpoints used to calculate the eigenvalues.
        Note: must have the BondHBlock and BondSBlock 
        """
        hkmat = self.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=time_symm)
        if not self.use_orthogonal_basis:
            skmat =  self.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=time_symm)
        else:
            skmat = torch.eye(hkmat.shape[1], dtype=self.cdtype).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)

        if self.use_orthogonal_basis:
            Heff = hkmat
        else:
            chklowt = th.linalg.cholesky(skmat)
            chklowtinv = th.linalg.inv(chklowt)
            Heff = (chklowtinv @ hkmat @ th.transpose(chklowtinv,dim0=1,dim1=2).conj())
        # the factor 13.605662285137 * 2 from Hartree to eV.
        # eigks = th.linalg.eigvalsh(Heff) * 13.605662285137 * 2
        if if_eigvec:
            eigks, eigvec = th.linalg.eigh(Heff)
        else:
            eigks = th.linalg.eigvalsh(Heff)
        
        if unit == "Hartree":
            factor = 13.605662285137 * 2
        elif unit == "eV":
            factor = 1.0
        elif unit == "Ry":
            factor = 13.605662285137
        else:
            log.error("The unit name is not correct !")
            raise ValueError
        eigks = eigks * factor
        # Qres = Q.detach()
        # else:
        #     chklowt = np.linalg.cholesky(skmat)
        #     chklowtinv = np.linalg.inv(chklowt)
        #     Heff = (chklowtinv @ hkmat @ np.transpose(chklowtinv,(0,2,1)).conj())
        #     eigks = np.linalg.eigvalsh(Heff) * 13.605662285137 * 2
        #     Qres = 0
        
        if if_eigvec:
            return eigks, eigvec
        else:
            return eigks, None

