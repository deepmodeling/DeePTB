import torch
import torch as th
import numpy as np
import logging
import re
from dptb.hamiltonian.transform_sk import RotationSK
from dptb.nnsktb.formula import SKFormula
from dptb.utils.constants import anglrMId

''' Over use of different index system cause the symbols and type and index kind of object need to be recalculated in different 
Class, this makes entanglement of classes difficult. Need to design an consistent index system to resolve.'''

log = logging.getLogger(__name__)

class HamilEig(RotationSK):
    """ This module is to build the Hamiltonian from the SK-type bond integral.
    """
    def __init__(self, dtype='tensor') -> None:
        super().__init__(rot_type=dtype)
        self.dtype = dtype
        self.use_orthogonal_basis = False
        self.hamil_blocks = None
        self.overlap_blocks = None

    def update_hs_list(self, struct, hoppings, onsiteEs, onsiteVs=None, overlaps=None, onsiteSs=None, **options):
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
        self.use_orthogonal_basis = False
        if overlaps is None:
            self.use_orthogonal_basis = True
        else:
            self.overlaps = overlaps
            self.onsiteSs = onsiteSs
            self.use_orthogonal_basis = False

        self.num_orbs_per_atom = []
        for itype in self.__struct__.proj_atom_symbols:
            norbs = self.__struct__.proj_atomtype_norbs[itype]
            self.num_orbs_per_atom.append(norbs)
        
    def get_hs_blocks(self, bonds_onsite = None, bonds_hoppings=None, onsite_envs=None):
        """using the SK type bond integral  to build the hamiltonian matrix and overlap matrix in the real space.

        The hamiltonian and overlap matrix block are stored in the order of bond list. for ecah bond ij, with lattice 
        vecto R, the matrix stored in [norbsi, norbsj]. norsbi and norbsj are the total number of orbtals on i and j sites.
        e.g. for C-atom with both s and p orbital on each site. norbi is 4.

        bonds_env: {iatom: [env_list]}
        """

        if bonds_onsite is None:
            _, bonds_onsite = self.__struct__.get_bond(sorted=None)
        if bonds_hoppings is None:
            bonds_hoppings, _ = self.__struct__.get_bond(sorted=None)

        # ToDo: 1. add d_ij dependence of onsite params 2. rewrite the onsite_index_map 3. confirm the formula of onsite output param
        hamil_blocks = []
        if not self.use_orthogonal_basis:
            overlap_blocks = []
        
        iatom_to_onsite_index = {}
        for ib in range(len(bonds_onsite)):
            ibond = bonds_onsite[ib].astype(int)
            iatom = ibond[1]
            iatom_to_onsite_index.update({iatom:ib})
            jatom = ibond[3]
            iatype = self.__struct__.proj_atom_symbols[ibond[1]]
            jatype = self.__struct__.proj_atom_symbols[jatom]
            assert iatype == jatype, "i type should equal j type."

            if self.dtype == 'tensor':
                sub_hamil_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
                if not self.use_orthogonal_basis:
                    sub_over_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
            else:
                sub_hamil_block = np.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
                if not self.use_orthogonal_basis:
                    sub_over_block = np.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
            
            ist = 0
            for ish in self.__struct__.proj_atom_anglr_m[iatype]:     # ['s','p',..]
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                shidi = anglrMId[ishsymbol]          # 0,1,2,...
                norbi = 2*shidi + 1 

                indx = self.__struct__.onsite_index_map[iatype][ish] # change onsite index map from {N:{s:}} to {N:{ss:, sp:}}
                if self.dtype == 'tensor':
                    sub_hamil_block[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi) * self.onsiteEs[ib][indx]
                    if not self.use_orthogonal_basis:
                        sub_over_block[ist:ist+norbi, ist:ist+norbi] = th.eye(norbi) * self.onsiteSs[ib][indx]
                else:
                    sub_hamil_block[ist:ist+norbi, ist:ist+norbi] = np.eye(norbi) * self.onsiteEs[ib][indx]
                    if not self.use_orthogonal_basis:
                        sub_over_block[ist:ist+norbi, ist:ist+norbi] = np.eye(norbi) * self.onsiteSs[ib][indx]
                ist = ist + norbi

            hamil_blocks.append(sub_hamil_block)
            if not self.use_orthogonal_basis:
                overlap_blocks.append(sub_over_block)

        # onsite strain
        if onsite_envs is not None:
            assert self.onsiteVs is not None
            for ib, env in enumerate(onsite_envs):
                
                iatype, iatom, jatype, jatom = self.__struct__.proj_atom_symbols[int(env[1])], env[1], self.__struct__.atom_symbols[int(env[3])], env[3]
                direction_vec = env[8:11].astype(np.float32)

                if self.dtype == 'tensor':
                    sub_hamil_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[iatype]])
                else:
                    sub_hamil_block = np.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[iatype]])
            
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
                            if self.dtype == 'tensor':
                                sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                            else:
                                sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = np.transpose(tmpH,(1,0))
                        else:
                            tmpH = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue=self.onsiteVs[ib][idx], Angvec=direction_vec)
                            sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = tmpH
                
                        jst = jst + norbj 
                    ist = ist + norbi   
                hamil_blocks[iatom_to_onsite_index[iatom]] += sub_hamil_block

        for ib in range(len(bonds_hoppings)):
            
            ibond = bonds_hoppings[ib,0:7].astype(int)
            #direction_vec = (self.__struct__.projected_struct.positions[ibond[3]]
            #          - self.__struct__.projected_struct.positions[ibond[1]]
            #          + np.dot(ibond[4:], self.__struct__.projected_struct.cell))
            #dist = np.linalg.norm(direction_vec)
            #direction_vec = direction_vec/dist
            direction_vec = bonds_hoppings[ib,8:11].astype(np.float32)
            iatype = self.__struct__.proj_atom_symbols[ibond[1]]
            jatype = self.__struct__.proj_atom_symbols[ibond[3]]

            if self.dtype == 'tensor':
                sub_hamil_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
                if not self.use_orthogonal_basis:
                    sub_over_block = th.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
            else:
                sub_hamil_block = np.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
                if not self.use_orthogonal_basis:
                    sub_over_block = np.zeros([self.__struct__.proj_atomtype_norbs[iatype], self.__struct__.proj_atomtype_norbs[jatype]])
            
            bondatomtype = iatype + '-' + jatype
            
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

                    idx = self.__struct__.bond_index_map[bondatomtype][ish+'-'+jsh]
                    if shidi < shidj:
                        tmpH = self.rot_HS(Htype=ishsymbol+jshsymbol, Hvalue=self.hoppings[ib][idx], Angvec=direction_vec)
                        # Hamilblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpH,dim0=0,dim1=1)
                        if self.dtype == 'tensor':
                            sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpH,dim0=0,dim1=1)
                        else:
                            sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * np.transpose(tmpH,(1,0))
                        if not self.use_orthogonal_basis:
                            tmpS = self.rot_HS(Htype=ishsymbol+jshsymbol, Hvalue=self.overlaps[ib][idx], Angvec=direction_vec)
                        # Soverblock[ist:ist+norbi, jst:jst+norbj] = th.transpose(tmpS,dim0=0,dim1=1)
                            if self.dtype == 'tensor':
                                sub_over_block[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * th.transpose(tmpS,dim0=0,dim1=1)
                            else:
                                sub_over_block[ist:ist+norbi, jst:jst+norbj] = (-1.0)**(shidi + shidj) * np.transpose(tmpS,(1,0))
                    else:
                        tmpH = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue=self.hoppings[ib][idx], Angvec=direction_vec)
                        sub_hamil_block[ist:ist+norbi, jst:jst+norbj] = tmpH
                        if not self.use_orthogonal_basis:
                            tmpS = self.rot_HS(Htype=jshsymbol+ishsymbol, Hvalue = self.overlaps[ib][idx], Angvec = direction_vec)
                            sub_over_block[ist:ist+norbi, jst:jst+norbj] = tmpS
                
                    jst = jst + norbj 
                ist = ist + norbi   
            hamil_blocks.append(sub_hamil_block)
            if not self.use_orthogonal_basis:
                overlap_blocks.append(sub_over_block)
        self.all_bonds = np.concatenate([bonds_onsite[:,0:7],bonds_hoppings[:,0:7]],axis=0)
        self.all_bonds = self.all_bonds.astype(int)
        self.hamil_blocks = hamil_blocks
        if not self.use_orthogonal_basis:
            self.overlap_blocks = overlap_blocks


    def hs_block_R2k(self, kpoints, HorS='H', time_symm=True, dtype='tensor'):
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

        if dtype == 'tensor':
            Hk = th.zeros([len(kpoints), totalOrbs, totalOrbs], dtype = th.complex64)
        else:
            Hk = np.zeros([len(kpoints), totalOrbs, totalOrbs], dtype = np.complex64)

        for ik in range(len(kpoints)):
            k = kpoints[ik]
            if dtype == 'tensor':
                hk = th.zeros([totalOrbs,totalOrbs],dtype = th.complex64)
            else:
                hk = np.zeros([totalOrbs,totalOrbs],dtype = np.complex64)
            for ib in range(len(self.all_bonds)):
                Rlatt = self.all_bonds[ib,4:7].astype(int)
                i = self.all_bonds[ib,1].astype(int)
                j = self.all_bonds[ib,3].astype(int)
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
            Hk[ik] = hk
        return Hk

    def Eigenvalues(self, kpoints, time_symm=True,dtype='tensor'):
        """ using the tight-binding H and S matrix calculate eigenvalues at kpoints.
        
        Args:
            kpoints: the k-kpoints used to calculate the eigenvalues.
        Note: must have the BondHBlock and BondSBlock 
        """
        hkmat = self.hs_block_R2k(kpoints=kpoints, HorS='H', time_symm=time_symm, dtype=dtype)
        if not self.use_orthogonal_basis:
            skmat =  self.hs_block_R2k(kpoints=kpoints, HorS='S', time_symm=time_symm, dtype=dtype)
        else:
            skmat = torch.eye(hkmat.shape[1], dtype=torch.complex64).unsqueeze(0).repeat(hkmat.shape[0], 1, 1)

        if self.dtype == 'tensor':
            chklowt = th.linalg.cholesky(skmat)
            chklowtinv = th.linalg.inv(chklowt)
            Heff = (chklowtinv @ hkmat @ th.transpose(chklowtinv,dim0=1,dim1=2).conj())
            # the factor 13.605662285137 * 2 from Hartree to eV.
            # eigks = th.linalg.eigvalsh(Heff) * 13.605662285137 * 2
            eigks, Q = th.linalg.eigh(Heff)
            eigks = eigks * 13.605662285137 * 2
            Qres = Q.detach()
        else:
            chklowt = np.linalg.cholesky(skmat)
            chklowtinv = np.linalg.inv(chklowt)
            Heff = (chklowtinv @ hkmat @ np.transpose(chklowtinv,(0,2,1)).conj())
            eigks = np.linalg.eigvalsh(Heff) * 13.605662285137 * 2
            Qres = 0

        return eigks, Qres