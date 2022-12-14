from typing_extensions import Self
from dptb.utils.tools import get_uniq_symbol
from dptb.utils.constants import anglrMId
import re
import numpy as np


class Index_Mapings(object):
    ''' creat index mappings for networks outs and the corresponding physical parameters.

        Paras 
        -----
        proj_atom_anglr_m: 
            the projection atoms angular momentum: eg: proj_atom_anglr_m = {'B':['s'],'N':['s','p']}
    '''

    def __init__(self, proj_atom_anglr_m=None):
        self.AnglrMID = anglrMId
        if  proj_atom_anglr_m is not None:
            self.update(proj_atom_anglr_m = proj_atom_anglr_m)

    def update(self, proj_atom_anglr_m):
        # bondtype, means the atoms types for bond. here ['N', 'B']
        self.bondtype = get_uniq_symbol(list(proj_atom_anglr_m.keys()))
        # projected angular momentum. get from struct class.
        self.ProjAnglrM = proj_atom_anglr_m

    def Bond_Ind_Mapings(self):
        ''' creat index mappings for networks outs and the hoppings.

        Output: 
        -------
        bond_index_map: dict the index mapping for bond hoppings. 
        bond_num_hops: dict the number of hops for each bond.
        e.g.
            for proj_atom_anglr_m = {'B':['s'],'N':['s','p']}, the output will be:
        
                bond_index_map = {'N-N': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]},
                                  'N-B': {'s-s': [0], 'p-s': [1]},
                                  'B-N': {'s-s': [0], 's-p': [1]},
                                  'B-B': {'s-s': [0]}}

                bond_num_hops = {'N-N': 4, 'N-B': 2, 'B-N': 2, 'B-B': 1}
        '''
        
        bond_index_map = {}
        bond_num_hops = {}
        for it in range(len(self.bondtype)):
            for jt in range(len(self.bondtype)):
                itype = self.bondtype[it]
                jtype = self.bondtype[jt]
                orbdict = {}
                ist = 0
                numhops = 0
                for ish in self.ProjAnglrM[itype]:
                    for jsh in self.ProjAnglrM[jtype]:
                        ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                        jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                        ishid = self.AnglrMID[ishsymbol]
                        jshid = self.AnglrMID[jshsymbol]
                        if it == jt:
                            if  jsh + '-' + ish in orbdict.keys():
                                orbdict[ish + '-' + jsh] = orbdict[jsh + '-' + ish]
                                continue
                            else:
                                numhops += min(ishid, jshid) + 1
                                orbdict[ish + '-' + jsh] = np.arange(ist, ist + min(ishid, jshid) + 1).tolist()

                        elif it < jt:
                            numhops += min(ishid, jshid) + 1
                            orbdict[ish + '-' + jsh] = np.arange(ist, ist + min(ishid, jshid) + 1).tolist()
                        else:
                            numhops += min(ishid, jshid) + 1
                            orbdict[ish + '-' + jsh] = bond_index_map[jtype + '-' + itype][jsh +'-'+ ish]
                            continue

                        # orbdict[ish+jsh] = paralist
                        ist += min(ishid, jshid) + 1
                        # print (itype, jtype, ish+jsh, ishid, jshid,paralist)
                bond_index_map[itype + '-' + jtype] = orbdict
                bond_num_hops[itype + '-' + jtype] = numhops

        return bond_index_map, bond_num_hops
    
    def _Onsite_Ind_Mapings(self):
        ''' creat index mappings for networks outs and the onsite energy.

        Output: 
        -------
        onsite_index_map: dict the index mapping for onsite energy. 
        onsite_num: dict the number of onsite energy for each atom type.
        e.g.
            for proj_atom_anglr_m = {'B':['s'],'N':['s','p']}, the output will be:
        
                onsite_index_map = {'N': {'s': [0], 'p': [1]}, 'B': {'s': [0]}}

                onsite_num = {'N': 2, 'B': 1}
        Note: here for this mode, the orbital s, p, d all assumed to de degenerated values for onsite energy. 
               Therefore, for each orbial only 1 value is enough.
        '''
        onsite_index_map = {}
        onsite_num = {}
        for it in range(len(self.bondtype)):
            itype = self.bondtype[it]
            orbdict = {}
            ist = 0
            numhops = 0
            for ish in self.ProjAnglrM[itype]:
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                ishid = self.AnglrMID[ishsymbol]
                orbdict[ish] = [ist]
                ist += 1
                numhops += 1
            onsite_index_map[itype] = orbdict
            onsite_num[itype] = numhops

        return onsite_index_map, onsite_num
    
    def _Onsite_Ind_Mapings_OrbSplit(self):
        ''' creat index mappings for networks outs and the onsite energy for split mode.

        Output: 
        -------
        onsite_index_map: dict the index mapping for onsite energy. 
        onsite_num: dict the number of onsite energy for each atom type.
        e.g.
            for proj_atom_anglr_m = {'B':['s'],'N':['s','p']}, the output will be:
        
                onsite_index_map = {'N': {'s': [0], 'p': [1, 2, 3]}, 'B': {'s': [0]}}

                onsite_num = {'N': 4, 'B': 1}
        Note 1: here for this mode, onsite energy of the orbital l = s, p, d, for different m, ie px py pz for p or dxy dyz .. for d can be different.
        Note 2: this mode is only for testing, not used in real production.
        '''
        onsite_index_map = {}
        onsite_num = {}
        for it in range(len(self.bondtype)):
            itype = self.bondtype[it]
            orbdict = {}
            ist = 0
            numhops = 0
            for ish in self.ProjAnglrM[itype]:
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                ishid = self.AnglrMID[ishsymbol]
                orbdict[ish] = np.arange(ist, ist + 2 * ishid + 1).tolist()
                ist += 2*ishid + 1
                numhops += 2*ishid + 1
            onsite_index_map[itype] = orbdict
            onsite_num[itype] = numhops

        return onsite_index_map, onsite_num


    def _OnsiteStrain_Ind_Mapings(self, atomtypes):
        ''' creat index mappings for networks outs and the onsite energy for strain mode.

        INPUT:
        ------
        atomtypes: list of atomtypes. e.g. ['N','B']

        Output: 
        -------
        onsite_intgrl_index_map: dict the index mapping for onsite_intgrl. 
        onsite_intgrl_num: dict the number of onsite_intgrl for each atom type.
        e.g.
            for proj_atom_anglr_m = {'B':['s'],'N':['s','p']}, the output will be:
        
                onsite_intgrl_index_map = {'N-N': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]},
                                           'N-B': {'s-s': [0], 's-p': [1], 'p-s': [1], 'p-p': [2, 3]},
                                           'B-N': {'s-s': [0]},
                                           'B-B': {'s-s': [0]}}

                onsite_intgrl_num =  {'N-N': 4, 'N-B': 4, 'B-N': 1, 'B-B': 1})
        Note: here for this mode, onsite energy is treated as a block matrix in the hamiltonian,
                 which can be formed as the hoppings part with the sk-like integrals.
        '''

        onsite_intgrl_index_map = {}
        onsite_intgrl_num = {}
        for it in range(len(self.bondtype)):
            for jt in range(len(atomtypes)):
                itype = self.bondtype[it]
                jtype = atomtypes[jt]
                orbdict = {}
                ist = 0
                num_onsite_intgrl = 0
                for ish in self.ProjAnglrM[itype]:
                    for jsh in self.ProjAnglrM[itype]:
                        ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                        jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                        ishid = self.AnglrMID[ishsymbol]
                        jshid = self.AnglrMID[jshsymbol]
                        if  jsh + '-' + ish in orbdict.keys():
                            orbdict[ish + '-' + jsh] = orbdict[jsh + '-' + ish]
                            continue
                        else:
                            num_onsite_intgrl += min(ishid, jshid) + 1
                            orbdict[ish + '-' + jsh] = np.arange(ist, ist + min(ishid, jshid) + 1).tolist()

                        ist += min(ishid, jshid) + 1
                # note: there is no symmetry of interchange i,j. since i is the center atom which the onsite orbitals lie on but j it the neighbour atom 
                # this is the difference with the hoppings part. where the orbitals from both i and j. there only on i.
                onsite_intgrl_index_map[itype + '-' + jtype] = orbdict
                onsite_intgrl_num[itype + '-' + jtype] = num_onsite_intgrl

        return onsite_intgrl_index_map, onsite_intgrl_num

    def Onsite_Ind_Mapings(self, onsitemode, atomtype=None):
        onsite_strain_index_map, onsite_strain_num = None, None
        if onsitemode in ['uniform', 'none']:
            onsite_index_map, onsite_num = self._Onsite_Ind_Mapings()
        elif onsitemode == 'split':
            onsite_index_map, onsite_num = self._Onsite_Ind_Mapings_OrbSplit()
        elif onsitemode == 'strain':
            onsite_index_map, onsite_num = self._Onsite_Ind_Mapings()
            assert atomtype is not None, "Error: atomtype should not be None when requires strain"
            onsite_strain_index_map, onsite_strain_num = self._OnsiteStrain_Ind_Mapings(atomtype)
        else:
            raise ValueError(f'Unknown onsitemode {onsitemode}')

        return onsite_strain_index_map, onsite_strain_num, onsite_index_map, onsite_num

if __name__ == '__main__':
    im = Index_Mapings(proj_atom_anglr_m={"N":["2s","2p"], "C":["2s","2p"]})
    ma, l = im.OnsiteStrain_Ind_Mapings(atomtypes=["N"])
    print(ma, l)
    

