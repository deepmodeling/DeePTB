from typing_extensions import Self
from dptb.utils.tools import get_uniq_symbol
from dptb.utils.constants import anglrMId, atomic_num_dict
import re
import torch
import numpy as np

class Index_Mapings_e3(object):
    def __init__(self, basis=None, method="e3tb"):
        self.basis = basis
        self.method = method
        if basis is not None:
            self.update(basis=basis)

        if self.method not in ["e3tb", "sktb"]:
            raise ValueError
        

    def update(self, basis):
        """_summary_

        Parameters
        ----------
        basis : dict
            the definition of the basis set, should be like:
            {"A":"2s2p3d1f", "B":"1s2f3d1f"} or
            {"A":["2s", "2p"], "B":["2s", "2p"]}
            when list, "2s" indicate a "s" orbital in the second shell.
            when str, "2s" indicates two s orbital, 
            "2s2p3d4f" is equivilent to ["1s","2s", "1p", "2p", "1d", "2d", "3d", "1f"]
        """

        self.atomtype = get_uniq_symbol(list(basis.keys())) # this will sort the atomtype according to the atomic number
        self.atomtype_map = {at:i for i, at in enumerate(self.atomtype)}
        self.bondtype = []

        for it, at in enumerate(self.atomtype):
            for jt, bt in enumerate(self.atomtype[it:]):
                bond = at+"-"+bt
                if bond not in at:
                    self.bondtype.append(bond)

        self.bondtype_map = {bt:i for i, bt in enumerate(self.bondtype)}

        # TODO: check the basis value

        self.basis = basis
        if isinstance(self.basis[self.atomtype[0]], str):
            orbtype_count = {"s":0, "p":0, "d":0, "f":0}
            orbs = map(lambda bs: re.findall(r'[1-9]+[A-Za-z]', bs), self.basis.values())
            for ib in orbs:
                for io in ib:
                    if int(io[0]) > orbtype_count[io[1]]:
                        orbtype_count[io[1]] = int(io[0])
            # split into list basis
            basis = {k:[] for k in self.atomtype}
            for ib in self.basis.keys():
                for io in ["s", "p", "d", "f"]:
                    if io in self.basis[ib]:
                        basis[ib].extend([str(i)+io for i in range(1, int(re.findall(r'[1-9]+'+io, self.basis[ib])[0][0])+1)])
            self.basis = basis

        elif isinstance(self.basis[self.atomtype[0]], list):
            nb = len(self.atomtype)
            orbtype_count = {"s":[0]*nb, "p":[0]*nb, "d":[0]*nb, "f":[0]*nb}
            for ib, bt in enumerate(self.atomtype):
                for io in self.basis[bt]:
                    orb = re.findall(r'[A-Za-z]', io)[0]
                    orbtype_count[orb][ib] += 1
            
            for ko in orbtype_count.keys():
                orbtype_count[ko] = max(orbtype_count[ko])

        self.orbtype_count = orbtype_count

        if self.method == "e3tb":
            self.edge_reduced_matrix_element = (1 * orbtype_count["s"] + 3 * orbtype_count["p"] + 5 * orbtype_count["d"] + 7 * orbtype_count["f"]) **2
            self.node_reduced_matrix_element = int(((orbtype_count["s"] + 9 * orbtype_count["p"] + 25 * orbtype_count["d"] + 49 * orbtype_count["f"]) + \
                                                    self.edge_reduced_matrix_element)/2)
        else:
            self.edge_reduced_matrix_element =  1 * (
                                                    1 * orbtype_count["s"] * orbtype_count["s"] + \
                                                    2 * orbtype_count["s"] * orbtype_count["p"] + \
                                                    2 * orbtype_count["s"] * orbtype_count["d"] + \
                                                    2 * orbtype_count["s"] * orbtype_count["f"]
                                                    ) + \
                                                2 * (
                                                    1 * orbtype_count["p"] * orbtype_count["p"] + \
                                                    2 * orbtype_count["p"] * orbtype_count["d"] + \
                                                    2 * orbtype_count["p"] * orbtype_count["f"]
                                                    ) + \
                                                3 * (
                                                    1 * orbtype_count["d"] * orbtype_count["d"] + \
                                                    2 * orbtype_count["d"] * orbtype_count["f"]
                                                    ) + \
                                                4 * (orbtype_count["f"] * orbtype_count["f"])
            
            self.node_reduced_matrix_element = orbtype_count["s"] + orbtype_count["p"] + orbtype_count["d"] + orbtype_count["f"]
                                     
        

        # sort the basis
        for ib in self.basis.keys():
            self.basis[ib] = sorted(
                self.basis[ib], 
                key=lambda s: (anglrMId[re.findall(r"[a-z]",s)[0]], re.findall(r"[1-9*]",s)[0])
                )

        # TODO: get full basis set
        full_basis = []
        for io in ["s", "p", "d", "f"]:
            full_basis = full_basis + [str(i)+io for i in range(1, orbtype_count[io]+1)]
        self.full_basis = full_basis

        # TODO: get the mapping from list basis to full basis
        self.basis_to_full_basis = {}
        for ib in self.basis.keys():
            count_dict = {"s":0, "p":0, "d":0, "f":0}
            self.basis_to_full_basis.setdefault(ib, {})
            for o in self.basis[ib]:
                io = re.findall(r"[a-z]", o)[0]
                count_dict[io] += 1
                self.basis_to_full_basis[ib][o] = str(count_dict[io])+io

        # also need to think if we modify as this, how can we add extra basis when fitting.


    def get_pairtype_maps(self):
        """
        The function `get_pairtype_maps` creates a mapping of orbital pair types, such as s-s, "s-p",
        to slices based on the number of hops between them.
        :return: a dictionary called `pairtype_map`.
        """
        
        self.pairtype_maps = {}
        ist = 0
        for io in ["s", "p", "d", "f"]:
            if self.orbtype_count[io] != 0:
                for jo in ["s", "p", "d", "f"]:
                    if self.orbtype_count[jo] != 0:
                        orb_pair = io+"-"+jo
                        il, jl = anglrMId[io], anglrMId[jo]
                        if self.method == "e3tb":
                            n_rme = (2*il+1) * (2*jl+1)
                        else:
                            n_rme = min(il, jl)+1
                        numhops =  self.orbtype_count[io] * self.orbtype_count[jo] * n_rme
                        self.pairtype_maps[orb_pair] = slice(ist, ist+numhops)

                        ist += numhops

        return self.pairtype_maps
    
    def get_pair_maps(self):

        # here we have the map from basis to full basis, but to define a map between basis pair to full basis pair,
        # one need to consider the id of the full basis pairs. Specifically, if we want to know the position where
        # "s*-2s" lies, we map it to the pair in full basis as "1s-2s", but we need to know the id of "1s-2s" in the 
        # features vector. For a full basis have three s: [1s, 2s, 3s], it will have 9 s features. Therefore, we need
        # to build a map from the full basis pair to the position in the vector.

        # We define the feature vector should look like [1s-1s, 1s-2s, 1s-3s, 2s-1s, 2s-2s, 2s-3s, 3s-1s, 3s-2s, 3s-3s,...]
        # it is sorted by the index of the left basis first, then the right basis. Therefore, we can build a map:

        # to do so we need the pair type maps first
        if not hasattr(self, "pairtype_maps"):
            self.pairtype_maps = self.get_pairtype_maps()
        self.pair_maps = {}
        for ib in self.bondtype:
            ia, ja = ib.split("-")
            self.pair_maps.setdefault(ib, {})
            for io in self.basis[ia]:
                for jo in self.basis[ja]:
                    full_basis_pair = self.basis_to_full_basis[ia][io]+"-"+self.basis_to_full_basis[ja][jo]
                    ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
                    iio, jjo = full_basis_pair[1], full_basis_pair[4]

                    if self.method == "e3tb":
                        n_feature = (2*anglrMId[iio]+1) * (2*anglrMId[jjo]+1)
                    else:
                        n_feature = min(anglrMId[iio], anglrMId[jjo])+1
                    

                    start = self.pairtype_maps[iio+"-"+jjo].start + \
                        n_feature * ((ir-1)*self.orbtype_count[jjo]+(jr-1))
                    
                    self.pair_maps[ib][io+"-"+jo] = slice(start, start+n_feature)
                        

        return self.pair_maps
    
    def get_node_maps(self):
        if not hasattr(self, "nodetype_maps"):
            self.get_nodetype_maps()
        
        self.node_maps = {}
        for at in self.atomtype:
            self.node_maps.setdefault(at, {})
            for i, io in enumerate(self.basis[at]):
                for jo in self.basis[at][i:]:
                    full_basis_pair = self.basis_to_full_basis[at][io]+"-"+self.basis_to_full_basis[at][jo]
                    ir, jr = int(full_basis_pair[0]), int(full_basis_pair[3])
                    iio, jjo = full_basis_pair[1], full_basis_pair[4]

                    if self.method == "e3tb":
                        n_feature = (2*anglrMId[iio]+1) * (2*anglrMId[jjo]+1)
                    else:
                        if io == jo:
                            n_feature = 1
                        else:
                            n_feature = 0
                
                    start = self.nodetype_maps[iio+"-"+jjo].start + \
                        n_feature * (2*self.orbtype_count[jjo]+1-ir) * (ir-1) / 2 + (jr - 1)
                    start = int(start)
                    
                    self.node_maps[at][io+"-"+jo] = slice(start, start+n_feature)

        return self.node_maps

    def get_nodetype_maps(self):
        self.nodetype_maps = {}
        ist = 0

        for i, io in enumerate(["s", "p", "d", "f"]):
            if self.orbtype_count[io] != 0:
                for jo in ["s", "p", "d", "f"][i:]:
                    if self.orbtype_count[jo] != 0:
                        orb_pair = io+"-"+jo
                        il, jl = anglrMId[io], anglrMId[jo]
                        if self.method == "e3tb":
                            numonsites =  self.orbtype_count[io] * self.orbtype_count[jo] * (2*il+1) * (2*jl+1)
                            if io == jo:
                                numonsites +=  self.orbtype_count[jo] * (2*il+1) * (2*jl+1)
                                numonsites = int(numonsites / 2)
                        else:
                            if io == jo:
                                numonsites = self.orbtype_count[io]
                            else:
                                numonsites = 0

                        self.nodetype_maps[orb_pair] = slice(ist, ist+numonsites)

                        ist += numonsites


        return self.nodetype_maps


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
        elif onsitemode == 'NRL':
            # TODO: design NRL onsite index map, 
            # usually NRL is the same as uniform. but in some case they treat t2g and eg orbitals as different.
            # therefore, we need new _Onsite_Ind_Mapings function for NRL.
            # here we just temporarily use uniform one!
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
    

