import ase
import warnings
import logging
import ase.neighborlist
from ase import Atoms
import torch
import numpy  as np
import re
from itertools import accumulate
import ase.io
from dptb.utils.constants import anglrMId,atomic_num_dict
from dptb.utils.tools import get_uniq_symbol, env_smoth
from dptb.utils.index_mapping import Index_Mapings
from dptb.structure.abstract_stracture import AbstractStructure

class BaseStruct(AbstractStructure):
    '''
        implement the read structure and get bond function
    '''
    def __init__(self, atom, format, cutoff, proj_atom_anglr_m, proj_atom_neles, onsitemode:str='none', time_symm=True):
        self.proj_atomtype_norbs = None
        self.onsitemode = onsitemode
        self.nbonds = 0
        assert isinstance(proj_atom_anglr_m, dict)
        assert isinstance(proj_atom_neles, dict)
        self.atom = atom
        self.format = format
        self.cutoff = cutoff
        self.proj_atom_anglr_m = proj_atom_anglr_m
        self.proj_atom_neles = proj_atom_neles
        self.time_symm = time_symm
        self.__projenv__ = {}
        self.__onsitenv__ = {}
        self.IndMap = Index_Mapings()
        self.update_struct(self.atom, format=format, onsitemode=onsitemode)

    def init_description(self):
        # init description
        self.atom_symbols = None
        self.aomtype = None
        # self.atomt_anglr_m = None
        self.proj_atomtype = None
        # self.proj_atom_anglr_m = None
        self.proj_atomtype_norbs = None
        self.__bonds_onsite__ = None
        self.__bonds__ = None
        self.if_env_ready = False
        self.if_onsitenv_ready = False
        self.onsite_cutoff = None

    def update_struct(self, atom, format, onsitemode:str='none'):
        self.init_description()
        self.onsitemode = onsitemode
        self.read_struct(atom,format=format)
        self.atom_symbols = np.array(self.struct.get_chemical_symbols(), dtype=str)
        self.atom_numbers = np.array(self.struct.get_atomic_numbers(), dtype=int)
        self.atomtype = get_uniq_symbol(atomsymbols=self.atom_symbols)
        self.projection()
        self.proj_atom_symbols = self.projected_struct.get_chemical_symbols()
        self.proj_atom_numbers = self.projected_struct.get_atomic_numbers()
        self.proj_atom_neles_per = np.array([self.proj_atom_neles[ii] for ii in self.proj_atom_symbols])
        self.proj_atom_to_atom_id = np.array(list(range(len(self.atom_symbols))))[self.projatoms]
        self.atom_to_proj_atom_id = np.array(list(accumulate([int(i) for i in self.projatoms]))) - 1
        self.proj_atomtype = get_uniq_symbol(atomsymbols=self.proj_atom_symbols)
        self.get_bond(cutoff=self.cutoff,time_symm=self.time_symm)
        self.if_env_ready = False
        self.if_onsitenv_ready = False

        self.IndMap.update(proj_atom_anglr_m=self.proj_atom_anglr_m)
        self.bond_index_map, self.bond_num_hops = self.IndMap.Bond_Ind_Mapings()
        self.onsite_strain_index_map, self.onsite_strain_num, self.onsite_index_map, self.onsite_num = self.IndMap.Onsite_Ind_Mapings(onsitemode, atomtype=self.atomtype)

    def read_struct(self, atom=None, format='ase'):
        '''The function reads a structure from a file or an ase object and stores it in the class
        
        Parameters
        ----------
        struct
            the structure to be read in. If format can be is 'ase', then struct is the ase.Atoms object. 
            Also struct can be a file name, with format can be 'xyz', 'vasp' etc. the supporting format is
            same as the ase.io.read.
        
        format, optional
            'ase' or file-format: e.g. 'xyz', 'vasp' etc.
        '''

        if format == 'ase':
            if type(atom) != ase.Atoms:
                logging.error("struct:TypeError, struct should be an instance of ASE Atoms")
                raise TypeError
            self.struct = atom
        else:
            structase = ase.io.read(filename=atom, format=format)
            self.struct = structase
        self.if_env_ready = False
        self.if_onsitenv_ready = False
        

    def projection(self):
        '''The function takes in a list of atom types and a list of angular momentum quantum numbers, and
        returns a projected structure with the atoms of the specified types

        Parameters
        ----------
        projAnglrM
            a list of lists of strings, each list of strings is a list of orbitals to be projected for a given
        atom type.
        projAtomType
            list of strings, the atom types to be projected

        Returns
        -------
            The projected structure.
        '''

        proj_atom_anglr_m = self.proj_atom_anglr_m
        self.proj_atomtype = get_uniq_symbol(list(proj_atom_anglr_m.keys()))

        if not isinstance(proj_atom_anglr_m, dict):
            logging.error("proj_atom_anglr_m:TypeError, must be a dict")
            raise TypeError

        #self.proj_atomtype_norbs = np.zeros(len(self.proj_atomtype), dtype=int)
        self.proj_atomtype_norbs = {}
        for ii in self.proj_atomtype:
            self.proj_atomtype_norbs[ii] = 0
            for iorb in proj_atom_anglr_m[ii]:
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',iorb))
                self.proj_atomtype_norbs[ii] += int(1 + 2 * anglrMId[ishsymbol])

        # projind = []
        self.projatoms = np.array([False] * len(self.struct.get_chemical_symbols()))
        for iproj in self.proj_atomtype:
            self.projatoms[np.where(np.asarray(self.struct.get_chemical_symbols()) == iproj)[0]] = True
            # ind_tmp = np.where(np.asarray(self.AtomSymbols)==iproj)[0]
            # projind.append(ind_tmp.tolist())
        # projind = np.concatenate(projind)
        symbols_arr = np.array(self.struct.get_chemical_symbols())

        self.projected_struct = Atoms(symbols=symbols_arr[self.projatoms].tolist(),
                                pbc=self.struct.pbc, cell=self.struct.cell,
                                positions=self.struct.positions[self.projatoms])

        return self.projected_struct

    def get_bond(self, cutoff=None, time_symm=True):
        if cutoff == None or cutoff == self.cutoff:
            if self.__bonds__ is not None and self.__bonds_onsite__ is not None:
                return self.__bonds__, self.__bonds_onsite__
            cutoff = self.cutoff
        if cutoff <= 0:
            logging.error("cutoff:ValueError, cutoff for bond is not positive'")
            raise ValueError
        
        self.__bonds__, self.__bonds_onsite__ = self.cal_bond(cutoff=cutoff, time_symm=time_symm)
        self.nbonds = len(self.__bonds__)

        return self.__bonds__, self.__bonds_onsite__

    def get_env(self, env_cutoff=None, sorted='iatom-jatom'):
        # for get env the default is turn on the smooth option.
        if self.if_env_ready:
            if env_cutoff == self.env_cutoff or env_cutoff == None:
                return self.__projenv__

        if env_cutoff is None:
            env_cutoff = self.env_cutoff

        if env_cutoff <= 0:
            logging.error("env_cutoff:ValueError, env_cutoff for bond is not positive'")
            raise ValueError
        else:
            self.__projenv__ = self.cal_env(env_cutoff=env_cutoff, sorted=sorted, smooth=True)
            self.env_cutoff = env_cutoff
            self.if_env_ready = True
            return self.__projenv__

    def get_onsitenv(self, onsite_cutoff=None, sorted='iatom'):
        if self.if_onsitenv_ready:
            if onsite_cutoff is None or onsite_cutoff == self.onsite_cutoff:
                return self.__onsitenv__
        
        assert isinstance(onsite_cutoff, float)

        if onsite_cutoff <= 0:
            logging.error("onsite_cutoff:ValueError, onsite_cutoff for bond is not positive'")
            raise ValueError
        else:
             # note: the onsite env is not smoothed. norm is |rij| not 1/|rij| or s(|rij|).
            self.__onsitenv__ = self.cal_env(env_cutoff=onsite_cutoff, sorted=sorted)
            self.onsite_cutoff = onsite_cutoff
            self.if_onsitenv_ready = True
            return self.__onsitenv__

    def cal_bond(self, cutoff=None, time_symm=True):
        '''It takes the structure, and returns the bonds and bonds on site.

        The bonds are the bonds between atoms, and the bonds on site are the bonds between an atom and
        itself.

        Parameters
        ----------
        timeSymm, optional
            whether to consider time symmetry. If True, only one bond between two atoms will be considered.
        cutOff
            the cutoff distance for the bonds.

        Returns
        -------
            The bonds and bonds on site are being returned.

        '''

        bonds_onsite = []

        ilist, jlist, Rlatt = ase.neighborlist.neighbor_list(quantities=['i', 'j', 'S'],
                                                             a=self.projected_struct, cutoff=cutoff)
        bonds_ = np.concatenate([np.reshape(ilist, [-1, 1]),
                                np.reshape(jlist, [-1, 1]), Rlatt], axis=1)

        nbonds = bonds_.shape[0]
        if time_symm:
            bonds_rd = []
            bonds_rd_dict = {}
            for inb in range(nbonds):
                atomi, atomj, R = bonds_[inb, 0], bonds_[inb, 1], bonds_[inb, 2:]
                bond_tmp = [atomi, atomj, R[0], R[1], R[2]]
                bond_tmp_xc = [atomj, atomi, -R[0], -R[1], -R[2]]
                
                bond_tmp_key = f'{atomi}_{atomj}_{R[0]}_{R[1]}_{R[2]}'
                bond_tmp_xc_key = f'{atomj}_{atomi}_{-R[0]}_{-R[1]}_{-R[2]}'

                if not (bond_tmp_xc_key in bonds_rd_dict) and not (bond_tmp_key in bonds_rd_dict):
                    bonds_rd_dict[bond_tmp_key] = ''
                    bonds_rd.append(bond_tmp)
                #if not (bond_tmp_xc in bonds_rd) and not (bond_tmp in bonds_rd):
                #    bonds_rd.append(bond_tmp)

            out_bonds = np.asarray(bonds_rd)
        else:
            out_bonds = np.asarray(bonds_)

        # iatom_nums = np.array([atomic_num_dict[self.proj_atom_symbols[i]] for i in out_bonds[:,0]])
        # jatom_nums = np.array([atomic_num_dict[self.proj_atom_symbols[i]] for i in out_bonds[:,1]])
        iatom_nums = self.proj_atom_numbers[out_bonds[:,0]]
        jatom_nums = self.proj_atom_numbers[out_bonds[:,1]]
        iatom_nums = iatom_nums[:,np.newaxis]
        jatom_nums = jatom_nums[:,np.newaxis]

        direction_vecs = self.projected_struct.positions[out_bonds[:,1]] - \
            self.projected_struct.positions[out_bonds[:,0]] + \
            np.dot(out_bonds[:,2:5], self.projected_struct.cell)
        norm = np.linalg.norm(direction_vecs,axis=1)
        norm = norm[:,np.newaxis]
        dircetion_cosine = direction_vecs / norm
        

        # bonds stores the bonds in the form of [i_atom_num, i, j_atom_num, j, Rx, Ry, Rz, |rj-ri|, \hat{rij: x, y, z}].
        bonds = np.concatenate((iatom_nums,out_bonds[:,[0]],jatom_nums,out_bonds[:,[1]],out_bonds[:,2:5], norm, dircetion_cosine),axis=1)
        bonds = torch.from_numpy(bonds)
        
        # on site bond
        for ii in range(len(self.proj_atom_symbols)):
            # self.bonds_onsite.append([ii, ii, 0, 0, 0])
            bonds_onsite.append([atomic_num_dict[self.proj_atom_symbols[ii]], ii, 
                                      atomic_num_dict[self.proj_atom_symbols[ii]], ii, 0, 0, 0])

        bonds_onsite = torch.tensor(bonds_onsite, dtype=torch.int32)

        return bonds, bonds_onsite # [itype, i, jtype, j, Rx, Ry, Rz, |rj-ri|, \hat{rij: x, y, z}]


    def cal_env(self, env_cutoff=None, sorted="iatom", smooth=False):
        '''

        Parameters
        ----------
        env_cutoff
        numenv

        Returns
            initiate the projenv: which looks like [i, j, itype, jtype, rx, ry, rz]
            beware that i,j in projenv is index from struct
        -------

        '''

        # if not isinstance(numenv, dict):
        #     logging.error("numenv:TypeError, must be a dict")
        #     raise TypeError
        #
        # if len(self.atomtype) != len(numenv):
        #     logging.error("numenv:ValueError, numenv must be list and have same length as total atom type in structure.")
        #     raise ValueError

        proj_env = {}
        ilist, jlist, Rlatt = ase.neighborlist.neighbor_list(quantities=['i', 'j', 'S'], a=self.struct, cutoff=env_cutoff)
        itypelist = self.atom_numbers[ilist]
        jtypelist = self.atom_numbers[jlist]

        shift_vec = self.struct.positions[jlist] - self.struct.positions[ilist] + np.matmul(Rlatt,
                                                                                           np.array(self.struct.cell))
        norm = np.linalg.norm(shift_vec, axis=1)
        shift_vec = shift_vec / np.reshape(norm, [-1,1])
        norm = np.reshape(norm, [-1, 1])
        if smooth:
            norm = env_smoth(norm, rcut=env_cutoff, rcut_smth=env_cutoff * 0.8)
        env_all_arrs = np.concatenate([np.reshape(itypelist, [-1, 1]), np.reshape(ilist, [-1, 1]), np.reshape(jtypelist, [-1, 1]), np.reshape(jlist, [-1, 1]), Rlatt,
                                       norm, shift_vec], axis=1)
        

        # (itype, i, jtype, j, Rx, Ry, Rz, |ri-rj|, rx, ry, rz)

        if sorted == "itype-jtype":
            envdict = {}

            for ii in range(len(env_all_arrs)):
                iatomtype = self.atom_symbols[env_all_arrs[ii][1].astype(int)]
                jatomtype = self.atom_symbols[env_all_arrs[ii][3].astype(int)]
                if iatomtype in self.proj_atomtype:
                    env_name = iatomtype+'-'+jatomtype
                    if envdict.get(env_name) is None:
                        envdict.update({env_name:[env_all_arrs[ii]]})
                    else:
                        envdict[env_name].append(env_all_arrs[ii])

            for kk in envdict:
                envdict[kk] = np.asarray(envdict[kk], dtype=float)
                envdict[kk][:, 1] = self.atom_to_proj_atom_id[envdict[kk][:, 1].astype(int)]
                proj_env[kk] = torch.from_numpy(np.asarray(envdict[kk], dtype=float))
        
        elif sorted == 'iatom':
            envdict = {}

            for ii in range(len(env_all_arrs)):
                iatom = env_all_arrs[ii][1].astype(int)
                if self.atom_symbols[iatom] in self.proj_atomtype:
                    env_name = iatom
                    if envdict.get(env_name) is None:
                        envdict.update({env_name:[env_all_arrs[ii]]})
                    else:
                        envdict[env_name].append(env_all_arrs[ii])

            for kk in envdict:
                envdict[kk] = np.asarray(envdict[kk], dtype=float)
                envdict[kk][:, 1] = self.atom_to_proj_atom_id[envdict[kk][:, 1].astype(int)]
                proj_env[kk] = torch.from_numpy(np.asarray(envdict[kk], dtype=float))
        
        elif sorted == None:
            env_all_arrs_ = []
            for ii in range(len(env_all_arrs)):
                iatomtype = self.atom_symbols[env_all_arrs[ii][1].astype(int)]
                jatomtype = self.atom_symbols[env_all_arrs[ii][3].astype(int)]
                if iatomtype in self.proj_atomtype:
                    env_all_arrs_.append(env_all_arrs[ii])
            env_all_arrs_ = np.array(env_all_arrs_,dtype=float)
            env_all_arrs_[:, 1] = self.atom_to_proj_atom_id[env_all_arrs_[:, 1].astype(int)]
            proj_env = env_all_arrs_
            proj_env = torch.from_numpy(np.asarray(proj_env, dtype=float))
        
        return proj_env # (itype, i, jtype, j, Rx, Ry, Rz, s(r), rx, ry, rz) or the dict of it


    def ijR2rij(self, ijR):
        rij = (self.struct.positions[ijR[:, 1]]
               - self.struct.positions[ijR[:, 0]]
               + np.dot(ijR[2:], np.array(self.struct.cell)))

        return rij

    def ibond_env(self, ibond, env_cutoff, numenv):
        # TODO: Now the smooth option is removed, It should be processed after the env is loaded in dptb
        """
            generate the environment for ecah bond.

            input
            -----
            ibond: i-th bond. ibond = Bonds[i]. has the form 》[i, j, rx,ry, rz]

            return
            ------
            envib4: N * 4 array. N is the sum of  NumEnv defined in input.
        """
        if not isinstance(numenv, dict):
            logging.error("numenv:TypeError, must be a dict")
            raise TypeError

        assert self.__projenv__ is not None
        assert self.if_env_ready


        lattice = np.asarray(self.struct.cell)
        positions = self.struct.positions
        #atomsymbols = self.struct.get_chemical_symbols()
        #uniqsybl = get_uniq_symbol(atomsymbols=atomsymbols)


        #numbondenv = np.array(numenv)
        number_env_continer = np.sum(list(numenv.values())) * 2
        envib4 = np.zeros([number_env_continer, 4])
        ist = 0
        for ib in [0, 1]:
            isite = ibond[ib]
            proj_env_site = np.asarray(self.__projenv__[isite], dtype=int)
            envitype = proj_env_site[:, 0]
            envlist = proj_env_site[:, 1]
            envRfrac = proj_env_site[:, 2:]
            envRcart = np.matmul(envRfrac, lattice)
            isitepos = self.projected_struct.positions[isite]
            envi = positions[envlist] - isitepos + envRcart
            rr = np.linalg.norm(envi, axis=1)
            envi_hat = envi / np.reshape(rr, [-1, 1])
            srr = env_smoth(rr, rcut=env_cutoff, rcut_smth=env_cutoff * 0.8)

            # for it in range(len(self.atomtype)):
            for it in self.atomtype:
                if np.sum(envitype == atomic_num_dict[it]) > 0:
                    srrit = np.reshape(srr[envitype == atomic_num_dict[it]], [-1, 1])
                    envi_hatit = envi_hat[envitype == atomic_num_dict[it]]
                    envi_hatit2 = np.concatenate([srrit, envi_hatit], axis=1)
                    envi_hatit2 = np.asarray(sorted(envi_hatit2, key=lambda s: s[0], reverse=True))

                    if np.sum(envitype == atomic_num_dict[it]) > numenv[it]:
                        print('Warning!, the size of env in cutoff is larger than NumEnv parameter.')
                        ied = ist + numenv[it]
                    else:
                        ied = ist + np.sum(envitype == atomic_num_dict[it])
                    # print(ist,ied)
                    envib4[ist:ied] = envi_hatit2[0:ied - ist]
                ist += numenv[it]

        return envib4

    def iatom_env(self, iatom, env_cutoff, numenv):
        """
                    generate the environment for ecah bond.

                    input
                    -----
                    ibond: i-th bond. ibond = Bonds[i]. has the form 》[i, j, rx,ry, rz]

                    return
                    ------
                    envib4: N * 4 array. N is the sum of  NumEnv defined in input.
                """
        if not isinstance(numenv, dict):
            logging.error("numenv:TypeError, must be a dict")
            raise TypeError

        assert self.__projenv__ is not None
        assert self.if_env_ready

        lattice = np.asarray(self.struct.cell)
        positions = self.struct.positions
        # atomsymbols = self.struct.get_chemical_symbols()
        # uniqsybl = get_uniq_symbol(atomsymbols=atomsymbols)

        # numbondenv = np.array(numenv)
        number_env_continer = np.sum(list(numenv.values())) * 2
        envib4 = np.zeros([number_env_continer, 4])
        ist = 0

        isite = iatom
        proj_env_site = np.asarray(self.__projenv__[isite], dtype=int)
        envitype = proj_env_site[:, 0]
        envlist = proj_env_site[:, 1]
        envRfrac = proj_env_site[:, 2:]
        envRcart = np.matmul(envRfrac, lattice)
        isitepos = self.projected_struct.positions[isite]
        envi = positions[envlist] - isitepos + envRcart
        rr = np.linalg.norm(envi, axis=1)
        envi_hat = envi / np.reshape(rr, [-1, 1])
        srr = env_smoth(rr, rcut=env_cutoff, rcut_smth=env_cutoff * 0.8)

        # for it in range(len(self.atomtype)):
        for it in self.atomtype:
            if np.sum(envitype == atomic_num_dict[it]) > 0:
                srrit = np.reshape(srr[envitype == atomic_num_dict[it]], [-1, 1])
                envi_hatit = envi_hat[envitype == atomic_num_dict[it]]
                envi_hatit2 = np.concatenate([srrit, envi_hatit], axis=1)
                envi_hatit2 = np.asarray(sorted(envi_hatit2, key=lambda s: s[0], reverse=True))

                if np.sum(envitype == atomic_num_dict[it]) > numenv[it]:
                    print('Warning!, the size of env in cutoff is larger than NumEnv parameter.')
                    ied = ist + numenv[it]
                else:
                    ied = ist + np.sum(envitype == atomic_num_dict[it])
                # print(ist,ied)
                envib4[ist:ied] = envi_hatit2[0:ied - ist]
            ist += numenv[it]

        return envib4

if __name__ == '__main__':
    from ase.build import graphene_nanoribbon
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p']}, proj_atom_neles={'C':4})

    print(basestruct.atom_to_proj_atom_id)
    print(basestruct.proj_atom_to_atom_id)