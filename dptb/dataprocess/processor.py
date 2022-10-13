import numpy as np
import torch
from typing import List
from dptb.utils.constants import dtype_dict
from dptb.structure.abstract_stracture import AbstractStructure

class Processor(object):
    # TODO: 现在strain的env 是通过get_env 获得，但是在dptb中的env是有另外的含义。是否已经考虑。
    def __init__(self, structure_list: List[AbstractStructure], kpoint, eigen_list, batchsize: int, env_cutoff: float = 3.0, onsitemode=None, onsite_cutoff=None, sorted_bond=None, sorted_onsite=None, sorted_env=None, device='cpu', dtype=torch.float32):
        super(Processor, self).__init__()
        if isinstance(structure_list, AbstractStructure):
            structure_list = [structure_list]
        self.structure_list = np.array(structure_list, dtype=object)
        self.kpoint = kpoint
        self.eigen_list = np.array(eigen_list, dtype=object)
        self.sorted_bond = sorted_bond
        self.sorted_env = sorted_env
        self.sorted_onsite = sorted_onsite
        self.onsite_cutoff = onsite_cutoff
        self.onsitemode = onsitemode

        self.n_st = len(self.structure_list)
        self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                     size=len(self.structure_list), replace=False)

        self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        self.__struct_workspace__ = []
        self.__struct_idx_workspace__ = []
        self.env_cutoff = env_cutoff
        self.batchsize = batchsize
        self.n_batch = int(self.n_st / batchsize)
        if self.n_st % batchsize:
            self.n_batch += 1

        assert self.batchsize > 0

        self.device = device
        self.dtype = dtype_dict[dtype]

    def shuffle(self):
        '''> If the batch size is larger than the number of unsampled structures, then we sample all the
        remaining structures and reset the unsampled list to the full list of structures. Otherwise, we
        sample the first `batch_size` structures from the unsampled list and remove them from the unsampled list
        
        Parameters
        ----------
        batch_size int : number of structures to be sampled        
        '''

        if self.batchsize >= len(self.__struct_unsampled__):
            self.__struct_workspace__ = self.__struct_unsampled__
            self.__struct_idx_workspace__ = self.__struct_idx_unsampled__
            self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                         size=len(self.structure_list), replace=False)
            self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        else:
            self.__struct_workspace__ = self.__struct_unsampled__[:self.batchsize]
            self.__struct_idx_workspace__ = self.__struct_idx_unsampled__[:self.batchsize]
            self.__struct_idx_unsampled__ = self.__struct_idx_unsampled__[self.batchsize:]
            self.__struct_unsampled__ = self.__struct_unsampled__[self.batchsize:]
    
    def get_env(self, cutoff=None, sorted=None):
        # TODO: the sorted mode should be explained here, in which case, we should use.
        '''It takes the environment of each structure in the workspace and concatenates them into one big
        environment
        
        Returns
        -------
            A dictionary of the environment for ent type for all the strucutes in  the works sapce.
        '''
        
        if len(self.__struct_workspace__) == 0:
            self.__struct_workspace__ = self.structure_list
        n_stw = len(self.__struct_workspace__)

        if cutoff is None:
            cutoff = self.env_cutoff
        else:
            assert isinstance(cutoff, float)
        
        if sorted is None:
            batch_env = []
            for st in range(n_stw):
                env = self.__struct_workspace__[st].get_env(env_cutoff=cutoff, sorted=sorted)
                assert len(env) > 0, "This structure has no environment atoms."
                batch_env.append(torch.tensor(np.concatenate([np.ones((env.shape[0], 1))*st, env], axis=1), dtype=self.dtype, device=self.device)) # numpy to tensor
            batch_env = torch.cat(batch_env, dim=0)
        
        elif sorted == "itype-jtype":
            batch_env = {}
            for st in range(n_stw):
                env = self.__struct_workspace__[st].get_env(env_cutoff=cutoff, sorted="itype-jtype")
                # (i,itype,s(r),rx,ry,rz)
                for ek in env.keys():
                    # to envalue the order for each structure of the envs.
                    env_ek = np.concatenate([np.ones((env[ek].shape[0], 1))*st, env[ek]], axis=1)

                    if batch_env.get(ek) is None:
                        batch_env[ek] = env_ek
                    else:
                        batch_env[ek] = np.concatenate([batch_env[ek], env_ek], axis=0)

            for ek in batch_env:
                batch_env[ek] = torch.tensor(batch_env[ek], dtype=self.dtype, device=self.device)

        elif sorted == "st":
            batch_env = {}
            for st in range(n_stw):
                env = self.__struct_workspace__[st].get_env(env_cutoff=cutoff, sorted=None)
                assert len(env) > 0, "This structure has no environment atoms."
                batch_env[st] = torch.tensor(np.concatenate([np.ones((env.shape[0], 1))*st, env], axis=1), dtype=self.dtype, device=self.device) # numpy to tensor

        else:
            raise NotImplementedError

        return batch_env # {env_type: (itype, i, jtype, j, jtype, Rx, Ry, Rz, s(r), rx, ry, rz)} or [(f, itype, i, jtype, j, jtype, Rx, Ry, Rz, s(r), rx, ry, rz)]

    def get_bond(self, sorted=None):
        '''It takes the bonds of each structure in the workspace and concatenates them into one big dictionary.
        
        Returns
        -------
            A Tensor of the bonds lists for bond type for all the strucutes in the works space.
        '''
        # ToDo: Remove require_dict options, unify the data structure.

        if len(self.__struct_workspace__) == 0:
            self.__struct_workspace__ = self.structure_list

        if sorted is None:
            batch_bond = []
            batch_bond_onsite = []
            n_stw = len(self.__struct_workspace__)
            for st in range(n_stw):
                bond, bond_onsite = self.__struct_workspace__[st].get_bond()
                bond = np.concatenate([np.ones((bond.shape[0], 1)) * st, bond], axis=1)
                bond_onsite = np.concatenate([np.ones((bond_onsite.shape[0], 1)) * st, bond_onsite], axis=1)
                batch_bond.append(torch.tensor(bond, dtype=self.dtype, device=self.device))
                batch_bond_onsite.append(torch.tensor(bond_onsite, dtype=self.dtype, device=self.device))

            batch_bond = torch.cat(batch_bond, dim=0)
            batch_bond_onsite = torch.cat(batch_bond_onsite, dim=0)

        elif sorted == "st":
            batch_bond = {}
            batch_bond_onsite = {}
            n_stw = len(self.__struct_workspace__)
            for st in range(n_stw):
                bond, bond_onsite = self.__struct_workspace__[st].get_bond()
                bond = np.concatenate([np.ones((bond.shape[0], 1)) * st, bond], axis=1)
                bond_onsite = np.concatenate([np.ones((bond_onsite.shape[0], 1)) * st, bond_onsite], axis=1)
                batch_bond.update({st:torch.tensor(bond, dtype=self.dtype, device=self.device)})
                batch_bond_onsite.update({st:torch.tensor(bond_onsite, dtype=self.dtype, device=self.device)})
            

        return batch_bond, batch_bond_onsite # [f, i_atom_num, i, j_atom_num, j, Rx, Ry, Rz, |rj-ri|, \hat{rij: x, y, z}] or dict

    @property
    def atomtype(self):
        '''It returns a list of unique atom types in the structure.
        
        Returns
        -------
            A list of unique atom types.
        '''
        at_list = []
        for st in self.structure_list:
            at_list += st.atomtype

        return list(set(at_list))

    @property
    def proj_atomtype(self):
        ''' This function returns a list of all the projected atom types in the structure list
        
        Returns
        -------
            A list of unique atom types in the structure list.
        '''
        at_list = []
        for st in self.structure_list:
            at_list += st.proj_atomtype

        return list(set(at_list))

    def __iter__(self):
        # processor = Processor; for i in processor: i: (batch_bond, batch_env, structures)
        self.it = 0 # label of iteration
        self.__struct_idx_unsampled__ = np.random.choice(np.array(list(range(len(self.structure_list)))),
                                                         size=len(self.structure_list), replace=False)

        self.__struct_unsampled__ = self.structure_list[self.__struct_idx_unsampled__]
        self.__struct_workspace__ = []
        self.__struct_idx_workspace__ = []
        return self

    def __next__(self):
        if self.it < self.n_batch:
            self.shuffle()
            bond, bond_onsite = self.get_bond(self.sorted_bond)

            if not self.onsitemode == 'strain':
                data = (bond, bond_onsite, self.get_env(sorted=self.sorted_env), None,  self.__struct_workspace__,
                    self.kpoint, self.eigen_list[self.__struct_idx_workspace__].astype(float))
            else:
                data = (bond, bond_onsite, self.get_env(sorted=self.sorted_env), self.get_env(cutoff=self.onsite_cutoff, sorted=self.sorted_onsite), self.__struct_workspace__,
                    self.kpoint, self.eigen_list[self.__struct_idx_workspace__].astype(float))

            self.it += 1
            return data
        else:
            raise StopIteration

    def __len__(self):
        return self.n_batch

    def atom_rearrangement(self, input):
        # input
        pass

if __name__ == '__main__':
    from ase.build import graphene_nanoribbon
    from dptb.structure.structure import BaseStruct
    atoms = graphene_nanoribbon(1.5, 1, type='armchair', saturated=True)
    basestruct = BaseStruct(atom=atoms, format='ase', cutoff=1.5, proj_atom_anglr_m={'C': ['s', 'p']}, proj_atom_neles={"C":4})

    p = Processor(mode = 'dptb', structure_list=[basestruct, basestruct, basestruct, basestruct], kpoint=1, eigen_list=[1,2,3,4], batchsize=1, env_cutoff=1.5)

    count = 0
    for data in p:
        print(count)
        count += 1
        if count == 2:
            break

    count = 0
    for data in p:
        print(count)
        count+=1
