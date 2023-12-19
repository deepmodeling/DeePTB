import torch
import numpy as np
from dptb.dataprocess.processor import Processor
from dptb.utils.constants import atomic_num_dict_r, atomic_num_dict, dtype_dict
from dptb.nnet.tb_net import TBNet
from dptb.sktb.skIntegrals import SKIntegrals
from dptb.sktb.struct_skhs import SKHSLists
from dptb.utils.tools import nnsk_correction

class NNTB(object):
    def __init__(self,
                 atomtype,
                 proj_atomtype,
                 axis_neuron,
                 env_net_config,
                 onsite_net_config,
                 hopping_net_config,
                 onsite_net_activation,
                 env_net_activation,
                 hopping_net_activation,
                 soc_net_config=None,
                 soc_net_activation=None,
                 onsite_net_type='res',
                 env_net_type='res',
                 hopping_net_type='res',
                 soc_net_type='res',
                 device='cpu',
                 dtype="float32",
                 if_batch_normalized=False,
                 **kwargs
                 ):
        
        self.dtype = dtype
        self.tb_net = TBNet(
                 proj_atomtype,
                 atomtype,
                 env_net_config,
                 onsite_net_config,
                 hopping_net_config,
                 onsite_net_activation,
                 env_net_activation,
                 hopping_net_activation,
                 soc_net_config=soc_net_config,
                 soc_net_activation=soc_net_activation,
                 onsite_net_type=onsite_net_type,
                 env_net_type=env_net_type,
                 hopping_net_type=hopping_net_type,
                 soc_net_type=soc_net_type,
                 if_batch_normalized=if_batch_normalized,
                 device=device,
                 dtype=self.dtype
                 )

        if soc_net_config:
            self.soc = True
        else:
            self.soc = False
        
        self.device = device
        self.axis_neuron = axis_neuron

    def get_desciptor(self, batch_env):
        '''> The function takes in a batch of environments, and returns a dictionary of descriptors for each
        environment
        
        Parameters
        ----------
        batch_env: dict,  for atoms i, the env is formed by its neighbour atoms.
            {env_type:(f,itype,i,jtype,j,Rx,Ry,Rz,s(r),rx,ry,rz)}, 
            where the key env type is 'center_atom_type'+ '-' + 'neighbour_atom_type' e.g. 'N-N', 'N-B', etc.
            f is the index of frames in the batch.
            i is the index of atoms in f.
            itype is atom number of the atom i. eg. 7 when the i-th atom is a nitrogen.
            s(r),rx,ry,rz is a representation of the atom i and its neighbour atom j. rij = rj-ri. postion.
        
        Returns
        -------
        batched_dcp: dict,
            {f-i:[f,i,itype,emb_fi]} 
            f-i is the key. emb_fi is the descriptor of the atom i in f.
        '''

        flag_list = batch_env.keys()
        for flag in flag_list:
            # {env_type:(f,itype, i, jtype, j,Rx,Ry,Rz s(r),rx,ry,rz)} -> {env_type:(f,itype,i,jtype,j,Rx,Ry,Rz,s(r),rx,ry,rz,emb)}
            batch_env[flag] = self.tb_net(batch_env[flag], flag=flag, mode='emb')

        batch_env = torch.cat(list(batch_env.values()), dim=0)

        batched_dcp = {}
        # rearranged by f-i
        for item in batch_env:
            name = str(int(item[0]))+'-'+str(int(item[2]))
            if batched_dcp.get(name) is None:
                batched_dcp[name] = [item]
            else:
                batched_dcp[name].append(item)

        # {f-i:[f,i,itype,emb_fi]}
        # ToDo fix_length: {}
        for flag in batched_dcp:
            data = torch.stack(batched_dcp[flag])
            # compute descriptor for f-i:
            r_norm = data[:,8:12]
            emb = torch.matmul(data[:,12:].T,r_norm) / r_norm.shape[0]
            emb = torch.matmul(emb, emb.T[:,:self.axis_neuron]).reshape(-1)
            batched_dcp[flag] = torch.cat([batched_dcp[flag][0][0:3],emb])

        self.env_out_dim = batched_dcp['0-0'].shape[0] - 3

        return batched_dcp # {f-i:[f,itype,i,emb_fi]}

    def hopping(self, batched_dcp, batch_bond):
        ''' The function takes in a descriptors batched_dcp and bond list, to get atom-descriptor to bond-descriptor 
        and  pass them to the neural network returns  batch_hoppings and the corresponding rearangerd bonds list batch_bond_hoppings
        
        Parameters
        ----------
        batched_dcp: dict
            {f-i:[f,itype, i, emb_fi]}

        batch_bond: tensor
            [f, itype, i, jtype, j, Rx, Ry, Rz, |rij|, rij_hat]
            f is the index of frames in the batch.
            i, j are the indices of atoms i and j in f. bond-ij connects i and j.
            itype, jtype are atom number of the atom i and j. eg. 7 when the i-th atom is a nitrogen.
            R the lattice vector of the bond-ij.
            |rij| is the distance between the atom i and j.
            rij_hat is the direction consine connecting the atom i and j.
        

        Returns
        -------
        batch_bond_hoppings: dict: {key: list[np.array]}
        {f:[f, itype, i, jtype,j, R, |rij|, rij_hat]}

        batch_hoppings: dict: {key: list[torch.tensor]}
        {f:[hoppings]}
        
        for the same f, the list of hoppings follows the order in batch_bond_hoppings. they have 1-1 correspondence.

        '''

        # atom-descriptor to bond-descriptor
        # batch_bond: 
        batch_bond = torch.concat(list(batch_bond.values()), dim=0)
        batch_bond = torch.cat((batch_bond, torch.zeros((batch_bond.shape[0], self.env_out_dim), dtype=self.dtype, device=self.device)), dim=1)

        batch_bond_sort = {}
        for ibond in range(len(batch_bond)):
            frameid = int(batch_bond[ibond][0])
            iatom = int(batch_bond[ibond][2])
            jatom = int(batch_bond[ibond][4])
            iatom_num = int(batch_bond[ibond][1])
            jatom_num = int(batch_bond[ibond][3])

            bondtype = atomic_num_dict_r[iatom_num]+'-'+atomic_num_dict_r[jatom_num]
            batch_bond[ibond][-self.env_out_dim:] = batch_bond[ibond][-self.env_out_dim:] + \
                                              batched_dcp[str(frameid)+'-'+str(iatom)][3:]
            batch_bond[ibond][-self.env_out_dim:] = batch_bond[ibond][-self.env_out_dim:] + \
                                              batched_dcp[str(frameid) + '-' + str(jatom)][3:]
            if batch_bond_sort.get(bondtype) is not None:
                batch_bond_sort[bondtype].append(batch_bond[ibond])
            else:
                batch_bond_sort[bondtype] = [batch_bond[ibond]]

        for bondtype in batch_bond_sort:
            dcp = torch.stack(batch_bond_sort[bondtype])
            # missing generate input tp hopping net
            # bond type for input flag to tb_net should be sorted.
            iatomtype, jatomtype = bondtype.split('-')[0], bondtype.split('-')[1]
            iatomnum, jatomnum = atomic_num_dict[iatomtype], atomic_num_dict[jatomtype]
            if iatomnum < jatomnum:
                bondflag = f'{jatomtype}-{iatomtype}'
            else: 
                bondflag = f'{iatomtype}-{jatomtype}'

            hopping = self.tb_net(dcp, flag=bondflag, mode='hopping')
            batch_bond_sort[bondtype] = hopping
                #  {bondtype: [f, itype, i, jtype,j,R, |rij|, rij_hat,hopping]}
        #batch_bond_sort = torch.cat(list(batch_bond_sort.values()), dim=0)
        batch_bond_hoppings, batch_hoppings = {}, {}
        for ibt in list(batch_bond_sort.values()):
            #print(len(ibt))
            for ib in ibt:
                f = int(ib[0])
                if batch_bond_hoppings.get(f) is not None:
                    batch_bond_hoppings[f].append(ib[0:12].detach())
                    batch_hoppings[f].append(ib[12:])
                else:
                    batch_bond_hoppings[f] = [ib[0:12].detach()]
                    batch_hoppings[f] = [ib[12:]]
                # {f:[f, itype, i, jtype,j,R, |rij|, rij_hat]}, {f:[hopping]}
        for f in batch_bond_hoppings.keys():
            batch_bond_hoppings[f] = torch.stack(batch_bond_hoppings[f])
        return batch_bond_hoppings, batch_hoppings


    def onsite(self, batched_dcp):
        '''> For each frame, we rearrange the embeddings by atom type, and then pass them to the neural network
        to get the onsite energies.
    
        Parameters
        ----------
        batched_dcp
            a dictionary of the form {f: [f, itype, i, jtype, j,  emb_fi]}
        
        Returns
        -------
        batch_bond_onsites: dict: {key: list[np.array]}
        {f:[f, itype, i, itype, i, 0, 0, 0, 0, 0 0 0]}
    
        batch_onsiteEs: dict: {key: list[torch.tensor]}
        {f:[onsiteEs]}

        '''

        # rearranged by atom type:
        # [f,itype,i,emb_fi]
        # batched_dcp = torch.stack(list(batched_dcp))
        dcp_at = {}
        soc_at = {}
        for item in list(batched_dcp.values()):
            iatomtype = atomic_num_dict_r[int(item[1])]
            if dcp_at.get(iatomtype) is None:
                dcp_at[iatomtype] = [item]
            else:
                dcp_at[iatomtype].append(item)    

        for atype in dcp_at:
            emb = torch.stack(dcp_at[atype])
            onsite = self.tb_net(emb, flag=atype, mode='onsite')
            dcp_at[atype] = onsite
            if self.soc:
                soc = self.tb_net(emb, flag=atype, mode='soc')
                soc_at[atype] = soc

            
            

        # rearrange to the output shape
        batch_onsiteEs, batch_bond_onsites = {}, {}
        for ions in list(dcp_at.values()):
            for ion in ions:
                f = int(ion[0])
                if batch_onsiteEs.get(f) is not None:
                    batch_onsiteEs[f].append(ion[3:])
                    batch_bond_onsites[f].append(torch.tensor([f, ion[1], ion[2], ion[1], ion[2], 0, 0, 0, 0, 0, 0, 0],
                                                            dtype=self.dtype, device=self.device).int())
                else:
                    batch_onsiteEs[f] = [ion[3:]]
                    batch_bond_onsites[f] = [torch.tensor([f, ion[1], ion[2], ion[1], ion[2], 0, 0, 0, 0, 0, 0, 0],
                                                        dtype=self.dtype, device=self.device).int()]
        batch_soc_lambdas = {}
        if self.soc:
            for ions in list(soc_at.values()):
                for ion in ions:
                    f = int(ion[0])
                    if batch_soc_lambdas.get(f) is not None:
                        batch_soc_lambdas[f].append(ion[3:])
                    else:
                        batch_soc_lambdas[f] = [ion[3:]]
            

        for f in batch_bond_onsites.keys():
            batch_bond_onsites[f] = torch.stack(batch_bond_onsites[f])
        
        if self.soc:
            return batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas
        else:
            return batch_bond_onsites, batch_onsiteEs, None

    def calc(self, batch_bond, batch_env):
        '''
        conduct one step forward computation, used in train, test and validation.
        '''

        batched_dcp = self.get_desciptor(batch_env)
        batch_bond_hoppings, batch_hoppings = self.hopping(batched_dcp=batched_dcp, batch_bond=batch_bond)
        
        batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas = self.onsite(batched_dcp=batched_dcp)
        return batch_bond_hoppings, batch_hoppings, batch_bond_onsites, batch_onsiteEs, batch_soc_lambdas



if __name__ == '__main__':
    a = torch.tensor([1,2,3])