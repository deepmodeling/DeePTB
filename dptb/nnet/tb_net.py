import logging
import torch
import torch.nn as nn
from dptb.utils.tools import get_uniq_symbol, get_uniq_bond_type, get_uniq_env_bond_type
from dptb.nnet.resnet import ResNet
from dptb.nnet.mlp import FFN
from dptb.utils.constants import atomic_num_dict

def _get_network(activation, config, if_batch_normalized=False, type='res', device='cpu', dtype=torch.float32):
    if type =='res':
        return ResNet(config=config, activation=activation, if_batch_normalized=if_batch_normalized, device=device, dtype=dtype)
    elif type == 'ffn':
        return FFN(config=config, activation=activation, if_batch_normalized=if_batch_normalized, device=device,
                      dtype=dtype)
    return RuntimeError("type should be mlp/res/..., not {}".format(type))

class TBNet(nn.Module):
    def __init__(self, proj_atomtype,
                 atomtype,
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
                 if_batch_normalized=False,
                 device='cpu',
                 dtype=torch.float32
                 ):
        super(TBNet, self).__init__()
        self.proj_atom_type = get_uniq_symbol(proj_atomtype)
        self.atom_type = get_uniq_symbol(atomtype)
        self.bond_type = get_uniq_bond_type(proj_atomtype)
        self.env_bond_type = get_uniq_env_bond_type(proj_atomtype, atomtype)
        self.hopping_nets = nn.ModuleDict({})
        self.onsite_nets = nn.ModuleDict({})
        self.env_nets = nn.ModuleDict({})
        if soc_net_config:
            self.soc_nets = nn.ModuleDict({})
        


        # init NNs
        # ToDo: add atom specific net_config.
        for atom in self.proj_atom_type:
            self.onsite_nets.update({
                atom:_get_network(
                config=onsite_net_config[atom],
                activation=onsite_net_activation,
                if_batch_normalized=if_batch_normalized,
                type=onsite_net_type,
                device=device,
                dtype=dtype
                )
            })
        
        if soc_net_config:
            for atom in self.proj_atom_type:
                self.soc_nets.update({
                    atom:_get_network(
                    config=soc_net_config[atom],
                    activation=soc_net_activation,
                    if_batch_normalized=if_batch_normalized,
                    type=soc_net_type,
                    device=device,
                    dtype=dtype
                    )
                })

        # ToDo: add env_bond type specific net_config.
        for env_bond in self.env_bond_type:
            self.env_nets.update({
                env_bond: _get_network(
                    config=env_net_config,
                    activation=env_net_activation,
                    if_batch_normalized=if_batch_normalized,
                    type=env_net_type,
                    device=device,
                    dtype=dtype
                )
            })
        #ToDo: add bond type specific net_config.
        for bond in self.bond_type:
            self.hopping_nets.update({
                bond:_get_network(
                config=hopping_net_config[bond],
                activation=hopping_net_activation,
                if_batch_normalized=if_batch_normalized,
                type=hopping_net_type,
                device=device,
                dtype=dtype
                )
            })

    def forward(self, x, flag, mode):
        '''

        Parameters
        ----------
        x:
            [(n_struct, 4), (Nj, 4)] when mode == bond
            where Ni, Nj is the env atoms considered for bond i-j.

            (Nk, 4) when mode == onsite
        mode:
            take values among ['bond', 'onsite', 'emb']
        flag:
            indicate what bond or atom are predicted
        Returns
        -------

        '''
        if mode == 'emb':
            # here x should be of form [(f,itype,i,jtype,j,Rx,Ry,Rz,s(r),rx,ry,rz)]
            sr = x[:,8].unsqueeze(1)
            out = self.env_nets[flag](sr)
            out = torch.cat((x, out), dim=1)

        elif mode == 'onsite':
            # x : [f,i,itype,emb_fi]
            emb = x[:,3:]
            out = self.onsite_nets[flag](emb)
            out = torch.cat((x[:,:3], out), dim=1)
            
            # out : [f,i,itype,onsite]
        elif mode == 'soc':
            # x : [f,i,itype,emb_fi]
            emb = x[:,3:]
            out = self.soc_nets[flag](emb)
            out = torch.cat((x[:,:3], out), dim=1)

        elif mode == 'hopping':
            # x : [f,itype,i,jtype,j,R,|rij|,rij_hat,emb_fij] --> [f, itype, i, jtype,j,R, |rij|, rij_hat,hopping]
            emb = torch.cat((x[:,[8]], x[:,12:]), dim=1)
            out = self.hopping_nets[flag](emb)
            out = torch.cat((x[:,:12], out), dim=1)
        else:
            return RuntimeError("mode should be emb/onsite/hopping, not {}".format(mode))

        return out



