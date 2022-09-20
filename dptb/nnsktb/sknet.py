from re import A
import torch 
import torch.nn as nn
import numpy as np

class DirectNet(nn.Module):
    def __init__(self, nin, nhidden, nout, device='cpu', dtype=torch.float32, **kwargs):
        super().__init__()
        assert nout is not None, "nout must be specified!"
        self.nhidden = nhidden
        self.layer1 = torch.nn.Parameter(torch.empty(nin, nhidden, device=device, dtype=dtype))
        self.layer2 = torch.nn.Parameter(torch.empty(nhidden, nout, device=device, dtype=dtype))
        torch.nn.init.normal_(self.layer1, mean=0, std=0.5)
        torch.nn.init.normal_(self.layer2, mean=0, std=0.5)
    
    def forward(self):
        return self.layer1 @ self.layer2 / self.nhidden



class SKNet(nn.Module):
    def __init__(self, skint_types: list, onsite_num: dict, bond_neurons: dict, onsite_neurons: dict, device='cpu', dtype=torch.float32, **kwargs):
        ''' define the nn.parameters for fittig sktb.

        Paras 
        -----
        skint_types: 
            the keys for sk integrals, like, ['N-N-2s-2p-sigma','N-N-2p-2p-pi',...]
        onsite_num: dict
            {'N':4,'B':4}
        
        bond_neurons: dict
            {'nhidden':int, 'nout':int}

        onsite_neurons:dict
            {'nhidden':int}
        
        '''

        super().__init__()
        assert len(set(skint_types)) == len(skint_types), "the values in skint_types in not unique."
        self.skint_types = skint_types
        self.num_skint_types = len(self.skint_types)
        self.onsite_num = onsite_num

        bond_config = {
            'nin': len(self.skint_types),
            'nhidden': bond_neurons.get('nhidden',1),
            'nout': bond_neurons.get('nout')
        }

        onsite_config = {}
        for ia in self.onsite_num:
            onsite_config[ia] = {
                'nin':1,
                'nhidden': onsite_neurons.get('nhidden',1),
                'nout': self.onsite_num[ia]
            }
        
        self.bond_net = DirectNet(**bond_config)
        
        self.onsite_net = nn.ModuleDict({})
        for ia in self.onsite_num:
            self.onsite_net.update({
                ia: DirectNet(**onsite_config[ia])
                })


        
    def forward(self, mode: str):
        '''> The function takes in a list of skin types and a list of coefficients, and returns a dictionary of
        skin types and their corresponding coefficients
        
        The function is called by the `forward` function of the `NN_model` class
        
        Parameters
        ----------
        mode : str
            'hopping' or 'onsite'
        
        Returns
        -------
            if mode is 'hopping'
                Dict:  
                A dictionary of skin types and their corresponding coefficients. like:
                {
                    'N-B-2s-2s-sigma': tensor([-0.0741,  0.6850,  0.6343,  0.5956], grad_fn=<UnbindBackward0>),
                    'B-B-2s-2s-sigma': tensor([ 4.1594e-02,  1.6971e+00, -1.7270e-05, -3.4321e-01], grad_fn=<UnbindBackward0>)
                    ...
                }   
            if mode is 'onsite' 
                Dict:
                The onsite energys: 
                { 
                    'N': tensor([es,ep,....]) or tensor([es,ep1,ep2,ep3,...])  the dependts on the parameters: `onsite_num`.
                    ...
                }
        '''
        
        if mode == 'hopping':
            out = self.bond_net()
            self.hop_coeffdict = dict(zip(self.skint_types, out))
            return self.hop_coeffdict
        elif mode == 'onsite':
            self.onsite_value = {}
            for ia in self.onsite_num:
                out = self.onsite_net[ia]()
                self.onsite_value[ia] = torch.reshape(out,[-1]) # {"N":[s, p, ...]}
            
            return self.onsite_value
        else:
            raise ValueError('Invalid mode: ' + mode)
        
    
    def get_hop_coeff(self, skint_type):
        if not hasattr(self, 'hop_coeffdict'):
            self.forward(mode='hopping')
        return self.hop_coeffdict[skint_type]


    
