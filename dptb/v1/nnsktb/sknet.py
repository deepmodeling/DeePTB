import torch 
import torch.nn as nn


class DirectNet(nn.Module):
    def __init__(self, nin, nhidden, nout, device='cpu', dtype=torch.float32, ini_std=0.5, **kwargs):
        super().__init__()
        assert nout is not None, "nout must be specified!"
        self.nhidden = nhidden
        self.layer1 = torch.nn.Parameter(torch.empty(nin, 1, nhidden, device=device, dtype=dtype),requires_grad=False)
        self.layer2 = torch.nn.Parameter(torch.empty(nin, nout, nhidden, device=device, dtype=dtype))
        #torch.nn.init.normal_(self.layer1, mean=0, std=ini_std)
        torch.nn.init.ones_(self.layer1)
        torch.nn.init.normal_(self.layer2, mean=0, std=ini_std)
    
    def forward(self):

        return (self.layer1 * self.layer2).sum(dim=2) / self.nhidden
        # return self.layer1 @ self.layer2 / self.nhidden


class SKNet(nn.Module):
    def __init__(self, skint_types: list, onsite_types:dict, soc_types: dict, hopping_neurons: dict, onsite_neurons: dict, soc_neurons: dict=None, 
                        onsite_index_dict:dict=None, onsitemode:str='none', overlap=False, device='cpu', dtype=torch.float32, **kwargs):
        ''' define the nn.parameters for fittig sktb.

        Paras 
        -----
        skint_types: 
            the keys for sk integrals, like, ['N-N-2s-2p-sigma','N-N-2p-2p-pi',...]

        skint_types: list
            the independent/reduced sk integral types, like, ['N-N-2s-2p-0','N-N-2p-2p-0',...]
        onsiteE_types: list
            the independent/reduced onsite Energy types. like ['N-2s-0', 'N-2p-0', 'B-2s-0'],
        onsiteint_types: list   
            the independent/reduced sk-like onsite integral types. like [''N-N-2s-2s-0',...]
        
        hopping_neurons: dict
            {'nhidden':int, 'nout':int}
        # Note: nout 是拟合公式中的待定参数。比如 varTang96 formula nout = 4. 

        onsite_neurons:dict
            {'nhidden':int}

        soc_neurons: dict
            {'nhidden':int}
        
        '''

        super().__init__()
        assert len(set(skint_types)) == len(skint_types), "the values in skint_types in not unique."
        assert skint_types is not None, "skint_types cannot be None"

        self.skint_types = skint_types 
        self.sk_options = kwargs.get("sk_options")
        self.onsitemode = onsitemode
        self.onsite_types = onsite_types
        self.soc_types = soc_types
        self.onsite_index_dict = onsite_index_dict
        self.overlap = overlap

        self.nhop_paras = hopping_neurons.get('nout')

        if overlap:

            self.noverlap_paras = hopping_neurons['nout_overlap']

            hopping_config = {
                'nin': len(self.skint_types),
                'nhidden': hopping_neurons.get('nhidden',1),
                'nout': self.nhop_paras + self.noverlap_paras,
                'ini_std':0.001}
        else:
            hopping_config = {
                'nin': len(self.skint_types),
                'nhidden': hopping_neurons.get('nhidden',1),
                'nout': hopping_neurons.get('nout'),
                'ini_std':0.001}
        self.hopping_net = DirectNet(device=device, dtype=dtype, **hopping_config)
        
        if self.onsitemode.lower() == 'none':
            pass
        elif self.onsitemode.lower() == 'strain':
            assert onsite_types is not None, "for strain mode, the onsiteint_types can not be None!"
            onsite_config = {
                'nin': len(self.onsite_types),
                'nhidden': onsite_neurons.get('nhidden',1),
                'nout': onsite_neurons.get('nout'),
                'ini_std':0.01}
            
            # Note: 这里onsite integral 选取和bond integral一样的公式，因此是相同的 nout.

            self.onsite_net = DirectNet(device=device, dtype=dtype, **onsite_config)
        else:
            # only support the uniform for this mode.
            assert onsite_types is not None, f"for {onsitemode} mode, onsiteE_types can not be None!"
            assert onsite_index_dict is not None, f"for {onsitemode} mode, onsiteE_index_dict can not be None!"

            onsite_config = {
                'nin': len(self.onsite_types),
                'nhidden': onsite_neurons.get('nhidden',1),
                'nout': onsite_neurons.get('nout',1),
                'ini_std':0.01}

            self.onsite_net = DirectNet(**onsite_config)
        
        if soc_neurons is not None:
            assert soc_types is not None

            soc_config = {
                'nin': len(self.soc_types),
                'nhidden': soc_neurons.get('nhidden',1),
                'nout': 1,
                'ini_std':0.01
            }

            self.soc_net = DirectNet(**soc_config)

        
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
            out = self.hopping_net()
            if self.overlap:
                self.hop_coeffdict = dict(zip(self.skint_types, out[:,:self.nhop_paras]))
                self.overlap_coeffdict = dict(zip(self.skint_types, out[:,self.nhop_paras:self.nhop_paras+self.noverlap_paras]))
            else:
                self.hop_coeffdict = dict(zip(self.skint_types, out))
                self.overlap_coeffdict = None
            return self.hop_coeffdict, self.overlap_coeffdict
        
        elif mode == 'soc':
            out = self.soc_net()
            out = out.abs()
            self.soc_values = dict(zip(self.soc_types, out))

            self.soc_value = {}
            for ia in self.onsite_index_dict:
                self.soc_value[ia] = torch.stack([self.soc_values[itag][0]  for itag in self.onsite_index_dict[ia]])

            return self.soc_value, None

        elif mode == 'onsite':
            """ two outputs, 1: for orbital enegy 2: for onsite integral.
                - the onsite integral is used to calculate the onsite matrix through SK transformation.
                - the orbital energy is just the onsite energy which is the diagonal elements of the onsite matrix.
                - for uniform mode, the output of nn is directly used as the onsite value.
                - for strain mode, the output of nn is used as the coefficient to multiply the onsite integral formula like the sk integral.
                - for other modes, the output of nn is used as a coefficient to multiply the onsite energy using a formula.
            """
            if self.onsitemode.lower() == 'none':
                return None, None
            elif self.onsitemode.lower() == 'strain':
                # in strain mode, the output of nn is used as the coefficient to multiply the onsite integral formula like the sk integral.
                out = self.onsite_net()
                self.onsite_coeffdict = dict(zip(self.onsite_types, out))
                return None, self.onsite_coeffdict
            elif self.onsitemode.lower() in ['uniform','split']:
                # the out put of nn is directly used as the onsite value.
                # output format e.g.: {'N':[es,ep],'B':[es,ep]}
                out = self.onsite_net()
                self.onsite_paras = dict(zip(self.onsite_types, out))

                self.onsite_value_formated = {}
                for ia in self.onsite_index_dict:
                    self.onsite_value_formated[ia] = torch.stack([self.onsite_paras[itag][0]  for itag in self.onsite_index_dict[ia]])
                return self.onsite_value_formated, None
            else:
                # the output of nn is used as a coefficient to multiply the onsite energy using a formula.
                # this formula is different from the onsite integral formula. and it directly gives the onsite energy.
                # the onsite integral will still need to sk transformation to be onsite matrix. 
                # output format e.g.: {'N-2s-0':[...],
                #                      'N-2s-0':[...],
                #                      'B-2s-0':[...],
                #                      'B-2p-0':[...]}
                # [...] vector: means the output coefficients for the orbital energy formula.
                out = self.onsite_net()
                self.onsite_paras = dict(zip(self.onsite_types, out))
                return self.onsite_paras, None     
        else:
            raise ValueError(f'Invalid mode: {mode}')
        
    
    def get_hop_coeff(self, skint_type):
        if not hasattr(self, 'hop_coeffdict'):
            self.forward(mode='hopping')
        return self.hop_coeffdict[skint_type]