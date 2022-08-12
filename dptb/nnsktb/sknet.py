import torch 
import torch.nn as nn

class SKNet(nn.Module):
    def __init__(self, skint_config, nout, interneural=None):
        ''' define the nn.parameters for fittig sktb.

        Paras 
        -----
        skint_config: 
            the keys for sk integrals, like, {'N-B':['2s-2p-sigma','2p-2p-pi',...]}
        '''
        super(SKNet, self).__init__()
        self.interneural = interneural
        self.skint_types=[]
        for ibondtype in skint_config:
            for isktype in skint_config[ibondtype]:
                self.skint_types.append(ibondtype+ '-' + isktype)
        self.num_skint_types = len(self.skint_types)

        if interneural is None:
            self.layer = torch.nn.Parameter(torch.empty(self.num_skint_types, nout))
            torch.nn.init.normal_(self.layer, mean=0, std=0.5)
        else:
            assert isinstance(interneural, int)
            self.layer1 = torch.nn.Parameter(torch.empty(self.num_skint_types, interneural))
            self.layer2 = torch.nn.Parameter(torch.empty(interneural, nout))
            torch.nn.init.normal_(self.layer1, mean=0, std=0.5)
            torch.nn.init.normal_(self.layer2, mean=0, std=0.5)
        
    def forward(self):
        '''> The function takes in a list of skin types and a list of coefficients, and returns a dictionary of
        skin types and their corresponding coefficients
        
        Returns
        -------
            A dictionary of skin types and their corresponding coefficients. like:
            {
                'N-B-2s-2s-sigma': tensor([-0.0741,  0.6850,  0.6343,  0.5956], grad_fn=<UnbindBackward0>),
                'B-B-2s-2s-sigma': tensor([ 4.1594e-02,  1.6971e+00, -1.7270e-05, -3.4321e-01], grad_fn=<UnbindBackward0>)
                ...
            }       
        
        '''

        if self.interneural is None:
            out = self.layer
        else:
            out =  self.layer1 @ self.layer2
        
        self.coeffdict = dict(zip(self.skint_types, out))

        return self.coeffdict 
    
    def get_coeff(self, skint_type):
        if not hasattr(self, 'coeffdict'):
            self.forward()
        
        return self.coeffdict[skint_type]



    