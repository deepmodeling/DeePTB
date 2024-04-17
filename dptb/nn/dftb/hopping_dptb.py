from dptb.nn.sktb.hopping import BaseHopping
import torch
from dptb.utils._xitorch.interpolate import Interp1D

class HoppingIntp(BaseHopping):

    def __init__(
            self,
            num_ingrls:int,
            method:str='linear',
            **kwargs,
            ) -> None:
        super().__init__()

        assert method in ['linear', 'cspline'], "Only linear and cspline are supported."

        self.num_ingrls = num_ingrls
        self.intp_method = method   

    def get_skhij(self, rij, **kwargs):
        
        return self.dftb(rij, **kwargs)
    
    def dftb(self, rij:torch.Tensor, xx:torch.Tensor, yy:torch.Tensor, **kwargs):  
        if not hasattr(self, 'intpfunc'):  #or torch.max(torch.abs(self.xx - xx)) > 1e-5:
            self.xx = xx
            xx = xx.reshape(1, -1).repeat(self.num_ingrls, 1)
            self.intpfunc = Interp1D(xx, method=self.intp_method)
        
        assert yy.shape[0] == self.num_ingrls
        assert len(yy.shape) == 2

        if len(rij.shape) == 1:
            rij =  torch.tile(rij.reshape([1,-1]), (self.num_ingrls,1))
        elif len(rij.shape) == 2:
            assert rij.shape[0] == self.num_ingrls, "the bond distance shape rij is not correct."
        else:
            raise ValueError("The shape of rij is not correct.")

        yyintp = self.intpfunc(xq=rij,y=yy)
        
        return yyintp.T
        