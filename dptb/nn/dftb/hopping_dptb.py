from dptb.nn.sktb.hopping import BaseHopping
import torch
from dptb.utils._xitorch.interpolate import Interp1D

class HoppingIntp(BaseHopping):

    def __init__(
            self,
            xdist:torch.tensor,
            num_ingrls:int,
            method:str='linear',
            **kwargs,
            ) -> None:
        super().__init__()

        assert method in ['linear', 'cspline'], "Only linear and cspline are supported."

        xx = torch.tile(xdist.reshape([1,-1]), (num_ingrls,1))
        self.num_ingrls = num_ingrls
        self.intpfunc = Interp1D(xx, method=method)


    def get_skhij(self, rij, **kwargs):
        
        return self.dftb(rij, **kwargs)
    
    def dftb(self, rij:torch.Tensor, yy:torch.Tensor, **kwargs):

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
        