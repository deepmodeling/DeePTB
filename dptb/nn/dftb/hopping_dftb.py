from dptb.nn.sktb.hopping import BaseHopping
import torch
from dptb.utils._xitorch.interpolate import Interp1D
import logging
log = logging.getLogger(__name__)
class HoppingIntp(BaseHopping):

    def __init__(
            self,
            num_ingrls:int,
            method:str='linear',
            **kwargs,
            ) -> None:
        super().__init__()

        assert method in ['linear', 'cspline'], "Only linear and cspline are supported."
        self.functype = 'dftb'
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

        if len(rij.shape) <= 1:
            rij =  torch.tile(rij.reshape([1,-1]), (self.num_ingrls,1))
        elif len(rij.shape) == 2:
            assert rij.shape[0] == self.num_ingrls, "the bond distance shape rij is not correct."
        else:
            raise ValueError("The shape of rij is not correct.")
        # 检查 rij 是否在 xx 的范围内
        min_x, max_x = self.xx.min(), self.xx.max()
        mask_in_range = (rij >= min_x) & (rij <= max_x)
        mask_out_range = ~mask_in_range
        if mask_out_range.any():
            # log.warning("Some rij values are outside the interpolation range and will be set to 0.")
            # 创建 rij 的副本，并将范围外的值替换为范围内的值（例如，使用 min_x）
            rij_modified = rij.clone()
            rij_modified[mask_out_range] = (min_x + max_x) / 2
            yyintp = self.intpfunc(xq=rij_modified, y=yy)
            yyintp[mask_out_range] = 0.0
        else:
            yyintp = self.intpfunc(xq=rij, y=yy)

        return yyintp.T
        