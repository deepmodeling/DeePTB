import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
from dptb.utils.register import Register
from dptb.nn.hr2hk import HR2HK
from typing import Union, Dict
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper

"""this is the register class for descriptors

all descriptors inplemendeted should be a instance of nn.Module class, and provide a forward function that
takes AtomicData class as input, and give AtomicData class as output.

"""
class Loss:
    _register = Register()

    def register(target):
        return Loss._register.register(target)
    
    def __new__(cls, method: str, **kwargs):
        if method in Loss._register.keys():
            return Loss._register[method](**kwargs)
        else:
            raise Exception(f"Loss method: {method} is not registered!")


@Loss.register("eig")
class EigLoss(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
        ):
        super(EigLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.hr2hk = HR2HK(
            basis=basis,
            idp=idp,
            edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
            node_field=AtomicDataDict.NODE_FEATURES_KEY,
            out_field=AtomicDataDict.HAMILTONIAN_KEY,
            dtype=dtype, 
            device=device,
        )

        self.overlap = overlap

        if overlap:
            self.sr2sk = HR2HK(
            basis=basis,
            idp=idp,
            edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
            node_field=AtomicDataDict.NODE_OVERLAP_KEY,
            out_field=AtomicDataDict.OVERLAP_KEY,
            dtype=dtype, 
            device=device,
        )
    
    def forward(
            self, 
            data: AtomicDataDict, 
            ref_data: AtomicDataDict, 
            band_max: Union[int, torch.LongTensor],
            band_min: Union[int, torch.LongTensor],
            emax: Union[float, torch.Tensor],
            emin: Union[float, torch.Tensor]=0.,
            ):
        
        data = self.hr2hk(data)
        Heff = data[AtomicDataDict.HAMILTONIAN_KEY]
        if self.overlap:
            data = self.sr2sk(data)
        
            chklowt = torch.linalg.cholesky(data[AtomicDataDict.OVERLAP_KEY])
            chklowt = torch.linalg.inv(chklowt)
            Heff = (chklowt @ Heff @ torch.transpose(chklowt,dim0=1,dim1=2).conj())
        
        eig_pred = torch.linalg.eigvals(Heff)
        if ref_data.get(AtomicDataDict.ENERGY_EIGENVALUE_KEY) is None:
            ref_data = self.hr2hk(ref_data)
            Heff = ref_data[AtomicDataDict.HAMILTONIAN_KEY]
            if self.overlap:
                ref_data = self.sr2sk(ref_data)
                chklowt = torch.linalg.cholesky(ref_data[AtomicDataDict.OVERLAP_KEY])
                chklowt = torch.linalg.inv(chklowt)
                Heff = (chklowt @ Heff @ torch.transpose(chklowt,dim0=1,dim1=2).conj())

            eig_label = torch.linalg.eigvals(Heff)
        else:
            eig_label = ref_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY]

        norbs = eig_pred.shape[-1]
        nbanddft = eig_label.shape[-1]
        num_kp = eig_label.shape[-2]
        assert num_kp == eig_pred.shape[-2]
        up_nband = min(norbs,nbanddft)

        if band_max is  None:
            band_max = up_nband
        else:
            assert band_max <= up_nband

        band_min = int(band_min)
        band_max = int(band_max)

        assert band_min < band_max
        assert len(eig_pred.shape) == 3 and len(eig_label.shape) == 3

        # 对齐eig_pred和eig_label
        eig_pred_cut = eig_pred[:,:,band_min:band_max]
        eig_label_cut = eig_label[:,:,band_min:band_max]


        batch_size, num_kp, num_bands = eig_pred_cut.shape

        eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)
        eig_label_cut = eig_label_cut - eig_label_cut.reshape(batch_size,-1).min(dim=1)[0].reshape(batch_size,1,1)

        
        if emax != None and emin != None:
            mask_in = eig_label_cut.lt(emax) * eig_label_cut.gt(emin)
        elif emax != None:
            mask_in = eig_label_cut.lt(emax)
        elif emin != None:
            mask_in = eig_label_cut.gt(emin)
        else:
            mask_in = None

        if mask_in is not None:
            if torch.any(mask_in).item():
                loss = mse_loss(eig_pred_cut.masked_select(mask_in), eig_label_cut.masked_select(mask_in))
        else:
            loss = mse_loss(eig_pred_cut, eig_label_cut)

        return loss