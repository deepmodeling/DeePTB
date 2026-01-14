import re
from typing import Union, Dict, Optional, List

import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch_scatter import scatter_mean
import matplotlib.pyplot as plt
from e3nn.o3 import Irreps

from dptb.nn.dftbsk import DFTBSK
from dptb.nn.energy import Eigenvalues
from dptb.nn.hamiltonian import E3Hamiltonian
from dptb.data import AtomicDataDict, AtomicData
from dptb.data.transforms import OrbitalMapper
from dptb.utils.torch_geometric import Batch
from dptb.utils.constants import anglrMId
from dptb.utils.register import Register

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

@Loss.register("skints")
class DFTBskLoss(nn.Module):
    def __init__(
                self,
                basis: Dict[str, Union[str, list]]=None,
                skdata: str=None,
                overlap: bool = False,
                dtype: Union[str, torch.dtype] = torch.float32, 
                device: Union[str, torch.device] = torch.device("cpu"),
                **kwargs) -> None:
        
        super().__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        
        self.loss = nn.MSELoss()

        self.dftbsk = DFTBSK(basis=basis, skdata=skdata, overlap=overlap, dtype=dtype, device=device,transform=False)

        self.overlap = overlap
    
    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        total_loss = 0.
        ref_data = AtomicData.to_AtomicDataDict(ref_data)
        ref_data = self.dftbsk(ref_data)

        # onsite loss
        onsite_loss = mse_loss(data[AtomicDataDict.NODE_FEATURES_KEY], ref_data[AtomicDataDict.NODE_FEATURES_KEY])

        # hopping loss
        hopping_loss = mse_loss(data[AtomicDataDict.EDGE_FEATURES_KEY], ref_data[AtomicDataDict.EDGE_FEATURES_KEY])

        # overlap loss
        total_loss = onsite_loss + hopping_loss
        if self.overlap:
            total_loss = total_loss + mse_loss(data[AtomicDataDict.EDGE_OVERLAP_KEY], ref_data[AtomicDataDict.EDGE_OVERLAP_KEY])
        
        return total_loss

@Loss.register("eigvals")
class EigLoss(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            diff_on: bool=False,
            eout_weight: float=0.01,
            diff_weight: float=0.01,
            diff_valence: dict=None,
            spin_deg: int = 2,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):
        super(EigLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.device = device
        self.diff_on = diff_on
        self.eout_weight = eout_weight
        self.diff_weight = diff_weight
        self.diff_valence = diff_valence  
        self.spin_deg = spin_deg  


        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        if not overlap:
            self.eigenvalue = Eigenvalues(
                idp=self.idp,
                h_edge_field = AtomicDataDict.EDGE_FEATURES_KEY,
                h_node_field = AtomicDataDict.NODE_FEATURES_KEY,
                h_out_field = AtomicDataDict.HAMILTONIAN_KEY,
                out_field = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
                s_edge_field = None,
                s_node_field = None,
                s_out_field = None, 
                dtype=dtype, 
                device=device,
                )
        else:
            self.eigenvalue = Eigenvalues(
                idp=self.idp,
                h_edge_field = AtomicDataDict.EDGE_FEATURES_KEY,
                h_node_field = AtomicDataDict.NODE_FEATURES_KEY,
                h_out_field = AtomicDataDict.HAMILTONIAN_KEY,
                out_field = AtomicDataDict.ENERGY_EIGENVALUE_KEY,
                s_edge_field = AtomicDataDict.EDGE_OVERLAP_KEY,
                s_node_field = AtomicDataDict.NODE_OVERLAP_KEY,
                s_out_field = AtomicDataDict.OVERLAP_KEY, 
                dtype=dtype, 
                device=device,
                )

        self.overlap = overlap
    
    def forward(
            self, 
            data: AtomicDataDict, 
            ref_data: AtomicDataDict,
            ):
        
        total_loss = 0.

        data = Batch.from_dict(data)
        ref_data = Batch.from_dict(ref_data)

        datalist = data.to_data_list()
        ref_datalist = ref_data.to_data_list()
        for data, ref_data in zip(datalist, ref_datalist):
            data = self.eigenvalue(AtomicData.to_AtomicDataDict(data))
            ref_data = AtomicData.to_AtomicDataDict(ref_data)
            if ref_data.get(AtomicDataDict.ENERGY_EIGENVALUE_KEY) is None:
                ref_data = self.eigenvalue(ref_data)
            
            emin, emax = ref_data.get(AtomicDataDict.ENERGY_WINDOWS_KEY, (None, None))
            band_min, band_max = ref_data.get(AtomicDataDict.BAND_WINDOW_KEY, (0, None))
            eig_pred = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # (n_kpt, n_band)
            eig_label = ref_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY][0] # (n_kpt, n_band_dft/n_band)

            if self.diff_valence is not None and isinstance(self.diff_valence, dict):
                nbands_exclude = sum([self.diff_valence[self.idp.type_to_chemical_symbol[int(ii)]] for ii in ref_data['atom_types']])
                assert nbands_exclude % self.spin_deg == 0
                nbands_exclude = nbands_exclude // self.spin_deg
            else:
                nbands_exclude = 0
            
            eig_label = eig_label[:,nbands_exclude:]

            norbs = eig_pred.shape[-1]
            nbanddft = eig_label.shape[-1]
            num_kp = eig_label.shape[-2]

            assert num_kp == eig_pred.shape[-2]
            up_nband = min(norbs, nbanddft)

            if band_max == None:
                band_max = up_nband
            else:
                assert band_max <= up_nband

            band_min = int(band_min)
            band_max = int(band_max)

            assert band_min < band_max
            assert len(eig_pred.shape) == 2 and len(eig_label.shape) == 2

            # 对齐eig_pred和eig_label
            eig_pred_cut = eig_pred[:,band_min:band_max]
            eig_label_cut = eig_label[:,band_min:band_max]


            num_kp, num_bands = eig_pred_cut.shape

            eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(-1).min()
            eig_label_cut = eig_label_cut - eig_label_cut.reshape(-1).min()

            
            if emax != None and emin != None:
                mask_in = eig_label_cut.lt(emax) * eig_label_cut.gt(emin)
                mask_out = eig_label_cut.gt(emax) + eig_label_cut.lt(emin)
            elif emax != None:
                mask_in = eig_label_cut.lt(emax)
                mask_out = eig_label_cut.gt(emax)
            elif emin != None:
                mask_in = eig_label_cut.gt(emin)
                mask_out = eig_label_cut.lt(emin)
            else:
                mask_in = None
                mask_out = None

            if mask_in is not None:
                if torch.any(mask_in).item():
                    loss = mse_loss(eig_pred_cut.masked_select(mask_in), eig_label_cut.masked_select(mask_in))
                if torch.any(mask_out).item():
                    loss = loss + self.eout_weight * mse_loss(eig_pred_cut.masked_select(mask_out), eig_label_cut.masked_select(mask_out))
            else:
                loss = mse_loss(eig_pred_cut, eig_label_cut)

            if self.diff_on:
                assert num_kp >= 1
                # randon choose nk_diff kps' eigenvalues to gen Delta eig.
                # nk_diff = max(nkps//4,1)     
                nk_diff = num_kp
                k_diff_i = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                k_diff_j = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                while (k_diff_i==k_diff_j).all():
                    k_diff_j = torch.randint(0, num_kp, (nk_diff,), device=self.device)
                if mask_in is not None:
                    eig_diff_lbl = eig_label_cut.masked_fill(mask_in, 0.)[:, k_diff_i,:] - eig_label_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
                    eig_ddiff_pred = eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_i,:] - eig_pred_cut.masked_fill(mask_in, 0.)[:,k_diff_j,:]
                else:
                    eig_diff_lbl = eig_label_cut[:,k_diff_i,:] - eig_label_cut[:,k_diff_j,:]
                    eig_ddiff_pred = eig_pred_cut[:,k_diff_i,:]  - eig_pred_cut[:,k_diff_j,:]
                loss_diff =  mse_loss(eig_diff_lbl, eig_ddiff_pred) 
                
                loss = loss + self.diff_weight * loss_diff

            total_loss += loss

        return total_loss / len(datalist)

@Loss.register("dos")
class DOSLoss(nn.Module):
    """
    Use the linalg.norm of Density-of-states (DOS) difference as the loss.
    In this case, the weights of kpoints are of need.
    """
    # rename some useful keys to shorter names
    EDGE_          = AtomicDataDict.EDGE_FEATURES_KEY
    EDGE_OVLP_     = AtomicDataDict.EDGE_OVERLAP_KEY
    NODE_          = AtomicDataDict.NODE_FEATURES_KEY
    NODE_OVLP_     = AtomicDataDict.NODE_OVERLAP_KEY
    HAMILT_        = AtomicDataDict.HAMILTONIAN_KEY
    OVLP_          = AtomicDataDict.OVERLAP_KEY
    EIGVAL_        = AtomicDataDict.ENERGY_EIGENVALUE_KEY
    ENERGY_WINDOW_ = AtomicDataDict.ENERGY_WINDOWS_KEY
    BAND_WINDOW_   = AtomicDataDict.BAND_WINDOW_KEY
    WK_            = AtomicDataDict.WEIGHT_KPOINT_KEY

    @staticmethod
    def calc_dos(ekb, wk, emin, emax, de, sigma=0.1):
        '''calculate the dos by convolution with a Gaussian pulse
        
        Parameters
        ----------
        ekb : torch.Tensor
            Eigenvalues of the kpoints, shape (nk, nb).
        wk : torch.Tensor
            Weights of the kpoints, shape (nk,).
        emin : float
            Minimum energy for the DOS calculation.
        emax : float
            Maximum energy for the DOS calculation.
        de : float
            Energy interval for the DOS calculation.
        
        Returns
        -------
        dos : torch.Tensor
            Density-of-states, shape (nbin,).
        '''
        # the direct implementation of the torch.histogram is not differentiable,
        # here we use a convolution technique to calculate the dos:
        # for each eigenvalue, add a pulse peak onto the dos, the peak is also
        # multiplied by the weight of the kpoint. After transversing all kpoints,
        # the dos is obtained :)
        def pulse(x, x0, sigma):
            '''a standard Gaussian pulse function'''
            return torch.exp(-(x - x0)**2 / (2 * sigma**2))

        # sanity check
        assert isinstance(ekb, torch.Tensor)
        assert isinstance(wk, torch.Tensor)
        nk1, nb = ekb.shape
        nk, = wk.shape
        assert nk1 == nk # assume a simple correspondence
        assert ekb.device == wk.device

        assert isinstance(emin, float)
        assert isinstance(emax, float)
        assert emin < emax

        assert isinstance(de,   float)
        assert isinstance(sigma, float)

        # calculate the dos by convolution
        nbin = int((emax - emin) / de)
        erange = torch.linspace(emin, emax, nbin, device=ekb.device)
        # to benefit from the torch broadcast feature, we reshape
        erange_ = erange.view(1, 1, nbin)
        ekb_    = ekb.view(nk, nb, 1)
        wk_     = wk.view(nk, 1, 1)

        dos = (wk_ * pulse(erange_, ekb_, sigma)).sum(dim=(0, 1))
        return dos / torch.trapz(dos, erange.squeeze())

    def __init__(self, 
                 basis: Optional[Dict[str, Union[str, List]]] = None,
                 idp: Optional[OrbitalMapper] = None,
                 overlap: bool = False,
                 degauss: float = 0.1,
                 de: float = 0.1,
                 spin_deg: int = 2,
                 dtype: Union[str, torch.dtype] = torch.float32,
                 device: Union[str, torch.device] = torch.device("cpu"),
                 **kwargs):
        '''
        Initiatiate a DOS Loss, which measures the difference in the Density-of-States (DOS)
        between the predicted and the labelled

        Parameters
        ----------
        basis: Optional[Dict[str, Union[str, List]]]
            Basis set for the calculation of DOS.
        idp: Optional[OrbitalMapper]
            Orbital mapper for the calculation of DOS.
        overlap: bool
            Whether to use overlap matrix for the calculation of DOS.
        degauss: float
            Standard deviation of the Gaussian pulse for the calculation of DOS.
        de: float
            Energy interval for the calculation of DOS.
        spin_deg: int
            Spin degree of freedom.
        dtype: Union[str, torch.dtype]
            Data type for the calculation of DOS.
        device: Union[str, torch.device]
            Device for the calculation of DOS.
        '''
        super().__init__()
        
        self.loss = nn.MSELoss()
        self.device = device
        self.degauss = degauss
        self.de = de
        self.spin_deg = spin_deg  
        self.idp = None

        if basis is not None:
            # E3TB is not supported yet
            assert idp is not None
            self.idp = idp

        eigvalk = ['idp', 'dtype', 'device', 'out_field', 
                   'h_edge_field', 'h_node_field', 'h_out_field', 
                   's_edge_field', 's_node_field', 's_out_field']
        eigvalv = [self.idp, dtype, device, self.EIGVAL_,
                   self.EDGE_, self.NODE_, self.HAMILT_,
                   None, None, None]
        options = dict(zip(eigvalk, eigvalv))
        if overlap:
            options.update({'s_edge_field': self.EDGE_OVLP_,
                            's_node_field': self.NODE_OVLP_,
                            's_out_field': self.OVLP_})
        self.eigenvalue = Eigenvalues(**options)
        self.overlap = overlap

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        '''
        
        Development notes
        -----------------
        caller: trainer.py:Trainer.iteration() call of self.train_lossfunc
        caller: trainer.py:Trainer.epoch() calls iteration()
        
        '''
        loss = 0
        # type conversion: Dict -> Batch -> List
        tbdata_collection  = Batch.from_dict(data).to_data_list()
        dftdata_collection = Batch.from_dict(ref_data).to_data_list()
        for tbdata, dftdata in zip(tbdata_collection, dftdata_collection):
            tbdata  = self.eigenvalue(AtomicData.to_AtomicDataDict(tbdata))
            dftdata = AtomicData.to_AtomicDataDict(dftdata)
            if dftdata.get(self.EIGVAL_) is None:
                dftdata = self.eigenvalue(dftdata)

            emin,  emax  = dftdata.get(self.ENERGY_WINDOW_, (None, None))
            ibmin, ibmax = dftdata.get(self.BAND_WINDOW_, (0, None))

            eigvaltb  = tbdata[self.EIGVAL_][0]
            assert len(eigvaltb.shape) == 2
            eigvaldft = dftdata[self.EIGVAL_][0]
            assert len(eigvaldft.shape) == 2

            # band range selection
            # sanity check
            nktb,  nbtb  = eigvaltb.shape
            nkdft, nbdft = eigvaldft.shape
            assert nktb == nkdft
            # band range selection: sanity check
            nb = min(nbtb, nbdft)
            ibmax = nb if ibmax is None else ibmax
            assert ibmax <= nb
            ibmin, ibmax = int(ibmin), int(ibmax)
            assert ibmin < ibmax # not allow the nb==0 case?
            # slice
            eigvaltb  = eigvaltb[:,ibmin:ibmax]
            eigvaldft = eigvaldft[:,ibmin:ibmax]
            nk, nbslice = eigvaltb.shape
            # alignment
            eigvaltb  = eigvaltb  - eigvaltb.reshape(-1).min()
            eigvaldft = eigvaldft - eigvaldft.reshape(-1).min()

            # integrate to get the DOS
            emin, emax = 0., max(eigvaltb.max().item(), eigvaldft.max().item())
            dostb  = DOSLoss.calc_dos(eigvaltb,  dftdata[self.WK_], 
                                      emin, emax, de=self.de, sigma=self.degauss)
            dosdft = DOSLoss.calc_dos(eigvaldft, dftdata[self.WK_], 
                                      emin, emax, de=self.de, sigma=self.degauss)
            # the loss is the MSE between two DOS
            loss += self.loss(dostb, dosdft)
        return loss # it seems do not matter if I normalize the loss with number of batches

# @Loss.register("hamil")
# class HamilLoss(nn.Module):
#     def __init__(
#             self, 
#             basis: Dict[str, Union[str, list]]=None,
#             idp: Union[OrbitalMapper, None]=None,
#             overlap: bool=False,
#             dtype: Union[str, torch.dtype] = torch.float32, 
#             device: Union[str, torch.device] = torch.device("cpu"),
#             **kwargs,
#         ):

#         super(HamilLoss, self).__init__()
#         self.loss1 = nn.L1Loss()
#         self.loss2 = nn.MSELoss()
#         self.overlap = overlap
#         self.device = device

#         if basis is not None:
#             self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
#             if idp is not None:
#                 assert idp == self.idp, "The basis of idp and basis should be the same."
#         else:
#             assert idp is not None, "Either basis or idp should be provided."
#             self.idp = idp

#     def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
#         # mask the data

#         # data[AtomicDataDict.NODE_FEATURES_KEY].masked_fill(~self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY]], 0.)
#         # data[AtomicDataDict.EDGE_FEATURES_KEY].masked_fill(~self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY]], 0.)

#         node_mean = ref_data[AtomicDataDict.NODE_FEATURES_KEY].mean(dim=-1, keepdim=True)
#         edge_mean = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].mean(dim=-1, keepdim=True)
#         node_weight = 1/((ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean).norm(dim=-1, keepdim=True)+1e-5)
#         edge_weight = 1/((ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean).norm(dim=-1, keepdim=True)+1e-5)
        
#         pre = (node_weight*(data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
#         tgt = (node_weight*(ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
#         onsite_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))

#         pre = (edge_weight*(data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#         tgt = (edge_weight*(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#         hopping_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
#         if self.overlap:
#             over_mean = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].mean(dim=-1, keepdim=True)
#             over_weight = 1/((ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean).norm(dim=-1, keepdim=True)+1e-5)
#             pre = (over_weight*(data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#             tgt = (over_weight*(ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
#             hopping_loss += self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
#         return hopping_loss + onsite_loss

def shift_mu(data: AtomicDataDict, ref_data: AtomicDataDict,idp:OrbitalMapper):
    mu_n = (data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]) * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
    mu_n = mu_n.sum(dim=-1) # [natoms]
    mu_n_diag = (data[AtomicDataDict.NODE_FEATURES_KEY][:,idp.full_mask_to_diag] - 
                    ref_data[AtomicDataDict.NODE_FEATURES_KEY][:,idp.full_mask_to_diag]) * ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag]
    mu_n_diag = mu_n_diag.sum(dim=-1) # [natoms]
    mu_n_all = mu_n * 2 - mu_n_diag

    mu_e = (data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]) * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
    mu_e = mu_e.sum(dim=-1) # [edges]
    mu_e_diag = (data[AtomicDataDict.EDGE_FEATURES_KEY][:,idp.full_mask_to_diag] - 
                    ref_data[AtomicDataDict.EDGE_FEATURES_KEY][:,idp.full_mask_to_diag])  * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag] 
    mu_e_diag = mu_e_diag.sum(dim=-1) # [edges]
    mu_e_all = mu_e*2 - mu_e_diag

    norm_ss_n =  (ref_data[AtomicDataDict.NODE_OVERLAP_KEY] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]).sum(dim=-1)
    norm_ss_n_diag = (ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY][:,idp.full_mask_to_diag]).sum(dim=-1)
    norm_ss_n_all = norm_ss_n * 2 - norm_ss_n_diag
    
    norm_ss_e =  (ref_data[AtomicDataDict.EDGE_OVERLAP_KEY] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]).sum(dim=-1)
    norm_ss_e_diag = (ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][:,idp.full_mask_to_diag]).sum(dim=-1)
    norm_ss_e_all = norm_ss_e * 2 - norm_ss_e_diag

    return mu_n_all, mu_e_all, norm_ss_n_all, norm_ss_e_all

@Loss.register("eig_ham")
class EigHamLoss(nn.Module):
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            diff_on: bool=False,
            eout_weight: float=0.01,
            diff_weight: float=0.01,
            diff_valence: dict=None,
            spin_deg: int = 2,
            coeff_ham: float=1.,
            coeff_ovp: float=1.,
            **kwargs,
        ):
        super(EigHamLoss, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift
        self.coeff_ham = coeff_ham
        assert self.coeff_ham <= 1.
        self.coeff_ovp = coeff_ovp

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.eigloss = EigLoss(
            idp=self.idp,
            overlap=overlap,
            diff_on=diff_on,
            eout_weight=eout_weight,
            diff_weight=diff_weight,
            diff_valence=diff_valence,
            spin_deg=spin_deg,
            dtype=dtype, 
            device=device,
        )

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data

        if self.onsite_shift:
            batch = data.get("batch", torch.zeros(data[AtomicDataDict.POSITIONS_KEY].shape[0]))
            mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data,idp=self.idp)
            
            if batch.max() == 0: # when batchsize is zero
                diffhs = mu_n.sum() + mu_e.sum()
                ss = norm_ss_n.sum() + norm_ss_e.sum()
                mu = diffhs / ss
                mu = mu.detach()
                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            
            elif batch.max() >= 1:
                slices = data["__slices__"]["pos"]
                slices_e = data["__slices__"]["edge_index"]

                mu_n = torch.stack([mu_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                mu_e = torch.stack([mu_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                norm_ss_n = torch.stack([norm_ss_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                norm_ss_e = torch.stack([norm_ss_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                mu = mu_n + mu_e 
                ss = norm_ss_n + norm_ss_e 
                mu = mu / ss
                mu = mu.detach()

                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                edge_mu_index = torch.zeros(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1], dtype=torch.long, device=self.device)
                for i in range(1, batch.max().item()+1):
                    edge_mu_index[data["__slices__"]["edge_index"][i]:data["__slices__"]["edge_index"][i+1]] += i
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                
        pre = data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        onsite_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

        pre = data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        hopping_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))
        
        if self.overlap:
            pre = data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            tgt = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            overlap_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            pre = data[AtomicDataDict.NODE_OVERLAP_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
            tgt = ref_data[AtomicDataDict.NODE_OVERLAP_KEY][self.idp.mask_to_nrme[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
            overlap_loss += 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            ham_loss = (1/3) * (hopping_loss + onsite_loss + (self.coeff_ovp / self.coeff_ham) * overlap_loss)
        else:
            ham_loss = 0.5 * (onsite_loss + hopping_loss)

        eigloss = self.eigloss(data, ref_data)

        return self.coeff_ham * ham_loss + (1 - self.coeff_ham) * eigloss

        


    

@Loss.register("hamil_abs")
class HamilLossAbs(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossAbs, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data
        if self.onsite_shift:
            batch = data.get("batch", torch.zeros(data[AtomicDataDict.POSITIONS_KEY].shape[0]))
            mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data,idp=self.idp)

            if batch.max() == 0: # when batchsize is zero
                diffhs = mu_n.sum() + mu_e.sum()
                ss = norm_ss_n.sum() + norm_ss_e.sum()
                mu = diffhs / ss
                mu = mu.detach()
                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            elif batch.max() >= 1:
                slices = data["__slices__"]["pos"]
                slices_e = data["__slices__"]["edge_index"]
                mu_n = torch.stack([mu_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                mu_e = torch.stack([mu_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                norm_ss_n = torch.stack([norm_ss_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                norm_ss_e = torch.stack([norm_ss_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])
               
                mu = mu_n + mu_e 
                ss = norm_ss_n + norm_ss_e 
                mu = mu / ss
                mu = mu.detach()

                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                edge_mu_index = torch.zeros(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1], dtype=torch.long, device=self.device)
                for i in range(1, batch.max().item()+1):
                    edge_mu_index[data["__slices__"]["edge_index"][i]:data["__slices__"]["edge_index"][i+1]] += i
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                
        pre = data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        onsite_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

        pre = data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        hopping_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))
        
        if self.overlap:
            pre = data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            tgt = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[ref_data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            overlap_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            pre = data[AtomicDataDict.NODE_OVERLAP_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
            tgt = ref_data[AtomicDataDict.NODE_OVERLAP_KEY][self.idp.mask_to_nrme[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
            overlap_loss += 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (onsite_loss + hopping_loss)

@Loss.register("hamil_blas")
class HamilLossBlas(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossBlas, self).__init__()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data
        if self.onsite_shift:
            batch = data.get("batch", torch.zeros(data[AtomicDataDict.POSITIONS_KEY].shape[0]))
            mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data,idp=self.idp)
            
            if batch.max() == 0: # when batchsize is zero
                diffhs = mu_n.sum() + mu_e.sum()
                ss = norm_ss_n.sum() + norm_ss_e.sum()
                mu = diffhs / ss
                mu = mu.detach()
                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            elif batch.max() >= 1:
                slices = data["__slices__"]["pos"]
                slices_e = data["__slices__"]["edge_index"]

                mu_n = torch.stack([mu_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                mu_e = torch.stack([mu_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                norm_ss_n = torch.stack([norm_ss_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                norm_ss_e = torch.stack([norm_ss_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                mu = mu_n +  mu_e 
                ss = norm_ss_n + norm_ss_e
                mu = mu / ss
                mu = mu.detach()

                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                edge_mu_index = torch.zeros(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1], dtype=torch.long, device=self.device)
                for i in range(1, batch.max().item()+1):
                    edge_mu_index[data["__slices__"]["edge_index"][i]:data["__slices__"]["edge_index"][i+1]] += i
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                
        onsite_loss = data[AtomicDataDict.NODE_FEATURES_KEY]-ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        onsite_index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().unique()
        onsite_loss = scatter_mean(
            src = onsite_loss.abs(), 
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
            )[onsite_index][self.idp.mask_to_nrme[onsite_index]].mean() + scatter_mean(
            src = onsite_loss**2,
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
        )[onsite_index][self.idp.mask_to_nrme[onsite_index]].mean().sqrt()
        onsite_loss *= 0.5

        hopping_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().unique()
        hopping_loss = data[AtomicDataDict.EDGE_FEATURES_KEY]-ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        hopping_loss = scatter_mean(
            src = hopping_loss.abs(), 
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
            )[hopping_index][self.idp.mask_to_erme[hopping_index]].mean() + scatter_mean(
            src = hopping_loss**2,
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
        )[hopping_index][self.idp.mask_to_erme[hopping_index]].mean().sqrt()
        hopping_loss *= 0.5
        
        if self.overlap:
            overlap_loss = data[AtomicDataDict.EDGE_OVERLAP_KEY]-ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            overlap_loss = scatter_mean(
                src = overlap_loss.abs(), 
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
                )[hopping_index][self.idp.mask_to_erme[hopping_index]].mean() + scatter_mean(
                src = overlap_loss**2,
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
            )[hopping_index][self.idp.mask_to_erme[hopping_index]].mean().sqrt()
            overlap_loss *= 0.5

            overlap_onsite_loss = data[AtomicDataDict.NODE_OVERLAP_KEY]-ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
            overlap_onsite_loss = scatter_mean(
                src = overlap_onsite_loss.abs(), 
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
                )[onsite_index][self.idp.mask_to_nrme[onsite_index]].mean() + scatter_mean(
                src = overlap_onsite_loss**2,
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
            )[onsite_index][self.idp.mask_to_nrme[onsite_index]].mean().sqrt()
            overlap_loss += overlap_onsite_loss * 0.5

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (onsite_loss + hopping_loss)


@Loss.register("hamil_wt")
class HamilLossWT(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            onsite_shift: bool=False,
            onsite_weight: Union[float, int, dict]=1.,
            hopping_weight: Union[float, int, dict]=1.,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossWT, self).__init__()
        self.overlap = overlap
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.onsite_weight = torch.ones(idp.num_types)
        self.hopping_weight = torch.ones(len(idp.bond_types))
        if isinstance(onsite_weight, float) or isinstance(onsite_weight, int):
            self.onsite_weight *= onsite_weight
        elif isinstance(onsite_weight, dict):
            for k,v in onsite_weight.items():
                self.onsite_weight[idp.chemical_symbol_to_type[k]] = v
        else:
            raise TypeError("onsite weight should be either float, int or dict")
        
        if isinstance(hopping_weight, float) or isinstance(hopping_weight, int):
            self.hopping_weight *= hopping_weight
        elif isinstance(hopping_weight, dict):
            for k,v in hopping_weight.items():
                self.hopping_weight[idp.bond_to_type[k]] = v
        else:
            raise TypeError("hopping weight should be either float, int or dict")
        
        self.onsite_weight = self.onsite_weight.unsqueeze(1)
        self.hopping_weight = self.hopping_weight.unsqueeze(1)

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data

        if self.onsite_shift:
            batch = data.get("batch", torch.zeros(data[AtomicDataDict.POSITIONS_KEY].shape[0]))
            mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data,idp=self.idp)

            if batch.max() == 0: # when batchsize is zero
                diffhs = mu_n.sum() + mu_e.sum()
                ss = norm_ss_n.sum() + norm_ss_e.sum()
                mu = diffhs / ss
                mu = mu.detach()
                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            elif batch.max() >= 1:
                slices = data["__slices__"]["pos"]
                slices_e = data["__slices__"]["edge_index"]
                
                mu_n = torch.stack([mu_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                mu_e = torch.stack([mu_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                norm_ss_n = torch.stack([norm_ss_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                norm_ss_e = torch.stack([norm_ss_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                mu = mu_n + mu_e
                ss = norm_ss_n + norm_ss_e
                mu = mu / ss
                mu = mu.detach()

                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                edge_mu_index = torch.zeros(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1], dtype=torch.long, device=self.device)
                for i in range(1, batch.max().item()+1):
                    edge_mu_index[data["__slices__"]["edge_index"][i]:data["__slices__"]["edge_index"][i+1]] += i
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                
        onsite_loss = data[AtomicDataDict.NODE_FEATURES_KEY]-ref_data[AtomicDataDict.NODE_FEATURES_KEY]
        onsite_index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten().unique()
        onsite_loss = (self.onsite_weight * scatter_mean(
            src = onsite_loss.abs(), 
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
            )[onsite_index])[self.idp.mask_to_nrme[onsite_index]].mean() + (self.onsite_weight**2 * scatter_mean(
            src = onsite_loss**2,
            index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.type_names)
        )[onsite_index])[self.idp.mask_to_nrme[onsite_index]].mean().sqrt()
        onsite_loss *= 0.5

        hopping_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten().unique()
        hopping_loss = data[AtomicDataDict.EDGE_FEATURES_KEY]-ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
        hopping_loss = (self.hopping_weight * scatter_mean(
            src = hopping_loss.abs(), 
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
            )[hopping_index])[self.idp.mask_to_erme[hopping_index]].mean() + (self.hopping_weight**2 * scatter_mean(
            src = hopping_loss**2,
            index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
            dim=0,
            dim_size=len(self.idp.bond_types)
        )[hopping_index])[self.idp.mask_to_erme[hopping_index]].mean().sqrt()
        hopping_loss *= 0.5
        
        if self.overlap:
            overlap_loss = data[AtomicDataDict.EDGE_OVERLAP_KEY]-ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            overlap_loss = (self.hopping_weight * scatter_mean(
                src = overlap_loss.abs(), 
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
                )[hopping_index])[self.idp.mask_to_erme[hopping_index]].mean() + (self.hopping_weight **2 * scatter_mean(
                src = overlap_loss**2,
                index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.bond_types)
            )[hopping_index])[self.idp.mask_to_erme[hopping_index]].mean().sqrt()
            overlap_loss *= 0.5

            overlap_onsite_loss = data[AtomicDataDict.NODE_OVERLAP_KEY]-ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
            overlap_onsite_loss = (self.onsite_weight * scatter_mean(
                src = overlap_onsite_loss.abs(), 
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
                )[onsite_index])[self.idp.mask_to_nrme[onsite_index]].mean() + ((self.onsite_weight ** 2) * scatter_mean(
                src = overlap_onsite_loss**2,
                index = data[AtomicDataDict.ATOM_TYPE_KEY].flatten(),
                dim=0,
                dim_size=len(self.idp.type_names)
            )[onsite_index])[self.idp.mask_to_nrme[onsite_index]].mean().sqrt()
            overlap_loss += overlap_onsite_loss * 0.5

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (onsite_loss + hopping_loss)
        

class HamilLossAnalysis(object):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32,
            decompose: bool = False,
            onsite_shift: bool=False,
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossAnalysis, self).__init__()
        self.overlap = overlap
        self.device = device
        self.decompose = decompose
        self.dtype = dtype
        self.device = device
        self.onsite_shift = onsite_shift

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        self.idp.get_irreps()

        if decompose:
            self.e3h = E3Hamiltonian(idp=self.idp, decompose=decompose, overlap=False, device=device, dtype=dtype)
            self.e3s = E3Hamiltonian(idp=self.idp, decompose=decompose, overlap=True, device=device, dtype=dtype)
    
    def __call__(self, data: AtomicDataDict, ref_data: AtomicDataDict, running_avg: bool=False):

        batch = data.get("batch", torch.zeros(data[AtomicDataDict.POSITIONS_KEY].shape[0]))
        if self.onsite_shift:
            mu_n, mu_e, norm_ss_n, norm_ss_e = shift_mu(data=data, ref_data=ref_data,idp=self.idp)

            if batch.max() == 0: # when batchsize is zero
                diffhs = mu_n.sum() + mu_e.sum()
                ss = norm_ss_n.sum() + norm_ss_e.sum()
                mu = diffhs / ss
                mu = mu.detach()
                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
            elif batch.max() >= 1:
                slices = data["__slices__"]["pos"]
                slices_e = data["__slices__"]["edge_index"]

                mu_n = torch.stack([mu_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                mu_e = torch.stack([mu_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                norm_ss_n = torch.stack([norm_ss_n[slices[i]:slices[i+1]].sum() for i in range(len(slices)-1)])
                norm_ss_e = torch.stack([norm_ss_e[slices_e[i]:slices_e[i+1]].sum() for i in range(len(slices_e)-1)])

                mu = mu_n + mu_e
                ss = norm_ss_n + norm_ss_e
                mu = mu / ss
                mu = mu.detach()

                ref_data[AtomicDataDict.NODE_FEATURES_KEY] = ref_data[AtomicDataDict.NODE_FEATURES_KEY] + mu[batch, None] * ref_data[AtomicDataDict.NODE_OVERLAP_KEY]
                edge_mu_index = torch.zeros(data[AtomicDataDict.EDGE_INDEX_KEY].shape[1], dtype=torch.long, device=self.device)
                for i in range(1, batch.max().item()+1):
                    edge_mu_index[data["__slices__"]["edge_index"][i]:data["__slices__"]["edge_index"][i+1]] += i
                ref_data[AtomicDataDict.EDGE_FEATURES_KEY] = ref_data[AtomicDataDict.EDGE_FEATURES_KEY] + mu[edge_mu_index, None] * ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
        
        for key in ["__slices__", "__cumsum__", "__cat_dims__", "__num_nodes_list__", "__data_class__"]:
            data.pop(key, None)
            ref_data.pop(key, None)

        if self.decompose:
            data = self.e3h(data)
            ref_data = self.e3h(ref_data)
            if self.overlap:
                data = self.e3s(data)
                ref_data = self.e3s(ref_data)
        
        if not running_avg or not hasattr(self, "stats"):
            self.stats = {}
            self.stats["mae"] = 0.
            self.stats["rmse"] = 0.
            self.stats["n_element"] = 0

            # init the self.stats
            self.stats.setdefault("onsite", {})
            self.stats.setdefault("hopping", {})
            if self.overlap:
                self.stats.setdefault("overlap", {})

            for at, tp in self.idp.chemical_symbol_to_type.items():
                self.stats["onsite"][at] = {
                    "rmse":0.,
                    "mae":0.,
                    "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                    "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "n_element":0,
                }
            
            for bt, tp in self.idp.bond_to_type.items():
                self.stats["hopping"][bt] = {
                    "rmse":0.,
                    "mae":0.,
                    "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                    "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                    "n_element":0,
                }

                if self.overlap:
                    self.stats["overlap"][bt] = {
                        "rmse":0.,
                        "mae":0.,
                        "rmse_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device), 
                        "mae_per_block_element":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "rmse_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "mae_per_irreps":torch.zeros(1, dtype=self.dtype, device=self.device),
                        "n_element":0,
                    }
                
        
        with torch.no_grad():
            n_total = 0
            err = data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            mask = self.idp.mask_to_nrme
            onsite = self.stats.get("onsite")
            for at, tp in self.idp.chemical_symbol_to_type.items():
                onsite_mask = mask[tp]
                onsite_err = err[data["atom_types"].flatten().eq(tp)]
                if onsite_err.shape[0] == 0:
                    continue
                onsite_err = onsite_err[:, onsite_mask] # [N_atom_i, n_element]

                rmserr = (onsite_err**2).mean(dim=0).sqrt()
                maerr = onsite_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[onsite_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[onsite_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                
                n_element_old = onsite[at]["n_element"]
                n_total += n_element_old + onsite_err.numel()
                ratio = n_element_old / (n_element_old + onsite_err.numel())
                onsite[at] = {
                    "rmse": ((onsite[at]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                    "mae":onsite[at]["mae"] * ratio + maerr.mean() * (1-ratio),
                    "rmse_per_block_element": ((onsite[at]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                    "mae_per_block_element": onsite[at]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                    "rmse_per_irreps": ((onsite[at]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                    "mae_per_irreps": onsite[at]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                    "n_element":n_element_old + onsite_err.numel(), 
                    }
                
                self.stats["mae"] += onsite[at]["mae"] * onsite[at]["n_element"]
                self.stats["rmse"] += onsite[at]["rmse"]**2 * onsite[at]["n_element"]

            err = data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            amp = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].abs()
            mask = self.idp.mask_to_erme
            hopping = self.stats.get("hopping", {})
            
            for bt, tp in self.idp.bond_to_type.items():
                hopping_mask = mask[tp]
                hopping_err = err[data["edge_type"].flatten().eq(tp)]
                if hopping_err.shape[0] == 0:
                    continue
                hopping_err = hopping_err[:, hopping_mask]
                
                rmserr = (hopping_err**2).mean(dim=0).sqrt()
                maerr = hopping_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[hopping_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[hopping_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                
                n_element_old = hopping[bt]["n_element"]
                n_total += n_element_old + hopping_err.numel()
                ratio = n_element_old / (n_element_old + hopping_err.numel())

                hopping[bt] = {
                    "rmse": ((hopping[bt]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                    "mae":hopping[bt]["mae"] * ratio + maerr.mean() * (1-ratio),
                    "rmse_per_block_element": ((hopping[bt]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                    "mae_per_block_element": hopping[bt]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                    "rmse_per_irreps": ((hopping[bt]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                    "mae_per_irreps": hopping[bt]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                    "n_element":n_element_old + hopping_err.numel(), 
                    }
                
                self.stats["mae"] += hopping[bt]["mae"] * hopping[bt]["n_element"]
                self.stats["rmse"] += hopping[bt]["rmse"]**2 * hopping[bt]["n_element"]
            
            if self.overlap:
                err = data[AtomicDataDict.EDGE_OVERLAP_KEY] - ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                amp = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].abs()
                mask = self.idp.mask_to_erme
                hopping = self.stats.get("overlap", {})

                for bt, tp in self.idp.bond_to_type.items():
                    hopping_mask = mask[tp]
                    hopping_err = err[data["edge_type"].flatten().eq(tp)]
                    if hopping_err.shape[0] == 0:
                        continue
                    hopping_err = hopping_err[:, hopping_mask]
                    
                    rmserr = (hopping_err**2).mean(dim=0).sqrt()
                    maerr = hopping_err.abs().mean(dim=0)
                    rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    rmse_per_irreps[hopping_mask] = rmserr
                    maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    maerr_per_irreps[hopping_mask] = maerr

                    rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                    maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                    

                    n_element_old = hopping[bt]["n_element"]
                    n_total += n_element_old + hopping_err.numel()
                    ratio = n_element_old / (n_element_old + hopping_err.numel())

                    hopping[bt] = {
                        "rmse": ((hopping[bt]["rmse"]**2) * ratio + (rmserr**2).mean() * (1-ratio)).sqrt(),
                        "mae":hopping[bt]["mae"] * ratio + maerr.mean() * (1-ratio),
                        "rmse_per_block_element": ((hopping[bt]["rmse_per_block_element"]**2) * ratio + rmserr**2 * (1-ratio)).sqrt(),
                        "mae_per_block_element": hopping[bt]["mae_per_block_element"]*ratio + maerr * (1-ratio),
                        "rmse_per_irreps": ((hopping[bt]["rmse_per_irreps"]**2) * ratio + rmse_per_irreps**2 * (1-ratio)).sqrt(),
                        "mae_per_irreps": hopping[bt]["mae_per_irreps"] * ratio + maerr_per_irreps * (1-ratio),
                        "n_element":n_element_old + hopping_err.numel(), 
                        }
                    
                    self.stats["mae"] += hopping[bt]["mae"] * hopping[bt]["n_element"]
                    self.stats["rmse"] += hopping[bt]["rmse"]**2 * hopping[bt]["n_element"]

            # compute overall mae, rmse
                    
            self.stats["mae"] = self.stats["mae"] / (n_total + 1e-6)
            self.stats["rmse"] = self.stats["rmse"] / (n_total + 1e-6)
            self.stats["rmse"] = self.stats["rmse"].sqrt()
            
        return self.stats
    
    def report(self):
        assert hasattr(self, "stats"), "The stats is not computed yet."

        print(f"TOTAL:")
        print(f"MAE: {self.stats['mae']}")
        print(f"RMSE: {self.stats['rmse']}")
        print(f"\n")
        
        with torch.no_grad():
            print(f"Onsite: ")
            for at, tp in self.idp.chemical_symbol_to_type.items():
                print(f"{at}:")
                print(f"MAE: {self.stats['onsite'][at]['mae']}")
                print(f"RMSE: {self.stats['onsite'][at]['rmse']}")

                # compute the onsite per block err
                onsite_mae = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                onsite_rmse = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                mae_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)
                mae_per_block_element[self.idp.mask_to_nrme[tp]] = self.stats["onsite"][at]["mae_per_block_element"]
                rmse_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)              
                rmse_per_block_element[self.idp.mask_to_nrme[tp]] = self.stats["onsite"][at]["rmse_per_block_element"]
                
                ist = 0
                for i,iorb in enumerate(self.idp.full_basis):
                    jst = 0
                    li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
                    for j,jorb in enumerate(self.idp.full_basis):
                        orbpair = iorb + "-" + jorb
                        lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                        
                        # constructing hopping blocks
                        if iorb == jorb:
                            factor = 0.5
                        else:
                            factor = 1.0

                        # constructing onsite blocks
                        if i <= j:
                            onsite_mae[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * mae_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)
                            onsite_rmse[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * rmse_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)

                        jst += 2*lj+1
                    ist += 2*li+1

                onsite_mae += onsite_mae.clone().T
                onsite_rmse += onsite_rmse.clone().T

                imask = self.idp.mask_to_basis[tp]
                onsite_mae = onsite_mae[imask][:,imask]
                onsite_rmse = onsite_rmse[imask][:,imask]

                vmax = onsite_mae.max().item()
                plt.matshow(onsite_mae.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("MAE")
                plt.colorbar()
                plt.show()

                vmax = onsite_rmse.max().item()
                plt.matshow(onsite_rmse.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("RMSE")
                plt.colorbar()
                plt.show()

            # compute the hopping per block err
            print(f"Hopping: ")
            for bt, tp in self.idp.bond_to_type.items():
                print(f"{bt}:")
                print(f"MAE: {self.stats['hopping'][bt]['mae']}")
                print(f"RMSE: {self.stats['hopping'][bt]['rmse']}")
                hopping_mae = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                hopping_rmse = torch.zeros((self.idp.full_basis_norb, self.idp.full_basis_norb,), dtype=self.dtype, device=self.device)
                mae_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)
                mae_per_block_element[self.idp.mask_to_erme[tp]] = self.stats["hopping"][bt]["mae_per_block_element"]
                rmse_per_block_element = torch.zeros((self.idp.reduced_matrix_element,), dtype=self.dtype, device=self.device)              
                rmse_per_block_element[self.idp.mask_to_erme[tp]] = self.stats["hopping"][bt]["rmse_per_block_element"]
                ist = 0
                for i,iorb in enumerate(self.idp.full_basis):
                    jst = 0
                    li = anglrMId[re.findall(r"[a-zA-Z]+", iorb)[0]]
                    for j,jorb in enumerate(self.idp.full_basis):
                        orbpair = iorb + "-" + jorb
                        lj = anglrMId[re.findall(r"[a-zA-Z]+", jorb)[0]]
                        
                        # constructing hopping blocks
                        if iorb == jorb:
                            factor = 0.5
                        else:
                            factor = 1.0

                        # constructing onsite blocks
                        if i <= j:
                            hopping_mae[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * mae_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)
                            hopping_rmse[ist:ist+2*li+1,jst:jst+2*lj+1] = factor * rmse_per_block_element[self.idp.orbpair_maps[orbpair]].reshape(2*li+1, 2*lj+1)

                        jst += 2*lj+1
                    ist += 2*li+1

                hopping_mae += hopping_mae.clone().T
                hopping_rmse += hopping_rmse.clone().T
                
                iat, jat = bt.split("-")
                imask = self.idp.mask_to_basis[self.idp.chemical_symbol_to_type[iat]]
                jmask = self.idp.mask_to_basis[self.idp.chemical_symbol_to_type[jat]]
                hopping_mae = hopping_mae[imask][:,jmask]
                hopping_rmse = hopping_rmse[imask][:,jmask]

                vmax = hopping_mae.max().item()
                plt.matshow(hopping_mae.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("MAE")
                plt.colorbar()
                plt.show()

                vmax = hopping_mae.max().item()
                plt.matshow(hopping_rmse.detach().cpu().numpy(), cmap="Blues", vmin=0, vmax=vmax)
                plt.title("RMSE")
                plt.colorbar()
                plt.show()

        



    def __cal_norm__(self, irreps: Irreps, x: torch.Tensor):
        id = 0
        out = []
        if len(x.shape) == 1:
            x = x.unsqueeze_(0)
        for mul, ir in irreps:
            tensor = x[:,id:id+mul*ir.dim].reshape(-1, mul, ir.dim)
            id = id + mul*ir.dim
            tensor = tensor.norm(dim=-1)
            out.append(tensor)

        return torch.cat(out, dim=-1).squeeze(0)
