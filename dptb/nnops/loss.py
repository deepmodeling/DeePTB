import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
from dptb.utils.register import Register
from dptb.nn.energy import Eigenvalues
from dptb.nn.hamiltonian import E3Hamiltonian
from typing import Any, Union, Dict
from dptb.data import AtomicDataDict, AtomicData
from dptb.data.transforms import OrbitalMapper
from e3nn.o3 import Irreps
from torch_scatter import scatter_mean
from dptb.utils.torch_geometric import Batch

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
            eig_pred = data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] # (n_kpt, n_band)
            eig_label = ref_data[AtomicDataDict.ENERGY_EIGENVALUE_KEY] # (n_kpt, n_band_dft/n_band)

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

@Loss.register("hamil")
class HamilLoss(nn.Module):
    def __init__(
            self, 
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            overlap: bool=False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLoss, self).__init__()
        self.loss1 = nn.L1Loss()
        self.loss2 = nn.MSELoss()
        self.overlap = overlap
        self.device = device

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

    def forward(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        # mask the data

        # data[AtomicDataDict.NODE_FEATURES_KEY].masked_fill(~self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY]], 0.)
        # data[AtomicDataDict.EDGE_FEATURES_KEY].masked_fill(~self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY]], 0.)

        node_mean = ref_data[AtomicDataDict.NODE_FEATURES_KEY].mean(dim=-1, keepdim=True)
        edge_mean = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].mean(dim=-1, keepdim=True)
        node_weight = 1/((ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean).norm(dim=-1, keepdim=True)+1e-5)
        edge_weight = 1/((ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean).norm(dim=-1, keepdim=True)+1e-5)
        
        pre = (node_weight*(data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        tgt = (node_weight*(ref_data[AtomicDataDict.NODE_FEATURES_KEY]-node_mean))[self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        onsite_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))

        pre = (edge_weight*(data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        tgt = (edge_weight*(ref_data[AtomicDataDict.EDGE_FEATURES_KEY]-edge_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        hopping_loss = self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
        if self.overlap:
            over_mean = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].mean(dim=-1, keepdim=True)
            over_weight = 1/((ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean).norm(dim=-1, keepdim=True)+1e-5)
            pre = (over_weight*(data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            tgt = (over_weight*(ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]-over_mean))[self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            hopping_loss += self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt))
        
        return hopping_loss + onsite_loss
    

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

        # data[AtomicDataDict.NODE_FEATURES_KEY].masked_fill(~self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY]], 0.)
        # data[AtomicDataDict.EDGE_FEATURES_KEY].masked_fill(~self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY]], 0.)

        if self.onsite_shift:
            assert data["batch"].max() == 0, "The onsite shift is only supported for batchsize=1."
            data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] = \
                data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] - \
                data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]].min()
            
            ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] = \
                ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] - \
                ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]].min()
        
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
        # data[AtomicDataDict.NODE_FEATURES_KEY].masked_fill(~self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY]], 0.)
        # data[AtomicDataDict.EDGE_FEATURES_KEY].masked_fill(~self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY]], 0.)

        if self.onsite_shift:
            data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] = \
                data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] - \
                data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]].min()
            
            ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] = \
                ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]] - \
                ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_ndiag[ref_data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]].min()
        
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
            )[self.idp.mask_to_erme].mean().sqrt()
            overlap_loss *= 0.5

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
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossAnalysis, self).__init__()
        self.overlap = overlap
        self.device = device
        self.decompose = decompose

        if basis is not None:
            self.idp = OrbitalMapper(basis, method="e3tb", device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp

        if decompose:
            self.e3h = E3Hamiltonian(idp=idp, decompose=decompose, overlap=False, device=device, dtype=dtype)
            self.e3s = E3Hamiltonian(idp=idp, decompose=decompose, overlap=True, device=device, dtype=dtype)
    
    def __call__(self, data: AtomicDataDict, ref_data: AtomicDataDict):
        if self.decompose:
            data = self.e3h(data)
            ref_data = self.e3h(ref_data)
            if self.overlap:
                data = self.e3s(data)
                ref_data = self.e3s(ref_data)

        
        with torch.no_grad():
            out = {}
            out["mae"] = 0.
            out["rmse"] = 0.
            n_total = 0
            err = data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            amp = ref_data[AtomicDataDict.NODE_FEATURES_KEY].abs()
            mask = self.idp.mask_to_nrme
            onsite = out.setdefault("onsite", {})
            for at, tp in self.idp.chemical_symbol_to_type.items():
                onsite_mask = mask[tp]
                onsite_err = err[data["atom_types"].flatten().eq(tp)]
                onsite_amp = amp[data["atom_types"].flatten().eq(tp)]
                onsite_err = onsite_err[:, onsite_mask]
                onsite_amp = onsite_amp[:, onsite_mask]

                rmserr = (onsite_err**2).mean(dim=0).sqrt()
                maerr = onsite_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[onsite_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[onsite_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                l1amp = onsite_amp.abs().mean(dim=0)
                l2amp = (onsite_amp**2).mean(dim=0).sqrt()
                n_total += onsite_err.numel()
                onsite[at] = {
                    "rmse":(rmserr**2).mean().sqrt(),
                    "mae":maerr.mean(),
                    "rmse_per_block_element":rmserr, 
                    "mae_per_block_element":maerr,
                    "rmse_per_irreps":rmse_per_irreps,
                    "mae_per_irreps":maerr_per_irreps,
                    "l1amp":l1amp,
                    "l2amp":l2amp,
                    "n_element":onsite_err.numel(), 
                    }
                
                out["mae"] += onsite[at]["mae"] * onsite_err.numel()
                out["rmse"] += onsite[at]["rmse"]**2 * onsite_err.numel()
                


            err = data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            amp = ref_data[AtomicDataDict.EDGE_FEATURES_KEY].abs()
            mask = self.idp.mask_to_erme
            hopping = out.setdefault("hopping", {})
            
            for bt, tp in self.idp.bond_to_type.items():
                hopping_mask = mask[tp]
                hopping_err = err[data["edge_type"].flatten().eq(tp)]
                hopping_amp = amp[data["edge_type"].flatten().eq(tp)]
                hopping_err = hopping_err[:, hopping_mask]
                hopping_amp = hopping_amp[:, hopping_mask]
                
                rmserr = (hopping_err**2).mean(dim=0).sqrt()
                maerr = hopping_err.abs().mean(dim=0)
                rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                rmse_per_irreps[hopping_mask] = rmserr
                maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                maerr_per_irreps[hopping_mask] = maerr

                rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                l1amp = hopping_amp.abs().mean(dim=0)
                l2amp = (hopping_amp**2).mean(dim=0).sqrt()
                n_total += hopping_err.numel()
                hopping[bt] = {
                    "rmse":(rmserr**2).mean().sqrt(),
                    "mae":maerr.mean(),
                    "rmse_per_block_element":rmserr, 
                    "mae_per_block_element":maerr,
                    "rmse_per_irreps":rmse_per_irreps,
                    "mae_per_irreps":maerr_per_irreps,
                    "l1amp":l1amp,
                    "l2amp":l2amp,
                    "n_element":hopping_err.numel(),
                    }
                
                out["mae"] += hopping[bt]["mae"] * hopping_err.numel()
                out["rmse"] += hopping[bt]["rmse"]**2 * hopping_err.numel()
            
            if self.overlap:
                err = data[AtomicDataDict.EDGE_OVERLAP_KEY] - ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                amp = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY].abs()
                mask = self.idp.mask_to_erme
                overlap = out.setdefault("overlap", {})

                for bt, tp in self.idp.bond_to_type.items():
                    hopping_mask = mask[tp]
                    hopping_err = err[data["edge_type"].flatten().eq(tp)]
                    hopping_amp = amp[data["edge_type"].flatten().eq(tp)]
                    hopping_err = hopping_err[:, hopping_mask]
                    hopping_amp = hopping_amp[:, hopping_mask]
                    rmserr = (hopping_err**2).mean(dim=0).sqrt()
                    maerr = hopping_err.abs().mean(dim=0)
                    rmse_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    rmse_per_irreps[hopping_mask] = rmserr
                    maerr_per_irreps = torch.zeros(err.shape[1], dtype=err.dtype, device=err.device)
                    maerr_per_irreps[hopping_mask] = maerr

                    rmse_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, rmse_per_irreps)
                    maerr_per_irreps = self.__cal_norm__(self.idp.orbpair_irreps, maerr_per_irreps)
                    l1amp = hopping_amp.abs().mean(dim=0)
                    l2amp = (hopping_amp**2).mean(dim=0).sqrt()

                    n_total += hopping_err.numel()
                    overlap[bt] = {
                        "rmse":(rmserr**2).mean().sqrt(),
                        "mae":maerr.mean(),
                        "rmse_per_block_element":rmserr, 
                        "mae_per_block_element":maerr,
                        "rmse_per_irreps":rmse_per_irreps,
                        "mae_per_irreps":maerr_per_irreps,
                        "l1amp":l1amp,
                        "l2amp":l2amp,
                        "n_element":hopping_err.numel(),
                        }
                    
                    out["mae"] += overlap[bt]["mae"] * hopping_err.numel()
                    out["rmse"] += overlap[bt]["rmse"]**2 * hopping_err.numel()

            # compute overall mae, rmse
                    
            out["mae"] = out["mae"] / n_total
            out["rmse"] = out["rmse"] / n_total
            out["rmse"] = out["rmse"].sqrt()
            
        return out

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