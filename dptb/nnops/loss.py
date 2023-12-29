import torch.nn as nn
import torch
from torch.nn.functional import mse_loss
from dptb.utils.register import Register
from dptb.nn.energy import Eigenvalues
from dptb.nn.hamiltonian import E3Hamiltonian
from typing import Any, Union, Dict
from dptb.data import AtomicDataDict, AtomicData
from dptb.data.transforms import OrbitalMapper
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
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):
        super(EigLoss, self).__init__()
        self.loss = nn.MSELoss()
        self.device = device

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
            up_nband = min(norbs+band_min,nbanddft)

            if band_max == None:
                band_max = up_nband
            else:
                assert band_max <= up_nband

            band_min = int(band_min)
            band_max = int(band_max)

            assert band_min < band_max
            assert len(eig_pred.shape) == 2 and len(eig_label.shape) == 2

            # 对齐eig_pred和eig_label
            eig_pred_cut = eig_pred[:,:band_max-band_min]
            eig_label_cut = eig_label[:,band_min:band_max]


            num_kp, num_bands = eig_pred_cut.shape

            eig_pred_cut = eig_pred_cut - eig_pred_cut.reshape(-1).min()
            eig_label_cut = eig_label_cut - eig_label_cut.reshape(-1).min()

            
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
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            **kwargs,
        ):

        super(HamilLossAbs, self).__init__()
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
        
        pre = data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.NODE_FEATURES_KEY][self.idp.mask_to_nrme[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]]
        onsite_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

        pre = data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        tgt = ref_data[AtomicDataDict.EDGE_FEATURES_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
        hopping_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))
        
        if self.overlap:
            pre = data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            tgt = ref_data[AtomicDataDict.EDGE_OVERLAP_KEY][self.idp.mask_to_erme[data[AtomicDataDict.EDGE_TYPE_KEY].flatten()]]
            overlap_loss = 0.5*(self.loss1(pre, tgt) + torch.sqrt(self.loss2(pre, tgt)))

            return (1/3) * (hopping_loss + onsite_loss + overlap_loss)
        else:
            return 0.5 * (hopping_loss + onsite_loss) 
        

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
            err = data[AtomicDataDict.NODE_FEATURES_KEY] - ref_data[AtomicDataDict.NODE_FEATURES_KEY]
            mask = self.idp.mask_to_nrme[data["atom_types"].flatten()]
            onsite = out.setdefault("onsite", {})
            for at, tp in self.idp.chemical_symbol_to_type.items():
                onsite_mask = mask[data["atom_types"].flatten().eq(tp)]
                onsite_err = err[data["atom_types"].flatten().eq(tp)]
                onsite_err = torch.stack([vec[ma] for vec, ma in zip(onsite_err, onsite_mask)])
                rmserr = (onsite_err**2).mean(dim=0).sqrt()
                maerr = onsite_err.abs().mean(dim=0)
                onsite[at] = {
                    "rmse":(rmserr**2).mean().sqrt(),
                    "mae":maerr.mean(),
                    "rmse_per_block_element":rmserr, 
                    "mae_per_block_element":maerr
                    }

            err = data[AtomicDataDict.EDGE_FEATURES_KEY] - ref_data[AtomicDataDict.EDGE_FEATURES_KEY]
            mask = self.idp.mask_to_erme[data["edge_type"].flatten()]
            hopping = out.setdefault("hopping", {})
            for bt, tp in self.idp.bond_to_type.items():
                hopping_mask = mask[data["edge_type"].flatten().eq(tp)]
                hopping_err = err[data["edge_type"].flatten().eq(tp)]
                hopping_err = torch.stack([vec[ma] for vec, ma in zip(hopping_err, hopping_mask)])
                rmserr = (hopping_err**2).mean(dim=0).sqrt()
                maerr = hopping_err.abs().mean(dim=0)
                hopping[bt] = {
                    "rmse":(rmserr**2).mean().sqrt(),
                    "mae":maerr.mean(),
                    "rmse_per_block_element":rmserr, 
                    "mae_per_block_element":maerr
                    }
            
            if self.overlap:
                err = data[AtomicDataDict.EDGE_OVERLAP_KEY] - ref_data[AtomicDataDict.EDGE_OVERLAP_KEY]
                mask = self.idp.mask_to_erme[data["edge_type"].flatten()]
                overlap = out.setdefault("overlap", {})
                for bt, tp in self.idp.bond_to_type.items():
                    hopping_mask = mask[data["edge_type"].flatten().eq(tp)]
                    hopping_err = err[data["edge_type"].flatten().eq(tp)]
                    hopping_err = torch.stack([vec[ma] for vec, ma in zip(hopping_err, hopping_mask)])
                    rmserr = (hopping_err**2).mean(dim=0).sqrt()
                    maerr = hopping_err.abs().mean(dim=0)

                    overlap[bt] = {
                        "rmse":(rmserr**2).mean().sqrt(),
                        "mae":maerr.mean(),
                        "rmse_per_block_element":rmserr, 
                        "mae_per_block_element":maerr
                        }

        return out