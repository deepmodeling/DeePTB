import torch
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
from .sktb import OnsiteFormula
from dptb.nn.dftb.hopping_dftb import HoppingIntp
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.nn.dftb.sk_param import SKParam
import logging

log = logging.getLogger(__name__)


class DFTBSK(torch.nn.Module):
    name = "dftbsk"
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp_sk: Union[OrbitalMapper, None]=None,
            skdata: str=None,
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            num_xgrid: int = -1,
            **kwargs,
            ) -> None:
        
        super(DFTBSK, self).__init__()
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device

        if basis is not None:
            self.idp_sk = OrbitalMapper(basis, method="sktb", device=self.device)
            if idp_sk is not None:
                assert idp_sk.basis == self.idp_sk.basis, "The basis of idp and basis should be the same."
        else:
            assert idp_sk is not None, "Either basis or idp should be provided."
            self.idp_sk = idp_sk

        self.transform = transform
        self.basis = self.idp_sk.basis
        self.idp_sk.get_orbpair_maps()
        self.idp_sk.get_skonsite_maps()
        self.model_options = {
            "dftbsk":{
                "skdata": skdata         
                }
        }


        self.onsite_fn = OnsiteFormula(idp=self.idp_sk, functype='dftb', dtype=dtype, device=device)
        self.hopping_fn = HoppingIntp(num_ingrls=self.idp_sk.reduced_matrix_element, method='linear')

        if num_xgrid == -1:
            skparams = SKParam(basis=self.basis, skdata=skdata)
         
            distance_param = skparams.skdict['Distance']
            hopping_param = skparams.skdict['Hopping']
            onsite_param = skparams.skdict['OnsiteE']
            if overlap:
                overlap_param = skparams.skdict['Overlap']

            assert hopping_param.shape == (len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(distance_param)), "The hopping param shape is not correct."
            

        elif num_xgrid > 0:
            distance_param = torch.zeros([num_xgrid],dtype=self.dtype, device=self.device)
            hopping_param = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, num_xgrid], dtype=self.dtype, device=self.device)
            onsite_param = torch.zeros([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, self.onsite_fn.num_paras], dtype=self.dtype, device=self.device)
            if overlap:
                overlap_param = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, num_xgrid], dtype=self.dtype, device=self.device)
        else:
            raise ValueError("The number of xgrid is not correct.")

        # register buffer, so that it can be saved in the state_dict and not be updated by optimizer.
        self.register_buffer("distance_param", distance_param)
        self.register_buffer("hopping_param", hopping_param)
        self.register_buffer("onsite_param", onsite_param)
        if overlap:
            self.register_buffer("overlap_param", overlap_param)
        else:
            log.warning("The overlap is set to False, by default the dftbsk model the overlap should be true. please make sure you know what you are doing.")

        self.hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, dtype=self.dtype, device=self.device, 
                                        strain=False,soc=False)
        if overlap:
            self.overlap = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, node_field=AtomicDataDict.NODE_OVERLAP_KEY, dtype=self.dtype, device=self.device)
            # 这里是为了解决当轨道中包含多个相同 l 的轨道时，overlap 也具有数值。比如 1s-2s之间的overlap. 一般对于 dftb的参数spd 轨道没有这一项，此时all(self.idp_sk.mask_diag) 为True。
            # 当 not all(self.idp_sk.mask_diag) 时。其实这里变成可训练参数也不合适，毕竟这里是直接对接DFTB参数，是不会进行训练的。不过这里这么暂时放着吧。遇到再说。
            overlaponsite_param = torch.ones([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, 1], dtype=self.dtype, device=self.device)
            if not all(self.idp_sk.mask_diag):
                log.warning('In dftbsk model, there are multi-orbital with the same angular momentum l, hence there will be overlap between the orbitals. but the implementation is not full supported!')
                self.overlaponsite_param = torch.nn.Parameter(overlaponsite_param)
            else:
                self.overlaponsite_param = overlaponsite_param
        self.idp = self.hamiltonian.idp

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
        edge_number = self.idp_sk.untransform_bond(edge_index).T
        rij = data[AtomicDataDict.EDGE_LENGTH_KEY]

        data[AtomicDataDict.EDGE_FEATURES_KEY] = torch.zeros((len(edge_index), self.idp_sk.reduced_matrix_element), dtype=self.dtype, device=self.device)
        
        if hasattr(self, "overlap"):
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = torch.zeros((len(edge_index), self.idp_sk.reduced_matrix_element), dtype=self.dtype, device=self.device)
            data[AtomicDataDict.NODE_OVERLAP_KEY] = self.overlaponsite_param[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
            data[AtomicDataDict.NODE_OVERLAP_KEY][:,self.idp_sk.mask_diag] = 1.

        for ibtype in self.idp_sk.bond_types:
            ibtype_idx = self.idp_sk.bond_to_type[ibtype]
            mask = edge_index == ibtype_idx
            
            data[AtomicDataDict.EDGE_FEATURES_KEY][mask] = self.hopping_fn.get_skhij(rij[mask], xx=self.distance_param, yy=self.hopping_param[ibtype_idx])
            
            if hasattr(self, "overlap"):
                data[AtomicDataDict.EDGE_OVERLAP_KEY][mask] = self.hopping_fn.get_skhij(rij[mask], xx=self.distance_param, yy=self.overlap_param[ibtype_idx])

        atomic_numbers = self.idp_sk.untransform_atom(data[AtomicDataDict.ATOM_TYPE_KEY].flatten())
        
        data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                atomic_numbers=atomic_numbers, nn_onsite_paras=self.onsite_param)
        
        if AtomicDataDict.NODE_SOC_SWITCH_KEY not in data:
            data[AtomicDataDict.NODE_SOC_SWITCH_KEY] =  torch.full((data['pbc'].shape[0], 1), False)
        else:
            data[AtomicDataDict.NODE_SOC_SWITCH_KEY].fill_(False)
        
        # sk param to hamiltonian and overlap
        if self.transform:
            data = self.hamiltonian(data)
            if hasattr(self, "overlap"):
                data = self.overlap(data)

        return data
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint: str, 
        basis: Dict[str, Union[str, list]]=None,
        skdata: str=None,
        overlap: bool=None,
        dtype: Union[str, torch.dtype]=None, 
        device: Union[str, torch.device]=None,
        transform: bool=True,
        **kwargs,
        ):

        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        dftb={
                "skdata": skdata         
        }

        assert checkpoint.split(".")[-1] == "pth", "The checkpoint should be a pth file." 
        f = torch.load(checkpoint, map_location=device, weights_only=False)

        for k,v in common_options.items():
            if v is None:
                common_options[k] = f["config"]["common_options"][k]
                log.info(f"{k} is not provided in the input json, set to the value {common_options[k]} in model ckpt.")
        for k,v in dftb.items():
            if v is None:
                dftb[k] = f["config"]["model_options"]["dftb"][k]
                log.info(f"{k} is not provided in the input json, set to the value {dftb[k]} in model ckpt.")

        num_xgrid = f["model_state_dict"]["distance_param"].shape[0]
        model = cls(**common_options, **dftb, num_xgrid=num_xgrid, transform=transform)
        
        if f["config"]["common_options"]["basis"] == common_options["basis"]:
            model.load_state_dict(f["model_state_dict"])
        else:
            log.warning("The basis in the input json is different from the basis in the model ckpt, the model state is not loaded.")
        
        return model