import torch
from typing import Union, Dict, Literal
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
from .sktb import OnsiteFormula
from dptb.nn.dftb.hopping_dftb import HoppingIntp, HoppingIntpSmooth
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.nn.dftb.sk_param import SKParam
from dptb.utils.constants import DFTBPLUS_DIST_FUDGE, DFTBPLUS_N_INTER
import logging

log = logging.getLogger(__name__)

# Valid interpolation methods
VALID_INTERP_METHODS = ['linear', 'cspline', 'smooth_intp']
InterpMethod = Literal['linear', 'cspline', 'smooth_intp']


class DFTBSK(torch.nn.Module):
    """
    DFTB Slater-Koster model for tight-binding calculations.

    Parameters
    ----------
    basis : dict, optional
        Orbital basis specification for each element type.
    idp_sk : OrbitalMapper, optional
        Pre-configured OrbitalMapper instance.
    skdata : str, optional
        Path to Slater-Koster data files (.skf).
    overlap : bool, default False
        Whether to include overlap matrix calculation.
    dtype : torch.dtype, default torch.float32
        Data type for tensors.
    device : torch.device, default 'cpu'
        Device for computation.
    transform : bool, default True
        Whether to transform SK parameters to Hamiltonian.
    num_xgrid : int, default -1
        Number of grid points. -1 means read from SK files.
    interp_method : str, default 'linear'
        Interpolation method: 'linear', 'cspline', or 'smooth_intp'.
        - 'linear': Simple 2-point linear interpolation (fast, C⁰)
        - 'cspline': Cubic spline interpolation (smooth, C²)
        - 'smooth_intp': DFTB+ compatible 8-point polynomial with smooth cutoff (C²)
    smooth_ski : bool, default False
        Shorthand for interp_method='smooth_intp'. When True, uses smooth
        SK integral interpolation which produces identical results to DFTB+.
    dist_fudge : float, optional
        Extrapolation zone size in Bohr for smooth_intp method. Default is 1.0.
    n_interp_points : int, optional
        Number of interpolation points for smooth_intp method. Default is 8.
    """
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
            interp_method: InterpMethod = 'linear',
            smooth_ski: bool = False,
            dist_fudge: float = None,
            n_interp_points: int = None,
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
        # Determine interpolation method
        if smooth_ski:
            if interp_method != 'smooth_intp':
                log.debug("smooth_ski is True, override interp_method to 'smooth_intp'.")
            interp_method = 'smooth_intp'
        if interp_method not in VALID_INTERP_METHODS:
            raise ValueError(
                f"Invalid interp_method '{interp_method}'. "
                f"Must be one of {VALID_INTERP_METHODS}"
            )
        self.interp_method = interp_method

        self.model_options = {
            "dftbsk":{
                "skdata": skdata,
                "interp_method": interp_method,
                }
        }

        self.onsite_fn = OnsiteFormula(idp=self.idp_sk, functype='dftb', dtype=dtype, device=device)

        # Create hopping interpolation function
        if interp_method == 'smooth_intp':
            # _dist_fudge and _n_points use provided values or default constants
            _dist_fudge = dist_fudge if dist_fudge is not None else DFTBPLUS_DIST_FUDGE
            _n_points = n_interp_points if n_interp_points is not None else DFTBPLUS_N_INTER
            # Use DFTB+ compatible interpolation
            self.hopping_fn = HoppingIntpSmooth(
                num_ingrls=self.idp_sk.reduced_matrix_element,
                dist_fudge=_dist_fudge,
                n_points=_n_points,
            )
            log.debug(f"Using DFTB+ compatible interpolation: {_n_points}-point polynomial, "
                     f"dist_fudge={_dist_fudge} Bohr")
        else:
            self.hopping_fn = HoppingIntp(
                num_ingrls=self.idp_sk.reduced_matrix_element,
                method=interp_method
            )

        if num_xgrid == -1:
            skparams = SKParam(basis=self.basis, skdata=skdata, dtype=self.dtype, device=self.device)
         
            distance_param = skparams.skdict['Distance']
            hopping_param = skparams.skdict['Hopping']
            onsite_param = skparams.skdict['OnsiteE']
            mass_param = skparams.skdict['Mass']
            if overlap:
                overlap_param = skparams.skdict['Overlap']

            assert hopping_param.shape == (len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, len(distance_param)), "The hopping param shape is not correct."
            

        elif num_xgrid > 0:
            distance_param = torch.zeros([num_xgrid],dtype=self.dtype, device=self.device)
            hopping_param = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, num_xgrid], dtype=self.dtype, device=self.device)
            onsite_param = torch.zeros([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, self.onsite_fn.num_paras], dtype=self.dtype, device=self.device)
            mass_param = torch.zeros([len(self.idp_sk.type_names), 1], dtype=self.dtype, device=self.device)
            if overlap:
                overlap_param = torch.zeros([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, num_xgrid], dtype=self.dtype, device=self.device)
        else:
            raise ValueError("The number of xgrid is not correct.")

        # register buffer, so that it can be saved in the state_dict and not be updated by optimizer.
        self.register_buffer("distance_param", distance_param)
        self.register_buffer("hopping_param", hopping_param)
        self.register_buffer("onsite_param", onsite_param)
        self.register_buffer("mass",mass_param)
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
            # not support soc, but we need to init the soc switch to False. there for we use the shape of pbc to init the soc switch.
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
        interp_method: InterpMethod=None,
        smooth_ski: bool=False,
        dist_fudge: float=None,
        n_interp_points: int=None,
        **kwargs,
        ):
        """
        Load a DFTBSK model from a checkpoint file.

        Parameters
        ----------
        checkpoint : str
            Path to the .pth checkpoint file.
        basis : dict, optional
            Orbital basis specification. If None, loaded from checkpoint.
        skdata : str, optional
            Path to SK data files. If None, loaded from checkpoint.
        overlap : bool, optional
            Whether to include overlap. If None, loaded from checkpoint.
        dtype : torch.dtype, optional
            Data type. If None, loaded from checkpoint.
        device : torch.device, optional
            Device for computation. If None, loaded from checkpoint.
        transform : bool, default True
            Whether to transform SK parameters to Hamiltonian.
        interp_method : str, optional
            Interpolation method. If None, loaded from checkpoint or defaults to 'linear'.
        smooth_ski : bool, default False
            Use smooth SK integral interpolation.
        dist_fudge : float, optional
            Extrapolation zone size for smooth_intp method.
        n_interp_points : int, optional
            Number of interpolation points for smooth_intp method.

        Returns
        -------
        DFTBSK
            Loaded model instance.
        """
        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        dftbsk_options = {
            "skdata": skdata,
            "interp_method": interp_method,
        }

        assert checkpoint.split(".")[-1] == "pth", "The checkpoint should be a pth file."
        f = torch.load(checkpoint, map_location=device, weights_only=False)

        for k, v in common_options.items():
            if v is None:
                common_options[k] = f["config"]["common_options"][k]
                log.info(f"{k} is not provided in the input json, set to the value {common_options[k]} in model ckpt.")

        # Handle dftbsk model options (support both 'dftb' and 'dftbsk' keys for backward compatibility)
        config_model_opts = f["config"].get("model_options", {})
        saved_dftbsk_opts = config_model_opts.get("dftbsk", config_model_opts.get("dftb", {}))

        for k, v in dftbsk_options.items():
            if v is None:
                if k in saved_dftbsk_opts:
                    dftbsk_options[k] = saved_dftbsk_opts[k]
                    log.info(f"{k} is not provided in the input json, set to the value {dftbsk_options[k]} in model ckpt.")
                elif k == "interp_method":
                    # Default to 'linear' for backward compatibility with old checkpoints
                    dftbsk_options[k] = 'linear'
                    log.info(f"{k} is not in model ckpt, defaulting to 'linear'.")

        # Handle smooth_ski flag
        if smooth_ski:
            log.debug("smooth_ski is True, override interp_method to 'smooth_intp'.")
            dftbsk_options["interp_method"] = 'smooth_intp'

        num_xgrid = f["model_state_dict"]["distance_param"].shape[0]
        model = cls(
            **common_options,
            **dftbsk_options,
            num_xgrid=num_xgrid,
            transform=transform,
            smooth_ski=False,  # Already handled above
            dist_fudge=dist_fudge,
            n_interp_points=n_interp_points,
        )

        if f["config"]["common_options"]["basis"] == common_options["basis"]:
            state_dict = f["model_state_dict"]
            if "mass" not in state_dict:
                state_dict["mass"] = model.mass
            model.load_state_dict(state_dict)
        else:
            log.warning("The basis in the input json is different from the basis in the model ckpt, the model state is not loaded.")

        return model