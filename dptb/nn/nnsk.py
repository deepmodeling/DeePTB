"""The file doing the process from the fitting net output sk formula parameters in node/edge feature to the tight binding two centre integrals parameters in node/edge feature.
in: Data
out Data

basically a map from a matrix parameters to edge/node features, or strain mode's environment edge features
"""

import torch
from dptb.utils.constants import h_all_types, anglrMId
from typing import Tuple, Union, Dict
from dptb.data.transforms import OrbitalMapper
from dptb.data import AtomicDataDict
import numpy as np
import torch.nn as nn
from .sktb import OnsiteFormula, bond_length_list, HoppingFormula
from dptb.nn.sktb.cov_radiiDB import Covalent_radii
from dptb.nn.sktb.bondlengthDB import atomic_radius_v1
from dptb.utils.constants import atomic_num_dict_r, atomic_num_dict
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.utils.tools import j_loader
from dptb.utils.constants import ALLOWED_VERSIONS
from dptb.nn.sktb.soc import SOCFormula
from dptb.data.AtomicData import get_r_map, get_r_map_bondwise
from dptb.nn.sktb.onsiteDB import  onsite_energy_database
import logging

log = logging.getLogger(__name__)

class NNSK(torch.nn.Module):
    name = "nnsk"
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp_sk: Union[OrbitalMapper, None]=None,
            onsite: Dict={"method": "none"},
            hopping: Dict={"method": "powerlaw", "rs":6.0, "w": 0.2},
            overlap: bool = False,
            soc:Dict = {},
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            freeze: Union[bool,str,list] = False,
            push: Union[bool,dict]=False,
            std: float = 0.01,
            atomic_radius: Union[str, Dict] = "v1",
            **kwargs,
            ) -> None:
        
        super(NNSK, self).__init__()

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
        self.onsite_options = onsite
        self.hopping_options = hopping
        self.soc_options = soc
        self.push = push
        self.atomic_radius = atomic_radius
        self.model_options = {
            "nnsk":{
                "onsite": onsite, 
                "hopping": hopping,
                "soc": soc,
                "freeze": freeze,
                "push": push,
                "std": std,
                "atomic_radius":atomic_radius                
                }
            }
        
        if atomic_radius == "v1":
            atomic_radius_dict = atomic_radius_v1
        elif atomic_radius == "cov":
            atomic_radius_dict = Covalent_radii
        else:
            raise ValueError(f"The atomic radius {atomic_radius} is not recognized.")

        
        atomic_numbers = [atomic_num_dict[key] for key in self.basis.keys()]
        self.atomic_radius_list =  torch.zeros(int(max(atomic_numbers))) - 100
        for at in self.basis.keys():
            assert  atomic_radius_dict[at] is not None, f"The atomic radius for {at} is not provided."
            self.atomic_radius_list[atomic_num_dict[at]-1] = atomic_radius_dict[at]

        if self.soc_options.get("method", None) is not None:
            self.idp_sk.get_sksoc_maps()
        
        self.count_push = 0

        # init_onsite, hopping, overlap formula

        self.onsite_fn = OnsiteFormula(idp=self.idp_sk, functype=self.onsite_options["method"], dtype=dtype, device=device)
        self.hopping_fn = HoppingFormula(functype=self.hopping_options["method"])
        if overlap:
            self.overlap_fn = HoppingFormula(functype=self.hopping_options["method"], overlap=True)
        if self.soc_options.get("method", None) is not None:
            self.soc_fn = SOCFormula(idp=self.idp_sk, functype=self.soc_options["method"], dtype=dtype, device=device)
        # init_param
        # 
        hopping_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
        nn.init.normal_(hopping_param, mean=0.0, std=std)
        self.hopping_param = torch.nn.Parameter(hopping_param)
        if overlap:
            overlap_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(overlap_param, mean=0.0, std=std)
            self.overlap_param = torch.nn.Parameter(overlap_param)

            overlaponsite_param = torch.ones([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, 1], dtype=self.dtype, device=self.device)
            if not all(self.idp_sk.mask_diag):
                self.overlaponsite_param = torch.nn.Parameter(overlaponsite_param)
            else:
                self.overlaponsite_param = overlaponsite_param # just use normal tensor incase the checkpoint of old version does not contrains overlaponsite param

        if self.soc_options.get("method", None) is not None:
            if self.soc_options.get("method", None) == 'none':
                self.soc_param = None
            elif self.soc_options.get("method", None) in ['uniform', 'uniform_noref']:
                soc_param = torch.empty([len(self.idp_sk.type_names), self.idp_sk.n_onsite_socLs, self.soc_fn.num_paras], dtype=self.dtype, device=self.device)
                nn.init.normal_(soc_param, mean=0.0, std=std)
                self.soc_param = torch.nn.Parameter(soc_param)
            else:
                raise NotImplementedError(f"The soc method {self.soc_options['method']} is not implemented.")

        if self.onsite_options["method"] == "strain":
            self.onsite_param = None
        elif self.onsite_options["method"] == "none":
            self.onsite_param = None
        elif self.onsite_options["method"] in ["NRL", "uniform", "uniform_noref"]:
            onsite_param = torch.empty([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, self.onsite_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(onsite_param, mean=0.0, std=std)
            self.onsite_param = torch.nn.Parameter(onsite_param)
        else:
            raise NotImplementedError(f"The onsite method {self.onsite_options['method']} is not implemented.")
        
        if self.onsite_options["method"] == "strain":
            # AB [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # AA [...]
            # but need to map to all pairs and all orbital pairs like AB, AA, BB, BA for [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # with this map: BA[sp, sd] = AB[ps, ds]
            strain_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(strain_param, mean=0.0, std=std)
            self.strain_param = torch.nn.Parameter(strain_param)
            # symmetrize the env for same atomic spices
            
        self.hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, dtype=self.dtype, device=self.device, 
                                        strain=hasattr(self, "strain_param"),soc=hasattr(self, "soc_param"))
        if overlap:
            self.overlap = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, node_field=AtomicDataDict.NODE_OVERLAP_KEY, dtype=self.dtype, device=self.device)
            self.register_buffer("ovp_factor", torch.tensor(1.0, dtype=self.dtype, device=self.device))
        self.idp = self.hamiltonian.idp
        
        if freeze:  
            self.freezefunc(freeze)

        self.check_push(push)

        if isinstance (self.hopping_options['rs'], dict):
            first_key = next(iter(self.hopping_options['rs'].keys()))
            key_parts = first_key.split("-")
            if len(key_parts) == 1: # atom-wise rs eg. {'A': 3.0,...}
                self.r_map = get_r_map(self.hopping_options['rs'])
                self.r_map_type = 1 # 1 for atom-wise
            elif len(key_parts) == 2: # bond-wise rs eg. {'A-B': 3.0,...}
                self.r_map = get_r_map_bondwise(self.hopping_options['rs'])
                self.r_map_type = 2 # 2 for bond-wise
            else:
                raise ValueError("The rs tag is not recognized. Please check the rs tag.")
            self.r_map = self.r_map.to(self.device)
            
    def freezefunc(self, freeze: Union[bool,str,list]):
        if freeze is False:
            return 0
        if isinstance(freeze, str):
            if freeze not in ['onsite', 'hopping', 'overlap', 'soc']:
                raise ValueError("The freeze tag is not recognized. Please check the freeze tag.")
            freeze = [freeze]
        elif isinstance(freeze, list):
            for freeze_str in freeze:
                if freeze_str not in ['onsite', 'hopping', 'overlap','soc']:
                    raise ValueError("The freeze tag is not recognized. Please check the freeze tag.")  

        frozen_params = []        
        if freeze is True:
            for name, param in self.named_parameters():
                param.requires_grad = False
                frozen_params.append(name)
            freeze_list = []
        else:
            assert isinstance(freeze,list)
            freeze_list = freeze.copy()
            for name, param in self.named_parameters():
                for freeze_str in freeze_list:
                    if freeze_str in name:
                        param.requires_grad = False
                        frozen_params.append(name)
                        freeze_list.remove(freeze_str)
                        break
                    elif freeze_str=='onsite' and 'strain' in name:
                        # strain and onsite will not be in the model at the same time.
                        param.requires_grad = False
                        frozen_params.append(name)
                        freeze_list.remove('onsite')
                        break
            
        
        if len(freeze_list) > 0:
            raise ValueError(f"The freeze tag {freeze_list} is not recognized or not contained in model. Please check the freeze tag.")
        if not frozen_params:
            raise ValueError("freeze is not set to None, but No parameters are frozen. Please check the freeze tag.")
        elif isinstance(freeze, list):
            if len(frozen_params) != len(freeze):
                raise ValueError("freeze is set to a list, but the number of frozen parameters is not equal to the length of the list. Please check the freeze tag.")
        else:
            assert freeze is True
            if len(frozen_params)!=len(dict(self.named_parameters()).keys()):
                raise ValueError("freeze is True, all parameters should frozen. But the frozen_params != all model.named_parameters. Please check the freeze tag.")
        log.info(f'The {frozen_params} are frozen!')
    
    # add check for push:
    def check_push(self, push: Dict):
        self.if_push = False
        if push is not None and push is not False:
            if abs(push.get("rs_thr")) + abs(push.get("rc_thr")) + abs(push.get("w_thr")) + abs(push.get("ovp_thr",0)) > 0:
                self.if_push = True

        if self.if_push:
            if abs(push.get("rs_thr")) >0:
                if isinstance(self.hopping_options["rs"], dict):
                    log.error(f"rs is a dict, so cannot be decayed. Please provide a float or int for rs.")

            if abs(push.get("rc_thr")) >0:
                if isinstance(self.hopping_options["rc"], dict):
                    log.error(f"rc is a dict, so cannot be decayed. Please provide a float or int for rc.")
                    raise ValueError("rc is a dict, so cannot be decayed. Please provide a float or int for rc.")

            if abs(push.get("ovp_thr",0)) > 0:
                if push.get("ovp_thr",0) > 0:
                    log.error(f"ovp_thr is positive, which means the ovp_factor will be increased. This is not allowed in the push mode.")
                    raise ValueError("ovp_thr is positive, which means the ovp_factor will be increased. This is not allowed in the push mode.")


    def push_decay(self, rs_thr: float=0., rc_thr: float=0., w_thr: float=0., ovp_thr: float=0., period:int=100):
        """Push the soft cutoff function

        Parameters
        ----------
        rs_thr : float
            the threshold step to push the rs
        w_thr : float
            the threshold step to push the w
        """

        self.count_push += 1
        if self.count_push % period == 0:
            if abs(rs_thr) > 0:
                self.hopping_options["rs"] += rs_thr
            if abs(w_thr) > 0:
                self.hopping_options["w"] += w_thr
            if abs(rc_thr) > 0:
                self.hopping_options["rc"] += rc_thr
            if abs(ovp_thr) > 0 :
                if self.ovp_factor >= abs(ovp_thr):
                    self.ovp_factor += ovp_thr
                    log.info(f"ovp_factor is decreased to {self.ovp_factor}")
                else:
                    log.info(f"ovp_factor is already less than {abs(ovp_thr)}, so not decreased.")

            self.model_options["nnsk"]["hopping"] = self.hopping_options

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get the env and bond from the data
        # calculate the sk integrals
        # calculate the onsite
        # calculate the hopping
        # calculate the overlap
        # return the data with updated edge/node features
        # map the parameters to the edge/node/env features
        # compute integrals from parameters using hopping and onsite clas

        if self.if_push:
            self.push_decay(**self.push)

        reflective_bonds = np.array([self.idp_sk.bond_to_type["-".join(self.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(self.idp_sk.bond_types))])
        params = self.hopping_param.data
        reflect_params = params[reflective_bonds]
        for k in self.idp_sk.orbpair_maps.keys():
            iorb, jorb = k.split("-")
            if iorb == jorb:
                # This is to keep the symmetry of the hopping parameters for the same orbital pairs
                # As-Bs = Bs-As; we need to do this because for different orbital pairs, we only have one set of parameters, 
                # eg. we only have As-Bp and Bs-Ap, but not Ap-Bs and Bp-As; and we will use Ap-Bs = Bs-Ap and Bp-As = As-Bp to calculate the hopping integral
                self.hopping_param.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])
        if hasattr(self, "overlap"):
            params = self.overlap_param.data
            reflect_params = params[reflective_bonds]
            for k in self.idp_sk.orbpair_maps.keys():
                iorb, jorb = k.split("-")
                if iorb == jorb:
                    self.overlap_param.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])

        
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)
        if data.get(AtomicDataDict.EDGE_TYPE_KEY, None) is None:
            self.idp_sk(data)

        # edge_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.EDGE_INDEX_KEY]].reshape(2, -1)
        # edge_index = self.idp_sk.transform_reduced_bond(*edge_number)
        edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
        edge_number = self.idp_sk.untransform_bond(edge_index).T
        # edge_index = self.idp_sk.transform_bond(*edge_number)

        # the edge number is the atomic number of the two atoms in the bond.
        # The bond length list is actually the nucli radius (unit of angstrom) at the atomic number.
        # now this bond length list is only available for the first 83 elements.
        # assert (edge_number <= 83).all(), "The bond length list is only available for the first 83 elements."
        # r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[edge_number-1].sum(0)
        # r0 = self.atomic_radius_list[edge_number-1].sum(0)  # bond length r0 = r1 + r2. (r1, r2 are atomic radii of the two atoms)
        r0 = self.atomic_radius_list.type(self.dtype).to(self.device)[edge_number-1].sum(0)
        assert (r0 > 0).all(), "The bond length list is only available for atomic numbers < 84 and excluding the lanthanides."
        
        hopping_options = self.hopping_options.copy() 
        if isinstance (self.hopping_options['rs'], dict):
            if self.r_map_type == 1:
                rs_edgewise = 0.5*self.r_map[edge_number-1].sum(0)
            elif self.r_map_type == 2:
                rs_edgewise = self.r_map[edge_number[0]-1, edge_number[1]-1]
            else:
                raise ValueError(f"r_map_type {self.r_map_type} is not recognized.")
                
            hopping_options['rs'] = rs_edgewise


        data[AtomicDataDict.EDGE_FEATURES_KEY] = self.hopping_fn.get_skhij(
            rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
            paraArray=self.hopping_param[edge_index], # [N_edge, n_pairs, n_paras],
            **hopping_options,
            r0=r0
            ) # [N_edge, n_pairs]

        if hasattr(self, "overlap"):
            equal_orbpair = torch.zeros(self.idp_sk.reduced_matrix_element, dtype=self.dtype, device=self.device)
            for orbpair_key, slices in self.idp_sk.orbpair_maps.items():
                if orbpair_key.split("-")[0] == orbpair_key.split("-")[1]:
                    equal_orbpair[slices] = 1.0
            # this paraconst is to make sure the overlap between the same orbital pairs of the save atom is 1.0 
            # this is taken from the formula of NRL-TB. 
            # the overlap tag now is only designed to be used in the NRL-TB case. In the future, we may need to change this.
            paraconst = edge_number[0].eq(edge_number[1]).float().view(-1, 1) * equal_orbpair.unsqueeze(0)

            data[AtomicDataDict.EDGE_OVERLAP_KEY] = self.ovp_factor * self.overlap_fn.get_sksij(
                rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
                paraArray=self.overlap_param[edge_index],
                paraconst=paraconst,
                **hopping_options,
                r0=r0,
                )
            
            data[AtomicDataDict.NODE_OVERLAP_KEY] = self.overlaponsite_param[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
            data[AtomicDataDict.NODE_OVERLAP_KEY][:,self.idp_sk.mask_diag] = 1.

        atomic_numbers = self.idp_sk.untransform_atom(data[AtomicDataDict.ATOM_TYPE_KEY].flatten())
        if self.onsite_fn.functype == "NRL":
            data = AtomicDataDict.with_onsitenv_vectors(data, with_lengths=True)
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                # atomic_numbers=data[AtomicDataDict.ATOMIC_NUMBERS_KEY],
                atomic_numbers=atomic_numbers,
                onsitenv_index=data[AtomicDataDict.ONSITENV_INDEX_KEY], 
                onsitenv_length=data[AtomicDataDict.ONSITENV_LENGTH_KEY], 
                nn_onsite_paras=self.onsite_param, 
                **self.onsite_options,
                )
        else:
            data[AtomicDataDict.NODE_FEATURES_KEY] = self.onsite_fn.get_skEs(
                atomic_numbers=atomic_numbers, 
                nn_onsite_paras=self.onsite_param
                )
            
        # if hasattr(self, "overlap"):
        #     data[AtomicDataDict.NODE_OVERLAP_KEY] = torch.ones_like(data[AtomicDataDict.NODE_OVERLAP_KEY])
        
        # compute strain
        if self.onsite_fn.functype == "strain":
            data = AtomicDataDict.with_onsitenv_vectors(data, with_lengths=True)
            onsitenv_number = self.idp_sk.untransform_atom(data[AtomicDataDict.ATOM_TYPE_KEY].flatten())[data[AtomicDataDict.ONSITENV_INDEX_KEY]].reshape(2, -1)
            onsitenv_index = self.idp_sk.transform_bond(*onsitenv_number)
            # reflect_index = self.idp_sk.transform_bond(*onsitenv_number.flip(0))
            # onsitenv_index[onsitenv_index<0] = reflect_index[onsitenv_index<0] + len(self.idp_sk.reduced_bond_types)
            # reflect_params = torch.zeros_like(self.strain_param)
            # for k, k_r in zip(self.idp_sk.pair_maps.keys(), reflect_keys):
            #     reflect_params[:,self.idp_sk.pair_maps[k],:] += self.strain_param[:,self.idp_sk.pair_maps[k_r],:]
            # onsitenv_params = torch.cat([self.strain_param, 
            #     reflect_params], dim=0)
            
            # r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[onsitenv_number-1].sum(0)
            r0 = self.atomic_radius_list.type(self.dtype).to(self.device)[onsitenv_number-1].sum(0)  # bond length r0 = r1 + r2. (r1, r2 are atomic radii of the two atoms)
            assert (r0 > 0).all(), "The bond length list is only available for atomic numbers < 84 and excluding the lanthanides."
            onsitenv_params = self.hopping_fn.get_skhij(
            rij=data[AtomicDataDict.ONSITENV_LENGTH_KEY],
            paraArray=self.strain_param[onsitenv_index], # [N_edge, n_pairs, n_paras],
            r0=r0,
            **self.onsite_options,
            ) # [N_edge, n_pairs]
            
            data[AtomicDataDict.ONSITENV_FEATURES_KEY] = onsitenv_params

        if self.soc_options.get("method", None) is not None:
            data[AtomicDataDict.NODE_SOC_KEY] = self.soc_fn.get_socLs(
                atomic_numbers=atomic_numbers, 
                nn_soc_paras=self.soc_param
                )
            if AtomicDataDict.NODE_SOC_SWITCH_KEY not in data:
                data[AtomicDataDict.NODE_SOC_SWITCH_KEY] =  torch.full((data['pbc'].shape[0], 1), True) 
            else:
                data[AtomicDataDict.NODE_SOC_SWITCH_KEY].fill_(True)
        else:
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
        onsite: Dict=None,
        hopping: Dict=None,
        soc:Dict=None,
        overlap: bool=None,
        dtype: Union[str, torch.dtype]=None, 
        device: Union[str, torch.device]=None,
        push: Dict=None,
        freeze: Union[bool,str,list] = False,
        std: float = 0.01,
        transform: bool = True,
        atomic_radius: Union[str, Dict] = None,
        **kwargs,
        ):
        # the mapping from the parameters of the ref_model and the current model can be found using
        # reference model's idp and current idp

        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        nnsk = {
            "onsite": onsite,
            "hopping": hopping,
            "soc": soc,
            "freeze": freeze,
            "push": push,
            "std": std,
            "atomic_radius": atomic_radius
        }


        if checkpoint.split(".")[-1] == "json":
            json_model = j_loader(checkpoint)

            assert 'version' in json_model, "The version of the model is not provided in the json model file."
            ckpt_version = json_model.get("version")
            if ckpt_version not in ALLOWED_VERSIONS:
                raise ValueError("The version of the model is not supported. only 1 and 2 are supported.")
            
            if ckpt_version == 2:
                assert json_model.get("model_params", None) is not None, "The model_params is not provided in the json model file."
                assert json_model.get("unit", None) is not None, "The unit is not provided in the json model file."
                assert json_model.get("model_options", None) is not None, "The model_options is not provided in the json model file."
                assert json_model.get("common_options", None) is not None, "The common_options is not provided in the json model file."
                
                if json_model.get("unit") != 'eV':
                    raise ValueError("The unit of the model is not supported. only eV is supported.")

            for k,v in common_options.items():
                if v is  None:
                    if json_model.get("common_options",{}).get(k, None) is  None:
                        raise ValueError(f"{k} is not provided in both the json model file and the input json.")     
                    else:
                        common_options[k] = json_model["common_options"][k]
                        log.info(f"{k} is not provided in the input json, set to the value {common_options[k]} in the json model file.")
            
            for k,v in nnsk.items():
                if k != 'push' and v is None:
                    if json_model.get("model_options",{}).get("nnsk",{}).get(k, None) is  None:
                        if k=='atomic_radius':
                            nnsk[k] = 'v1'
                        else:
                            raise ValueError(f"{k} is not provided in both the json model file and the input json.")
                    else:
                        nnsk[k] = json_model["model_options"]["nnsk"][k]
                        log.info(f"{k} is not provided in the input json, set to the value {nnsk[k]}in the json model file.")
            
            if ckpt_version == 1:
                if json_model.get("unit", None) is None:
                    ene_unit = "Hartree"
                    log.info('The unit is not provided in the json model file, since this is v1 version model, the default unit is Hartree.')
                else:
                    ene_unit = json_model["unit"]
            elif ckpt_version == 2:
                ene_unit = json_model["unit"]
            else:
                raise ValueError("The version of the model is not supported.")

            if common_options['overlap']:
                if ckpt_version == 2 and json_model.get("model_params",{}).get("overlap", None) is None:
                    log.error("The overlap parameters are not provided in the json model file, but the input is set to True.")
                    raise ValueError("The overlap parameters are not provided in the json model file, but the input is set to True.")
                elif ckpt_version == 1 and json_model.get("overlap", None) is None:
                    log.error("The overlap parameters are not provided in the json model file, but the input is set to True.")
                    raise ValueError("The overlap parameters are not provided in the json model file, but the input is set to True.")
                else:
                    if ckpt_version == 1:
                        overlap_param = json_model["overlap"]
                        overlaponsite_param = None
                    elif ckpt_version == 2:
                        overlap_param = json_model["model_params"]["overlap"]
                        if "overlaponsite" in json_model["model_params"]:
                            overlaponsite_param = json_model["model_params"]["overlaponsite"]
                        else:
                            overlaponsite_param = None
                    else:
                        raise ValueError("The version of the model is not supported.")
            else:
                if ckpt_version == 2 and json_model.get("model_params",{}).get("overlap", None) is not None:
                    log.error("The overlap parameters are provided in the json model file, but the input is set to False.")
                    raise ValueError("The overlap parameters are provided in the json model file, but the input is set to False.")
                elif ckpt_version == 1 and json_model.get("overlap", None) is not None:
                    log.error("The overlap parameters are provided in the json model file, but the input is set to False.")
                    raise ValueError("The overlap parameters are provided in the json model file, but the input is set to False.")
                else:
                    overlap_param = None
                    overlaponsite_param = None

            if nnsk['soc'].get("method", None) is not None:
                if ckpt_version == 2:
                    if json_model["model_params"].get("soc", None) is None:
                        log.warning("The soc parameters are not provided in the json model file, it will be initialized randomly.")
                        soc_param = None
                    else:
                        soc_param = json_model["model_params"]["soc"]
                elif ckpt_version == 1:
                    soc_param = json_model["soc"]
                else:
                    raise ValueError("The version of the model is not supported.")
            else:
                soc_param = None

            if ckpt_version ==1:
                v1_model = {
                    "unit": ene_unit,
                    "onsite": json_model["onsite"],
                    "hopping": json_model["hopping"],
                    "overlap": overlap_param,
                    "soc": soc_param}
            else:
                v1_model = {
                    "unit": ene_unit,
                    "onsite": json_model["model_params"]["onsite"],
                    "hopping": json_model["model_params"]["hopping"],
                    "overlap": overlap_param,
                    "overlaponsite": overlaponsite_param,
                    "soc": soc_param}
            
            model = cls._from_model_v1(
                v1_model=v1_model,
                **nnsk,
                **common_options,
                transform=transform
            )

            del v1_model

        else:
            if device == 'cuda':
                if not torch.cuda.is_available():
                    device = 'cpu'
                    log.warning("CUDA is not available. The model will be loaded on CPU.")
                    common_options.update({"device": device})

            f = torch.load(checkpoint, map_location=device, weights_only=False)
            for k,v in common_options.items():
                if v is None:
                    common_options[k] = f["config"]["common_options"][k]
                    log.info(f"{k} is not provided in the input json, set to the value {common_options[k]} in model ckpt.")
            for k,v in nnsk.items():
                if v is None and k != "push" :
                    if k=='atomic_radius' and f["config"]["model_options"]["nnsk"].get(k, None) is None:
                        nnsk[k] = 'v1'
                    else:
                        nnsk[k] = f["config"]["model_options"]["nnsk"][k]
                    log.info(f"{k} is not provided in the input json, set to the value {nnsk[k]} in model ckpt.")

            elements = list(common_options['basis'].keys())
            rs_out = {}
            if isinstance(nnsk['hopping']['rs'], dict):
                for irs, value in nnsk['hopping']['rs'].items():
                    parts = irs.split('-')
                    # 检查是否是一对元素且都在elements列表中
                    if (len(parts) == 2 and parts[0] in elements and parts[1] in elements) or \
                        (len(parts) == 1 and parts[0] in elements):
                        rs_out[irs] = value
            else:
                rs_out = nnsk['hopping']['rs']
            
            nnsk['hopping']['rs'] = rs_out
                
            model = cls(**common_options, **nnsk, transform=transform)

            if f["config"]["common_options"]["basis"] == common_options["basis"] and \
                f["config"]["model_options"]["nnsk"]["onsite"] == model.model_options["nnsk"]["onsite"] and \
                f["config"]["model_options"]["nnsk"]["hopping"] == model.model_options["nnsk"]["hopping"] and \
                    f["config"]["model_options"]["nnsk"]["soc"] == model.model_options["nnsk"]["soc"]:
                
                model.load_state_dict(f["model_state_dict"])
            
            else:
                #TODO: handle the situation when ref_model config is not the same as the current model
                # load hopping
                ref_idp =  OrbitalMapper(f["config"]["common_options"]["basis"], method="sktb")
                idp = OrbitalMapper(common_options["basis"], method="sktb")

                ref_idp.get_orbpair_maps()
                idp.get_orbpair_maps()


                params = f["model_state_dict"]["hopping_param"]
                for bond in ref_idp.bond_types:
                    if bond in idp.bond_types:
                        iasym, jasym = bond.split("-")
                        for ref_forbpair in ref_idp.orbpair_maps.keys():
                            rfiorb, rfjorb = ref_forbpair.split("-")
                            riorb, rjorb = ref_idp.full_basis_to_basis[iasym].get(rfiorb), ref_idp.full_basis_to_basis[jasym].get(rfjorb)
                            fiorb, fjorb = idp.basis_to_full_basis[iasym].get(riorb), idp.basis_to_full_basis[jasym].get(rjorb)
                            if fiorb is not None and fjorb is not None:
                                sli = idp.orbpair_maps.get(f"{fiorb}-{fjorb}")
                                b = bond
                                if sli is None:
                                    sli = idp.orbpair_maps.get(f"{fjorb}-{fiorb}")
                                    b = f"{jasym}-{iasym}"
                                model.hopping_param.data[idp.bond_to_type[b],sli] = \
                                    params[ref_idp.bond_to_type[b],ref_idp.orbpair_maps[ref_forbpair]]

                # load overlap
                if hasattr(model, "overlap_param") and f["model_state_dict"].get("overlap_param") != None:
                    params = f["model_state_dict"]["overlap_param"]
                    for bond in ref_idp.bond_types:
                        if bond in idp.bond_types:
                            iasym, jasym = bond.split("-")
                            for ref_forbpair in ref_idp.orbpair_maps.keys():
                                rfiorb, rfjorb = ref_forbpair.split("-")
                                riorb, rjorb = ref_idp.full_basis_to_basis[iasym].get(rfiorb), ref_idp.full_basis_to_basis[jasym].get(rfjorb)
                                fiorb, fjorb = idp.basis_to_full_basis[iasym].get(riorb), idp.basis_to_full_basis[jasym].get(rjorb)
                                if fiorb is not None and fjorb is not None:
                                    sli = idp.orbpair_maps.get(f"{fiorb}-{fjorb}")
                                    b = bond
                                    if sli is None:
                                        sli = idp.orbpair_maps.get(f"{fjorb}-{fiorb}")
                                        b = f"{jasym}-{iasym}"
                                    model.overlap_param.data[idp.bond_to_type[b],sli] = \
                                        params[ref_idp.bond_to_type[b],ref_idp.orbpair_maps[ref_forbpair]]
                
                # load overlaponsite
                if hasattr(model, "overlaponsite_param") and f["model_state_dict"].get("overlaponsite_param") != None:
                    params = f["model_state_dict"]["overlaponsite_param"]
                    ref_idp.get_skonsite_maps()
                    idp.get_skonsite_maps()
                    for asym in ref_idp.type_names:
                        if asym in idp.type_names:
                            for ref_forbpair in ref_idp.skonsite_maps.keys():
                                rfiorb, rfjorb = ref_forbpair.split("-")
                                riorb, rjorb = ref_idp.full_basis_to_basis[asym].get(rfiorb), ref_idp.full_basis_to_basis[asym].get(rfjorb)
                                fiorb, fjorb = idp.basis_to_full_basis[asym].get(riorb), idp.basis_to_full_basis[asym].get(rjorb)
                                if fiorb != None and fjorb != None and fiorb != fjorb:
                                    model.overlaponsite_param.data[idp.chemical_symbol_to_type[asym], idp.skonsite_maps[fiorb+"-"+fjorb]] = \
                                        params[ref_idp.chemical_symbol_to_type[asym], ref_idp.skonsite_maps[rfiorb+"-"+rfjorb]]
                
                # load onsite
                if model.onsite_param != None and f["model_state_dict"].get("onsite_param") != None:
                    params = f["model_state_dict"]["onsite_param"]
                    ref_idp.get_skonsite_maps()
                    idp.get_skonsite_maps()
                    for asym in ref_idp.type_names:
                        if asym in idp.type_names:
                            for ref_forbpair in ref_idp.skonsite_maps.keys():
                                rfiorb, rfjorb = ref_forbpair.split("-")
                                riorb, rjorb = ref_idp.full_basis_to_basis[asym].get(rfiorb), ref_idp.full_basis_to_basis[asym].get(rfjorb)
                                fiorb, fjorb = idp.basis_to_full_basis[asym].get(riorb), idp.basis_to_full_basis[asym].get(rjorb)
                                if fiorb and fjorb is not None:
                                    model.onsite_param.data[idp.chemical_symbol_to_type[asym], idp.skonsite_maps[fiorb+"-"+fjorb]] = \
                                        params[ref_idp.chemical_symbol_to_type[asym], ref_idp.skonsite_maps[rfiorb+"-"+rfjorb]]

                if hasattr(model, "soc_param") and model.soc_param is not None and f["model_state_dict"].get("soc_param", None) != None:
                    ref_idp.get_sksoc_maps()
                    idp.get_sksoc_maps()
                    params = f["model_state_dict"]["soc_param"]
                    for asym in ref_idp.type_names:
                        if asym in idp.type_names:
                            for ref_forb in ref_idp.sksoc_maps.keys():
                                rorb = ref_idp.full_basis_to_basis[asym].get(ref_forb)
                                forb = idp.basis_to_full_basis[asym].get(rorb)
                                if forb is not None:
                                    model.soc_param.data[idp.chemical_symbol_to_type[asym],idp.sksoc_maps[forb]] = \
                                        params[ref_idp.chemical_symbol_to_type[asym],ref_idp.sksoc_maps[ref_forb]]
                # load strain
                if hasattr(model, "strain_param") and f["model_state_dict"].get("strain_param") != None:
                    params = f["model_state_dict"]["strain_param"]
                    for bond in ref_idp.bond_types:
                        if bond in idp.bond_types:
                            iasym, jasym = bond.split("-")
                            for ref_forbpair in ref_idp.orbpair_maps.keys():
                                rfiorb, rfjorb = ref_forbpair.split("-")
                                riorb, rjorb = ref_idp.full_basis_to_basis[iasym].get(rfiorb), ref_idp.full_basis_to_basis[jasym].get(rfjorb)
                                fiorb, fjorb = idp.basis_to_full_basis[iasym].get(riorb), idp.basis_to_full_basis[jasym].get(rjorb)
                                if fiorb is not None and fjorb is not None:
                                    sli = idp.orbpair_maps.get(f"{fiorb}-{fjorb}")
                                    b = bond
                                    if sli is None:
                                        sli = idp.orbpair_maps.get(f"{fjorb}-{fiorb}")
                                        b = f"{jasym}-{iasym}"
                                    model.strain_param.data[idp.bond_to_type[b], sli] = \
                                        params[ref_idp.bond_to_type[b],ref_idp.orbpair_maps[ref_forbpair]]

            del f
        
        if freeze:  
            model.freezefunc(freeze)

        return model

    @classmethod
    def _from_model_v1(
        cls, 
        v1_model: dict, 
        basis: Dict[str, Union[str, list]]=None,
        idp_sk: Union[OrbitalMapper, None]=None, 
        onsite: Dict={"method": "none"},
        hopping: Dict={"method": "powerlaw", "rs":6.0, "w": 0.2},
        overlap: bool = False,
        soc:Dict = {},
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
        std: float = 0.01,
        freeze: Union[bool,str,list] = False,
        push: Union[bool,None,dict] = False,
        transform: bool = True,
        atomic_radius: Union[str, Dict] = None,
        **kwargs
        ):
        # could support json file and .pth file checkpoint of nnsk

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        dtype = dtype
        device = device

        if basis is not None:
            assert idp_sk is None
            idp_sk = OrbitalMapper(basis, method="sktb")
        else:
            assert idp_sk is not None
        
            
        basis = idp_sk.basis
        idp_sk.get_orbpair_maps()
        idp_sk.get_skonsite_maps()
        # idp_sk.get_orbpair_soc_maps()
        idp_sk.get_sksoc_maps()

        if device == 'cuda':
            if not torch.cuda.is_available():
                device = 'cpu'
                log.warning("CUDA is not available. The model will be loaded on CPU.")
                
        nnsk_model = cls(basis=basis, idp_sk=idp_sk,  onsite=onsite,
                          hopping=hopping, overlap=overlap, soc=soc, std=std,freeze=freeze, push=push, dtype=dtype, device=device,
                          atomic_radius=atomic_radius, transform=transform)

        onsite_param = v1_model["onsite"]
        hopping_param = v1_model["hopping"]
        soc_param = v1_model["soc"]
        if overlap:
            overlap_param = v1_model["overlap"]
            overlaponsite_param = v1_model["overlaponsite"]

        ene_unit = v1_model["unit"]

        assert len(hopping) > 0, "The hopping parameters should be provided."


        # load hopping params
        for orbpair, skparam in hopping_param.items():
            skparam = torch.tensor(skparam, dtype=dtype, device=device)
            if ene_unit == "Hartree" and hopping['method'] not in ['NRL', 'NRL1', 'NRL2']:
                skparam[0] *= 13.605662285137 * 2
            iasym, jasym, iorb, jorb, num = list(orbpair.split("-"))
            num = int(num)
            ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])
            fiorb, fjorb = idp_sk.basis_to_full_basis[iasym][iorb], idp_sk.basis_to_full_basis[jasym][jorb]
            
            
            if idp_sk.full_basis.index(fiorb) <= idp_sk.full_basis.index(fjorb):
                nline = idp_sk.transform_bond(iatomic_numbers=ian, jatomic_numbers=jan)
                nidx = idp_sk.orbpair_maps[f"{fiorb}-{fjorb}"].start + num
            else:
                nline = idp_sk.transform_bond(iatomic_numbers=jan, jatomic_numbers=ian)
                nidx = idp_sk.orbpair_maps[f"{fjorb}-{fiorb}"].start + num

            nnsk_model.hopping_param.data[nline, nidx] = skparam
            if ian != jan and fiorb == fjorb:
                nline = idp_sk.transform_bond(iatomic_numbers=jan, jatomic_numbers=ian)
                nnsk_model.hopping_param.data[nline, nidx] = skparam
        
        if overlap:
            for orbpair, skparam in overlap_param.items():
                skparam = torch.tensor(skparam, dtype=dtype, device=device)
                iasym, jasym, iorb, jorb, num = list(orbpair.split("-"))
                num = int(num)
                ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])
                fiorb, fjorb = idp_sk.basis_to_full_basis[iasym][iorb], idp_sk.basis_to_full_basis[jasym][jorb]


                if idp_sk.full_basis.index(fiorb) <= idp_sk.full_basis.index(fjorb):
                    nline = idp_sk.transform_bond(iatomic_numbers=ian, jatomic_numbers=jan)
                    nidx = idp_sk.orbpair_maps[f"{fiorb}-{fjorb}"].start + num
                else:
                    nline = idp_sk.transform_bond(iatomic_numbers=jan, jatomic_numbers=ian)
                    nidx = idp_sk.orbpair_maps[f"{fjorb}-{fiorb}"].start + num

                nnsk_model.overlap_param.data[nline, nidx] = skparam
                if ian != jan and fiorb == fjorb:
                    nline = idp_sk.transform_bond(iatomic_numbers=jan, jatomic_numbers=ian)
                    nnsk_model.overlap_param.data[nline, nidx] = skparam
            
            # load overlaponsite
            if overlaponsite_param is not None:
                for orbpair, skparam in overlaponsite_param.items():
                    skparam = torch.tensor(skparam, dtype=dtype, device=device)
                    iasym, iorb, jorb, num = list(orbpair.split("-"))
                    num = int(num)
                    ian = torch.tensor(atomic_num_dict[iasym])
                    fiorb, fjorb = idp_sk.basis_to_full_basis[iasym][iorb], idp_sk.basis_to_full_basis[iasym][jorb]
                    nline = idp_sk.transform_atom(atomic_numbers=ian)
                    nidx = idp_sk.skonsite_maps[fiorb+"-"+fjorb].start + num
                    nnsk_model.overlaponsite_param.data[nline, nidx] = skparam


        # load onsite params, differently with onsite mode
        if onsite["method"] == "strain":
            for orbpair, skparam in onsite_param.items():
                skparam = torch.tensor(skparam, dtype=dtype, device=device)
                if ene_unit == "Hartree":
                    skparam[0] *= 13.605662285137 * 2
                if len(list(orbpair.split("-"))) == 5:
                    iasym, jasym, iorb, jorb, num = list(orbpair.split("-"))
                else:
                    continue
                num = int(num)
                ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])

                fiorb, fjorb = idp_sk.basis_to_full_basis[iasym][iorb], idp_sk.basis_to_full_basis[iasym][jorb]

                nline = idp_sk.transform_bond(iatomic_numbers=ian, jatomic_numbers=jan)
                if idp_sk.full_basis.index(fiorb) <= idp_sk.full_basis.index(fjorb):
                    nidx = idp_sk.orbpair_maps[f"{fiorb}-{fjorb}"].start + num
                else:
                    nidx = idp_sk.orbpair_maps[f"{fjorb}-{fiorb}"].start + num

                nnsk_model.strain_param.data[nline, nidx] = skparam

                # if ian == jan:
                #     nidx = idp_sk.pair_maps[f"{fjorb}-{fiorb}"].start + num
                #     nnsk_model.strain_param.data[nline, nidx] = skparam

        elif onsite["method"] == "none":
            pass
        else:
            for orbon, skparam in onsite_param.items():
                skparam = torch.tensor(skparam, dtype=dtype, device=device)
                if ene_unit == "Hartree" and onsite["method"] not in ['NRL', 'NRL1', 'NRL2']:
                    skparam *= 13.605662285137 * 2
                orbon_s = list(orbon.split("-"))
                if len(orbon_s) == 3:
                    iasym, iorb, num = orbon_s
                    jorb = iorb
                else:
                    iasym, iorb, jorb, num = orbon_s

                num = int(num)
                ian = torch.tensor(atomic_num_dict[iasym])
                fiorb = idp_sk.basis_to_full_basis[iasym][iorb]
                fjorb = idp_sk.basis_to_full_basis[iasym][jorb]

                nline = idp_sk.transform_atom(atomic_numbers=ian)
                nidx = idp_sk.skonsite_maps[fiorb+"-"+fjorb].start + num

                nnsk_model.onsite_param.data[nline, nidx] = skparam
        if soc.get("method", None) is not None:
            if soc["method"] == "none":
                pass
            elif soc_param is None:
                pass
            else:
                assert soc_param is not None, "The soc parameters should be provided."
                for orbon, skparam in soc_param.items():
                    skparam = torch.tensor(skparam, dtype=dtype, device=device)
                    if ene_unit == "Hartree":
                        skparam *= 13.605662285137 * 2
                    iasym, iorb, num = list(orbon.split("-"))
                    num = int(num)
                    ian = torch.tensor(atomic_num_dict[iasym])
                    fiorb = idp_sk.basis_to_full_basis[iasym][iorb]

                    nline = idp_sk.transform_atom(atomic_numbers=ian)
                    nidx = idp_sk.sksoc_maps[fiorb].start + num

                    nnsk_model.soc_param.data[nline, nidx] = skparam

        if freeze:  
            nnsk_model.freezefunc(freeze)

        return nnsk_model
    
    def save(self,filepath):
        obj = {}
        model_options=self.model_options
        common_options={
            "basis":self.basis,
            "overlap":hasattr(self, "overlap_param"),
            "dtype":self.dtype,
            "device":self.device
        }
        obj.update({"config": {"model_options": model_options, "common_options": common_options}})
        obj.update({"model_state_dict": self.state_dict()})
        torch.save(obj, f=filepath)
        
    def to_json(self, version=2, basisref=None):
        """
        basisref= {'Atom':{"s":"2s", "p":"2p", "d":"3d", "f":"4f"}}
        """
        
        to_uniform = False
        new_basis = self.basis.copy()
        if basisref is not None:
            if  self.model_options['nnsk']['onsite']['method'] in ['uniform_noref']:
                for atom, orb in self.basis.items():
                    new_basis[atom] = []
                    if atom not in basisref:
                        raise ValueError("The atom in the model basis should be in the basisref.")
                    for o in orb:
                        if o not in ['s', 'p', 'd', 'f']:
                            raise ValueError("For uniform_noref mode, the orb in the model basis should be in ['s', 'p', 'd', 'f'].")
                        if o not in list(basisref[atom].keys()):
                            raise ValueError("The orb in the model basis should be in the basisref.")
                        new_basis[atom].append(basisref[atom][o]) 
                to_uniform = True
            else:
                print("The basisref is not used. since the onsite method is not uniform_noref.")

        ckpt = {}
        # load hopping params
        hopping = self.hopping_param.data.cpu().clone().numpy()
                
        ckpt['version'] = version
        ckpt['unit'] = 'eV'

        hopping_param = {}
        basis = self.idp_sk.basis
        if to_uniform:
            basis = new_basis

        if isinstance(self.dtype, str):
            dtype = self.dtype
        else:
            dtype = self.dtype.__str__().split('.')[-1]
        is_overlap = hasattr(self, "overlap_param")
        dd = "cpu" if self.device == torch.device("cpu") else "cuda"
        common_options = {
            "basis": basis,
            "dtype": dtype,
            "device": dd,
            "overlap": is_overlap,
        }

        mode_opt = self.model_options.copy()
        if to_uniform:
            mode_opt['nnsk']['onsite']['method'] = 'uniform'
        if version == 2:
            ckpt.update({"model_options": mode_opt, 
                        "common_options": common_options})


        for bt in self.idp_sk.bond_types:
            iasym, jasym = bt.split("-")
            ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])
            pos_line = self.idp_sk.transform_bond(ian, jan)
            rev_line = self.idp_sk.transform_bond(jan, ian)
            for orbpair, slices in self.idp_sk.orbpair_maps.items():
                fiorb, fjorb = orbpair.split("-")
                iorb = self.idp_sk.full_basis_to_basis[iasym].get(fiorb)
                jorb = self.idp_sk.full_basis_to_basis[jasym].get(fjorb)
                if to_uniform:
                    iorb = basisref[iasym][iorb]
                    jorb = basisref[jasym][jorb]
                if iorb != None and jorb != None:
                    # iasym-jasym-iorb-jorb
                    for i in range(slices.stop-slices.start):
                        if ian < jan:
                            continue
                        elif ian > jan:
                            if fiorb == fjorb: # this might have problems
                                hopping_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = ((hopping[pos_line, slices][i] + hopping[rev_line, slices][i])*0.5).tolist()
                            else:
                                hopping_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = hopping[pos_line, slices][i].tolist()
                                iiorb = self.idp_sk.full_basis_to_basis[iasym].get(fjorb)
                                jjorb = self.idp_sk.full_basis_to_basis[jasym].get(fiorb)
                                if to_uniform:
                                    iiorb = basisref[iasym][iiorb]
                                    jjorb = basisref[jasym][jjorb]
                                hopping_param[f"{iasym}-{jasym}-{iiorb}-{jjorb}-{i}"] = hopping[rev_line, slices][i].tolist()
                        elif ian == jan:
                            if self.idp_sk.full_basis.index(fiorb) <= self.idp_sk.full_basis.index(fjorb):
                                hopping_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = hopping[pos_line, slices][i].tolist()
                            #if fiorb != fjorb:
                            #   hopping_param[f"{iasym}-{jasym}-{jorb}-{iorb}-{i}"] = hopping[pos_line, slices][i].tolist()
                        else:
                            raise ValueError("The atomic number should be the same or different.")
        
        if is_overlap:
            overlap_param={}
            overlap = self.overlap_param.data.cpu().clone().numpy()
            for bt in self.idp_sk.bond_types:
                iasym, jasym = bt.split("-")
                ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])
                pos_line = self.idp_sk.transform_bond(ian, jan)
                rev_line = self.idp_sk.transform_bond(jan, ian)
                for orbpair, slices in self.idp_sk.orbpair_maps.items():
                    fiorb, fjorb = orbpair.split("-")
                    iorb = self.idp_sk.full_basis_to_basis[iasym].get(fiorb)
                    jorb = self.idp_sk.full_basis_to_basis[jasym].get(fjorb)
                    if to_uniform:
                        iorb = basisref[iasym][iorb]
                        jorb = basisref[jasym][jorb]
                    if iorb != None and jorb != None:
                        # iasym-jasym-iorb-jorb
                        for i in range(slices.stop-slices.start):
                            if ian < jan:
                                continue
                            elif ian > jan:
                                if fiorb == fjorb: # this might have problems
                                    overlap_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = ((overlap[pos_line, slices][i] + overlap[rev_line, slices][i])*0.5).tolist()
                                else:
                                    overlap_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = overlap[pos_line, slices][i].tolist()
                                    iiorb = self.idp_sk.full_basis_to_basis[iasym].get(fjorb)
                                    jjorb = self.idp_sk.full_basis_to_basis[jasym].get(fiorb)
                                    if to_uniform:
                                        iiorb = basisref[iasym][iiorb]
                                        jjorb = basisref[jasym][jjorb]
                                    overlap_param[f"{iasym}-{jasym}-{iiorb}-{jjorb}-{i}"] = overlap[rev_line, slices][i].tolist()
                            elif ian == jan:
                                if self.idp_sk.full_basis.index(fiorb) <= self.idp_sk.full_basis.index(fjorb):
                                    overlap_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = overlap[pos_line, slices][i].tolist()
                            else:
                                raise ValueError("The atomic number should be the same or different.")    
            if not all(self.idp_sk.mask_diag):
                # write out onsite overlap param for non diag term
                overlaponsite = self.overlaponsite_param.data.cpu().clone().numpy()
                overlaponsite_param = {}
                for asym in self.idp_sk.type_names:
                    for orbpair, slices in self.idp_sk.skonsite_maps.items():
                        fiorb, fjorb = orbpair.split("-")
                        if fiorb != fjorb:
                            iorb = self.idp_sk.full_basis_to_basis[asym][fiorb]
                            jorb = self.idp_sk.full_basis_to_basis[asym][fjorb]
                            if to_uniform:
                                iorb = basisref[asym][iorb]
                                jorb = basisref[asym][jorb]
                            for i in range(slices.start, slices.stop): 
                                ind = i-slices.start
                                overlaponsite_param[f"{asym}-{iorb}-{jorb}-{ind}"] = (overlaponsite[self.idp_sk.chemical_symbol_to_type[asym], i]).tolist()


        if hasattr(self, "strain_param"):
            strain = self.strain_param.data.cpu().clone().numpy()
            onsite_param = {}
            for bt in self.idp_sk.bond_types:
                iasym, jasym = bt.split("-")
                ian, jan = torch.tensor(atomic_num_dict[iasym]), torch.tensor(atomic_num_dict[jasym])
                for orbpair, slices in self.idp_sk.orbpair_maps.items():
                    fiorb, fjorb = orbpair.split("-")
                    iorb = self.idp_sk.full_basis_to_basis[iasym].get(fiorb)
                    jorb = self.idp_sk.full_basis_to_basis[jasym].get(fjorb)
                    if to_uniform:
                        iorb = basisref[iasym][iorb]
                        jorb = basisref[jasym][jorb]
                    if iorb != None and jorb != None and self.idp_sk.full_basis.index(fiorb) <= self.idp_sk.full_basis.index(fjorb):
                        for i in range(slices.stop-slices.start):
                            onsite_param[f"{iasym}-{jasym}-{iorb}-{jorb}-{i}"] = strain[pos_line, slices][i].tolist()

        # onsite need more test and work
        elif hasattr(self, "onsite_param") and self.onsite_param is not None:
            onsite =self.onsite_param.data.cpu().clone().numpy()
            onsite_param = {}
            for asym in self.idp_sk.type_names:
                for orbpair, slices in self.idp_sk.skonsite_maps.items():
                    fiorb, fjorb = orbpair.split("-")
                    iorb = self.idp_sk.full_basis_to_basis[asym][fiorb]
                    jorb = self.idp_sk.full_basis_to_basis[asym][fjorb]
                    if to_uniform:
                        iorb = basisref[asym][iorb]
                        jorb = basisref[asym][jorb]
                        ref_ene = onsite_energy_database[asym][iorb]
                    else:
                        ref_ene = 0.0
                    if iorb == jorb:
                        for i in range(slices.start, slices.stop): 
                            ind = i-slices.start      
                            onsite_param[f"{asym}-{iorb}-{ind}"] = (onsite[self.idp_sk.chemical_symbol_to_type[asym], i]-ref_ene).tolist()
                    #else:
                    #    for i in range(slices.start, slices.stop): 
                    #        ind = i-slices.start
                    #        onsite_param[f"{asym}-{iorb}-{jorb}-{ind}"] = (onsite[self.idp_sk.chemical_symbol_to_type[asym], i]).tolist()
        else:
            onsite_param = {}

        has_soc = hasattr(self, "soc_param") and self.soc_param is not None
        if has_soc:
            soc = self.soc_param.data.cpu().clone().numpy()
            soc_param = {}
            for asym in self.idp_sk.type_names:
                for fiorb, slices in self.idp_sk.sksoc_maps.items():
                    iorb = self.idp_sk.full_basis_to_basis[asym][fiorb]
                    if to_uniform:
                        iorb = basisref[asym][iorb]
                    for i in range(slices.start, slices.stop): 
                        ind = i-slices.start      
                        soc_param[f"{asym}-{iorb}-{ind}"] = (np.abs(soc[self.idp_sk.chemical_symbol_to_type[asym], i])).tolist()
        
        model_params = {
                    "onsite": onsite_param,
                    "hopping": hopping_param
        }
        if is_overlap:
            model_params.update({"overlap": overlap_param})
            if not all(self.idp_sk.mask_diag):
                model_params.update({"overlaponsite": overlaponsite_param})

        if has_soc:
            model_params.update({"soc": soc_param})

        if version ==1:
            ckpt.update(model_params)
        elif version == 2:
            ckpt.update({"model_params": model_params})
        else:
            raise ValueError("The version of the model is not supported.")
        
        return ckpt