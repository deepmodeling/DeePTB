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
from dptb.utils.constants import atomic_num_dict_r, atomic_num_dict
from dptb.nn.hamiltonian import SKHamiltonian
from dptb.utils.tools import j_loader

class NNSK(torch.nn.Module):
    name = "nnsk"
    def __init__(
            self,
            basis: Dict[str, Union[str, list]]=None,
            idp_sk: Union[OrbitalMapper, None]=None,
            onsite: Dict={"method": "none"},
            hopping: Dict={"method": "powerlaw", "rs":6.0, "w": 0.2},
            overlap: bool = False,
            dtype: Union[str, torch.dtype] = torch.float32, 
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            freeze: bool = False,
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
        self.model_options = {
            "nnsk":{
                "onsite": onsite, 
                "hopping": hopping,
                "freeze": freeze,                
                }
            }

        # init_onsite, hopping, overlap formula

        self.onsite_fn = OnsiteFormula(idp=self.idp_sk, functype=self.onsite_options["method"], dtype=dtype, device=device)
        self.hopping_fn = HoppingFormula(functype=self.hopping_options["method"])
        if overlap:
            self.overlap_fn = HoppingFormula(functype=self.hopping_options["method"], overlap=True)
        
        # init_param
        # 
        hopping_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
        nn.init.normal_(hopping_param, mean=0.0, std=0.01)
        self.hopping_param = torch.nn.Parameter(hopping_param)
        if overlap:
            overlap_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(overlap_param, mean=0.0, std=0.01)
            self.overlap_param = torch.nn.Parameter(overlap_param)

        if self.onsite_options["method"] == "strain":
            self.onsite_param = None
        elif self.onsite_options["method"] == "none":
            self.onsite_param = None
        elif self.onsite_options["method"] in ["NRL", "uniform"]:
            onsite_param = torch.empty([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, self.onsite_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(onsite_param, mean=0.0, std=0.01)
            self.onsite_param = torch.nn.Parameter(onsite_param)
        else:
            raise NotImplementedError(f"The onsite method {self.onsite_options['method']} is not implemented.")
        
        if self.onsite_options["method"] == "strain":
            # AB [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # AA [...]
            # but need to map to all pairs and all orbital pairs like AB, AA, BB, BA for [ss, sp, sd, ps, pp, pd, ds, dp, dd]
            # with this map: BA[sp, sd] = AB[ps, ds]
            strain_param = torch.empty([len(self.idp_sk.bond_types), self.idp_sk.reduced_matrix_element, self.hopping_fn.num_paras], dtype=self.dtype, device=self.device)
            nn.init.normal_(strain_param, mean=0.0, std=0.01)
            self.strain_param = torch.nn.Parameter(strain_param)
            # symmetrize the env for same atomic spices
            
        self.hamiltonian = SKHamiltonian(idp_sk=self.idp_sk, onsite=True, dtype=self.dtype, device=self.device, strain=hasattr(self, "strain_param"))
        if overlap:
            self.overlap = SKHamiltonian(idp_sk=self.idp_sk, onsite=False, edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, node_field=AtomicDataDict.NODE_OVERLAP_KEY, dtype=self.dtype, device=self.device)
        self.idp = self.hamiltonian.idp
        if freeze:
            for (name, param) in self.named_parameters():
                param.requires_grad = False

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        # get the env and bond from the data
        # calculate the sk integrals
        # calculate the onsite
        # calculate the hopping
        # calculate the overlap
        # return the data with updated edge/node features

        # map the parameters to the edge/node/env features
        
        # compute integrals from parameters using hopping and onsite clas

        # symmetrize the bond for same atomic spices
        # reflect_keys = np.array(list(self.idp_sk.pair_maps.keys()), dtype="str").reshape(len(self.idp_sk.full_basis), len(self.idp_sk.full_basis)).transpose(1,0).reshape(-1)
        # params = 0.5 * self.hopping_param.data[self.idp_sk.transform_reduced_bond(torch.tensor(list(self.idp_sk._valid_set)), torch.tensor(list(self.idp_sk._valid_set)))]
        # reflect_params = torch.zeros_like(params)
        # for k, k_r in zip(self.idp_sk.pair_maps.keys(), reflect_keys):
        #     reflect_params[:,self.idp_sk.pair_maps[k],:] += params[:,self.idp_sk.pair_maps[k_r],:]
        # self.hopping_param.data[self.idp_sk.transform_reduced_bond(torch.tensor(list(self.idp_sk._valid_set)), torch.tensor(list(self.idp_sk._valid_set)))] = \
        #     reflect_params + params
        
        # if hasattr(self, "overlap"):
        #     params = 0.5 * self.overlap_param.data[self.idp_sk.transform_reduced_bond(torch.tensor(list(self.idp_sk._valid_set)), torch.tensor(list(self.idp_sk._valid_set)))]
        #     reflect_params = torch.zeros_like(params)
        #     for k, k_r in zip(self.idp_sk.pair_maps.keys(), reflect_keys):
        #         reflect_params[:,self.idp_sk.pair_maps[k],:] += params[:,self.idp_sk.pair_maps[k_r],:]
        #     self.overlap_param.data[self.idp_sk.transform_reduced_bond(torch.tensor(list(self.idp_sk._valid_set)), torch.tensor(list(self.idp_sk._valid_set)))] = \
        #         reflect_params + params
        
        # # in strain case, all env pair need to be symmetrized
        # if self.onsite_fn.functype == "strain":
        #     params = 0.5 * self.strain_param.data
        #     reflect_params = torch.zeros_like(params)
        #     for k, k_r in zip(self.idp_sk.pair_maps.keys(), reflect_keys):
        #         reflect_params[:,self.idp_sk.pair_maps[k],:] += params[:,self.idp_sk.pair_maps[k_r],:]
        #     self.strain_param.data = reflect_params + params

        reflective_bonds = np.array([self.idp_sk.bond_to_type["-".join(self.idp_sk.type_to_bond[i].split("-")[::-1])] for i  in range(len(self.idp_sk.bond_types))])
        params = self.hopping_param.data
        reflect_params = params[reflective_bonds]
        for k in self.idp_sk.orbpair_maps.keys():
            iorb, jorb = k.split("-")
            if iorb == jorb:
                self.hopping_param.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])
        if hasattr(self, "overlap"):
            params = self.overlap_param.data
            reflect_params = params[reflective_bonds]
            for k in self.idp_sk.orbpair_maps.keys():
                iorb, jorb = k.split("-")
                if iorb == jorb:
                    self.overlap_param.data[:,self.idp_sk.orbpair_maps[k],:] = 0.5 * (params[:,self.idp_sk.orbpair_maps[k],:] + reflect_params[:,self.idp_sk.orbpair_maps[k],:])

        
        data = AtomicDataDict.with_edge_vectors(data, with_lengths=True)

        # edge_number = data[AtomicDataDict.ATOMIC_NUMBERS_KEY][data[AtomicDataDict.EDGE_INDEX_KEY]].reshape(2, -1)
        # edge_index = self.idp_sk.transform_reduced_bond(*edge_number)
        edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten() # it is bond_type index, transform it to reduced bond index
        edge_number = self.idp_sk.untransform_bond(edge_index).T
        edge_index = self.idp_sk.transform_bond(*edge_number)

        r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[edge_number-1].sum(0)

        data[AtomicDataDict.EDGE_FEATURES_KEY] = self.hopping_fn.get_skhij(
            rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
            paraArray=self.hopping_param[edge_index], # [N_edge, n_pairs, n_paras],
            **self.hopping_options,
            r0=r0
            ) # [N_edge, n_pairs]

        if hasattr(self, "overlap"):
            equal_orbpair = torch.zeros(self.idp_sk.reduced_matrix_element, dtype=self.dtype, device=self.device)
            for orbpair_key, slices in self.idp_sk.orbpair_maps.items():
                if orbpair_key.split("-")[0] == orbpair_key.split("-")[1]:
                    equal_orbpair[slices] = 1.0
            paraconst = edge_number[0].eq(edge_number[1]).float().view(-1, 1) * equal_orbpair.unsqueeze(0)

            data[AtomicDataDict.EDGE_OVERLAP_KEY] = self.overlap_fn.get_sksij(
                rij=data[AtomicDataDict.EDGE_LENGTH_KEY],
                paraArray=self.overlap_param[edge_index],
                paraconst=paraconst,
                **self.hopping_options,
                r0=r0,
                )

        atomic_numbers = self.idp_sk.untransform_atom(data[AtomicDataDict.ATOM_TYPE_KEY].flatten())
        if self.onsite_fn.functype == "NRL":
            data = AtomicDataDict.with_env_vectors(data, with_lengths=True)
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
            
            r0 = 0.5*bond_length_list.type(self.dtype).to(self.device)[onsitenv_number-1].sum(0)
            onsitenv_params = self.hopping_fn.get_skhij(
            rij=data[AtomicDataDict.ONSITENV_LENGTH_KEY],
            paraArray=self.strain_param[onsitenv_index], # [N_edge, n_pairs, n_paras],
            r0=r0,
            **self.onsite_options,
            ) # [N_edge, n_pairs]
            
            data[AtomicDataDict.ONSITENV_FEATURES_KEY] = onsitenv_params

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
        overlap: bool=None,
        dtype: Union[str, torch.dtype]=None, 
        device: Union[str, torch.device]=None,
        freeze: bool = None,
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
            "freeze": freeze,
        }


        if checkpoint.split(".")[-1] == "json":
            for k,v in common_options.items():
                assert v is not None, f"You need to provide {k} when you are initializing a model from a json file."
            for k,v in nnsk.items():
                assert v is not None, f"You need to provide {k} when you are initializing a model from a json file."

            v1_model = j_loader(checkpoint)
            model = cls._from_model_v1(
                v1_model=v1_model,
                **nnsk,
                **common_options,
            )

            del v1_model

        else:
            f = torch.load(checkpoint, map_location=device)
            for k,v in common_options.items():
                if v is None:
                    common_options[k] = f["config"]["common_options"][k]
            for k,v in nnsk.items():
                if v is None:
                    nnsk[k] = f["config"]["model_options"]["nnsk"][k]

            model = cls(**common_options, **nnsk)

            if f["config"]["common_options"]["basis"] == common_options["basis"] and \
                f["config"]["model_options"] == model.model_options:
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
                            riorb, rjorb = ref_idp.full_basis_to_basis[iasym][rfiorb], ref_idp.full_basis_to_basis[jasym][rfjorb]
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
                                riorb, rjorb = ref_idp.full_basis_to_basis[rfiorb], ref_idp.full_basis_to_basis[rfjorb]
                                fiorb, fjorb = idp.basis_to_full_basis.get(riorb), idp.basis_to_full_basis.get(rjorb)
                                if fiorb is not None and fjorb is not None:
                                    sli = idp.orbpair_maps.get(f"{fiorb}-{fjorb}")
                                    b = bond
                                    if sli is None:
                                        sli = idp.orbpair_maps.get(f"{fjorb}-{fiorb}")
                                        b = f"{jasym}-{iasym}"
                                    model.overlap_param.data[idp.bond_to_type[b],sli] = \
                                        params[ref_idp.bond_to_type[b],ref_idp.orbpair_maps[ref_forbpair]]
                
                # load onsite
                if model.onsite_param != None and f["model_state_dict"].get("onsite_param") != None:
                    params = f["model_state_dict"]["onsite_param"]
                    ref_idp.get_skonsite_maps()
                    idp.get_skonsite_maps()
                    for asym in ref_idp.type_names:
                        if asym in idp.type_names:
                            for ref_forb in ref_idp.skonsite_maps.keys():
                                rorb = ref_idp.full_basis_to_basis[asym][ref_forb]
                                forb = idp.basis_to_full_basis[asym].get(rorb)
                                if forb is not None:
                                    model.onsite_param.data[idp.chemical_symbol_to_type[asym],idp.skonsite_maps[forb]] = \
                                        params[ref_idp.chemical_symbol_to_type[asym],ref_idp.skonsite_maps[ref_forb]]

                # load strain
                if hasattr(model, "strain_param") and f["model_state_dict"].get("strain_param") != None:
                    params = f["model_state_dict"]["strain_param"]
                    for bond in ref_idp.bond_types:
                        if bond in idp.bond_types:
                            for ref_forbpair in ref_idp.orbpair_maps.keys():
                                rfiorb, rfjorb = ref_forbpair.split("-")
                                riorb, rjorb = ref_idp.full_basis_to_basis[rfiorb], ref_idp.full_basis_to_basis[rfjorb]
                                fiorb, fjorb = idp.basis_to_full_basis.get(riorb), idp.basis_to_full_basis.get(rjorb)
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
            for (name, param) in model.named_parameters():
                param.requires_grad = False
            else:
                param.requires_grad = True # in case initilizing some frozen checkpoint while with current freeze setted as False

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
        dtype: Union[str, torch.dtype] = torch.float32, 
        device: Union[str, torch.device] = torch.device("cpu"),
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

        nnsk_model = cls(basis=basis, idp_sk=idp_sk, dtype=dtype, device=device, onsite=onsite, hopping=hopping, overlap=overlap)

        onsite_param = v1_model["onsite"]
        hopping_param = v1_model["hopping"]

        assert len(hopping) > 0, "The hopping parameters should be provided."

        # load hopping params
        for orbpair, skparam in hopping_param.items():
            skparam = torch.tensor(skparam, dtype=dtype, device=device)
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
        
        # load onsite params, differently with onsite mode
        if onsite["method"] == "strain":
            for orbpair, skparam in onsite_param.items():
                skparam = torch.tensor(skparam, dtype=dtype, device=device)
                skparam[0] *= 13.605662285137 * 2
                iasym, jasym, iorb, jorb, num = list(orbpair.split("-"))
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
                skparam *= 13.605662285137 * 2
                iasym, iorb, num = list(orbon.split("-"))
                num = int(num)
                ian = torch.tensor(atomic_num_dict[iasym])
                fiorb = idp_sk.basis_to_full_basis[iasym][iorb]

                nline = idp_sk.transform_atom(atomic_numbers=ian)
                nidx = idp_sk.skonsite_maps[fiorb].start + num

                nnsk_model.onsite_param.data[nline, nidx] = skparam

        return nnsk_model
        
