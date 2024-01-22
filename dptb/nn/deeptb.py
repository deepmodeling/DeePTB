import torch.nn as nn
import torch
from typing import Union, Tuple, Optional, Callable, Dict
import torch.nn.functional as F
from dptb.nn.embedding import Embedding
from dptb.data.transforms import OrbitalMapper
from dptb.nn.base import AtomicFFN, AtomicResNet, AtomicLinear, Identity
from dptb.data import AtomicDataDict
from dptb.nn.hamiltonian import E3Hamiltonian, SKHamiltonian
from dptb.nn.nnsk import NNSK
from e3nn.o3 import Linear
from dptb.nn.rescale import E3PerSpeciesScaleShift, E3PerEdgeSpeciesScaleShift


""" if this class is called, it suggest user choose a embedding method. If not, it should directly use _sktb.py
"""

def get_neuron_config(nl):
    n = len(nl)
    if n % 2 == 0:
        d_out = nl[-1]
        nl = nl[:-1]
    config = []
    for i in range(1,len(nl)-1, 2):
        config.append({'in_features': nl[i-1], 'hidden_features': nl[i], 'out_features': nl[i+1]})

    if n % 2 == 0:
        config.append({'in_features': nl[-1], 'out_features': d_out})

    return config

class DPTB(nn.Module):
    quantities = ["hamiltonian", "energy"]
    name = "dptb"
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            overlap: bool = False,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            **kwargs,
    ):
        
        """The top level DeePTB model class.

        Parameters
        ----------
        embedding_config : dict
            _description_
        prediction_config : dict
            _description_
        basis : Dict[str, Union[str, list], None], optional
            _description_, by default None
        idp : Union[OrbitalMapper, None], optional
            _description_, by default None
        transform : bool, optional
            _description_, decide whether to transform the irreducible matrix element to the hamiltonians
        dtype : Union[str, torch.dtype], optional
            _description_, by default torch.float32
        device : Union[str, torch.device], optional
            _description_, by default torch.device("cpu")

        Raises
        ------
        NotImplementedError
            _description_
        """
        super(DPTB, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.model_options = {"embedding": embedding.copy(), "prediction": prediction.copy()}
        self.transform = transform
        
        
        self.method = prediction.get("method", "e3tb")
        # self.soc = prediction.get("soc", False)
        self.prediction = prediction

        if basis is not None:
            self.idp = OrbitalMapper(basis, method=self.method, device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_orbpair_maps()

        n_species = len(self.basis.keys())
        # initialize the embedding layer
        self.embedding = Embedding(**embedding, dtype=dtype, device=device, idp=self.idp, n_atom=n_species)
        
        # initialize the prediction layer
            
        if self.method == "sktb":
            prediction["neurons"] = [self.embedding.out_node_dim] + prediction["neurons"] + [self.idp.n_onsite_Es]
            prediction["config"] = get_neuron_config(prediction["neurons"])

            self.node_prediction_h = AtomicResNet(
                **prediction,
                in_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            prediction["neurons"][0] = self.embedding.out_edge_dim
            prediction["neurons"][-1] = self.idp.reduced_matrix_element
            prediction["config"] = get_neuron_config(prediction["neurons"])
            self.edge_prediction_h = AtomicResNet(
                **prediction,
                in_field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            if overlap:
                self.edge_prediction_s = AtomicResNet(
                    **prediction,
                    in_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    out_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    device=device, 
                    dtype=dtype
                )

        elif prediction.get("method") == "e3tb":
            self.node_prediction_h = E3PerSpeciesScaleShift(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                num_types=n_species,
                irreps_in=self.embedding.out_node_irreps,
                out_field = AtomicDataDict.NODE_FEATURES_KEY,
                shifts=0.,
                scales=1.,
                dtype=self.dtype,
                device=self.device,
                **prediction,
            )
            
            self.edge_prediction_h = E3PerEdgeSpeciesScaleShift(
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                num_types=n_species,
                irreps_in=self.embedding.out_edge_irreps,
                out_field = AtomicDataDict.EDGE_FEATURES_KEY,
                shifts=0.,
                scales=1.,
                dtype=self.dtype,
                device=self.device,
                **prediction,
            )
            if overlap:
                raise NotImplementedError("The overlap prediction is not implemented for e3tb method.")

        else:
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction["method"]))

        
        if self.method == "sktb":
            self.hamiltonian = SKHamiltonian(
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                idp_sk=self.idp, 
                dtype=self.dtype, 
                device=self.device,
                onsite=True,
                )
            if overlap:
                self.overlap = SKHamiltonian(
                    idp_sk=self.idp, 
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
                    dtype=self.dtype, 
                    device=self.device,
                    onsite=False,
                    )

        elif self.method == "e3tb":
            self.hamiltonian = E3Hamiltonian(
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                idp=self.idp, 
                dtype=self.dtype, 
                device=self.device
                )
            if overlap:
                self.overlap = E3Hamiltonian(
                    idp=self.idp, 
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
                    dtype=self.dtype, 
                    device=self.device,
                    overlap=True,
                    )


    def forward(self, data: AtomicDataDict.Type):

        data = self.embedding(data)
        if self.overlap:
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        
        data = self.node_prediction_h(data)
        data = self.edge_prediction_h(data)
        if self.overlap:
            data = self.edge_prediction_s(data)
        
        if self.transform:
            data = self.hamiltonian(data)
            if self.overlap:
                data = self.overlap(data)

        return data
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint, 
        embedding: dict={},
        prediction: dict={},
        overlap: bool=None,
        basis: Dict[str, Union[str, list]]=None,
        dtype: Union[str, torch.dtype]=None,
        device: Union[str, torch.device]=None,
        transform: bool = True,
        **kwargs
        ):

        ckpt = torch.load(checkpoint)
        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        model_options = {
            "embedding": embedding,
            "prediction": prediction,
        }

        if len(embedding) == 0 or len(prediction) == 0:
            model_options.update(ckpt["config"]["model_options"])

        for k,v in common_options.items():
            if v is None:
                common_options[k] = ckpt["config"]["common_options"][k]
        
        model = cls(**model_options, **common_options, transform=transform)
        model.load_state_dict(ckpt["model_state_dict"])

        del ckpt

        return model

class MIX(nn.Module):
    name = "mix"
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            nnsk: dict,
            basis: Dict[str, Union[str, list]]=None,
            overlap: bool = False,
            idp_sk: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(MIX, self).__init__()

        self.dtype = dtype
        self.device = device

        self.dptb = DPTB(
            embedding=embedding, 
            prediction=prediction, 
            basis=basis, 
            idp=idp_sk,
            overlap=overlap, 
            dtype=dtype, 
            device=device,
            transform=False,
            )
        
        self.nnsk = NNSK(
            basis=basis,
            idp_sk=idp_sk, 
            **nnsk,
            overlap=overlap,
            dtype=dtype, 
            device=device,
            transform=False,
            )
        
        self.model_options = self.nnsk.model_options
        self.model_options.update(self.dptb.model_options)
        
        self.hamiltonian = self.nnsk.hamiltonian
        if overlap:
            self.overlap = self.nnsk.overlap


    def forward(self, data: AtomicDataDict.Type):
        data_dptb = self.dptb(data)
        data_nnsk = self.nnsk(data)

        data_nnsk[AtomicDataDict.EDGE_FEATURES_KEY] = data_nnsk[AtomicDataDict.EDGE_FEATURES_KEY] * (1 + data_dptb[AtomicDataDict.EDGE_FEATURES_KEY])
        data_nnsk[AtomicDataDict.NODE_FEATURES_KEY] = data_nnsk[AtomicDataDict.NODE_FEATURES_KEY] * (1 + data_dptb[AtomicDataDict.NODE_FEATURES_KEY])

        data_nnsk = self.hamiltonian(data_nnsk)
        if hasattr(self, "overlap"):
            data_nnsk = self.overlap(data_nnsk)

        return data_nnsk
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint, 
        embedding: dict=None,
        prediction: dict=None,
        nnsk: dict=None,
        basis: Dict[str, Union[str, list]]=None,
        overlap: bool = None,
        dtype: Union[str, torch.dtype] = None,
        device: Union[str, torch.device] = None,
        ):
        # the mapping from the parameters of the ref_model and the current model can be found using
        # reference model's idp and current idp
        
        ckpt = torch.load(checkpoint)
        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }
        model_options = {
            "embedding": embedding,
            "prediction": prediction,
            "nnsk": nnsk,
        }
        
        if len(nnsk) == 0:
            model_options["nnsk"] = ckpt["config"]["model_options"]["nnsk"]

        if len(embedding) == 0 or len(prediction) == 0:
            assert ckpt["config"]["model_options"].get("embedding") is not None and ckpt["config"]["model_options"].get("prediction") is not None, \
            "The reference model checkpoint should come from a mixed model if dptb info is not provided."

            model_options["embedding"] = ckpt["config"]["model_options"]["embedding"]
            model_options["prediction"] = ckpt["config"]["model_options"]["prediction"]

        for k,v in common_options.items():
            if v is None:
                common_options[k] = ckpt["config"]["common_options"][k]

        if ckpt["config"]["model_options"].get("embedding") is not None and ckpt["config"]["model_options"].get("prediction") is not None:
            # read from mixed model
            model = cls(**model_options, **common_options)
            model.load_state_dict(ckpt["model_state_dict"])

        else:
            assert ckpt["config"]["model_options"].get("nnsk") is not None, "The referenced checkpoint should provide at least the nnsk model info."
            # read from nnsk model

            model = cls(**model_options, **common_options)
            model.nnsk.load_state_dict(ckpt["model_state_dict"])
        
        del ckpt
        
        return model