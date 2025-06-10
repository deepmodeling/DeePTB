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
from dptb.nn.dftbsk import DFTBSK
from e3nn.o3 import Linear
from dptb.nn.rescale import E3PerSpeciesScaleShift, E3PerEdgeSpeciesScaleShift
import logging

log = logging.getLogger(__name__)

""" if this class is called, it suggest user choose a embedding method. If not, it should directly use _sktb.py
"""

def get_neuron_config(nl):
    """Extracts the configuration of a neural network from a list of layer sizes.

    Args:
        nl: A list of integers representing the number of neurons in each layer.
            If the list has an even number of elements, the last element is assumed
            to be the output layer size.

    Returns:
        A list of dictionaries, where each dictionary describes the configuration of
        a layer in the neural network. Each dictionary has the following keys:
        - in_features: The number of input neurons for the layer.
        - hidden_features: The number of hidden neurons for the layer (if applicable).
        - out_features: The number of output neurons for the layer.
    
        e.g.
        [1, 2, 3, 4, 5, 6] -> [{'in_features': 1, 'hidden_features': 2, 'out_features': 3}, 
                               {'in_features': 3, 'hidden_features': 4, 'out_features': 5}, 
                               {'in_features': 5, 'out_features': 6}]
        [1, 2, 3, 4, 5]    -> [{'in_features': 1, 'hidden_features': 2, 'out_features': 3}, 
                               {'in_features': 3, 'hidden_features': 4, 'out_features': 5}]
    """
    
    n = len(nl)
    assert n > 1, "The neuron config should have at least 2 layers."
    if n % 2 == 0:
        d_out = nl[-1]
        nl = nl[:-1]
    config = []
    for i in range(1,len(nl)-1, 2):
        config.append({'in_features': nl[i-1], 'hidden_features': nl[i], 'out_features': nl[i+1]})

    if n % 2 == 0:
        config.append({'in_features': nl[-1], 'out_features': d_out})

    return config

class NNENV(nn.Module):
    quantities = ["hamiltonian", "energy"]
    name = "nnenv"
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
        super(NNENV, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.dtype = dtype
        self.device = device
        self.model_options = {"embedding": embedding.copy(), "prediction": prediction.copy()}
        self.transform = transform
        
        
        self.method = prediction.get("method", "e3tb")
        # self.soc = prediction.get("soc", False)
        self.prediction = prediction

        prediction_copy = prediction.copy()

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
            prediction_copy["neurons"] = [self.embedding.out_node_dim] + prediction_copy["neurons"] + [self.idp.n_onsite_Es]
            prediction_copy["config"] = get_neuron_config(prediction_copy["neurons"])

            self.node_prediction_h = AtomicResNet(
                **prediction_copy,
                in_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            prediction_copy["neurons"][0] = self.embedding.out_edge_dim
            prediction_copy["neurons"][-1] = self.idp.reduced_matrix_element
            prediction_copy["config"] = get_neuron_config(prediction_copy["neurons"])
            self.edge_prediction_h = AtomicResNet(
                **prediction_copy,
                in_field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            if overlap:
                self.idp.get_skonsite_maps()
                self.idp_sk = self.idp
                self.edge_prediction_s = AtomicResNet(
                    **prediction_copy,
                    in_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    out_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    device=device, 
                    dtype=dtype
                )

                overlaponsite_param = torch.ones([len(self.idp.type_names), self.idp.n_onsite_Es, 1], dtype=self.dtype, device=self.device)
                if not all(self.idp.mask_diag):
                    self.overlaponsite_param = torch.nn.Parameter(overlaponsite_param)
                else:
                    self.overlaponsite_param = overlaponsite_param

        elif prediction_copy.get("method") == "e3tb":
            self.node_prediction_h = E3PerSpeciesScaleShift(
                field=AtomicDataDict.NODE_FEATURES_KEY,
                num_types=n_species,
                irreps_in=self.embedding.out_node_irreps,
                out_field = AtomicDataDict.NODE_FEATURES_KEY,
                shifts=0.,
                scales=1.,
                dtype=self.dtype,
                device=self.device,
                **prediction_copy,
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
                **prediction_copy,
            )

            if overlap:
                self.idp_sk = OrbitalMapper(self.idp.basis, method="sktb", device=self.device)
                self.idp_sk.get_skonsite_maps()
                prediction_copy["neurons"] = [self.embedding.latent_dim] + prediction_copy["neurons"] + [self.idp_sk.reduced_matrix_element]
                prediction_copy["config"] = get_neuron_config(prediction_copy["neurons"])
                self.edge_prediction_s = AtomicResNet(
                    **prediction_copy,
                    in_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    out_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    device=device,
                    dtype=dtype
                )

                overlaponsite_param = torch.ones([len(self.idp_sk.type_names), self.idp_sk.n_onsite_Es, 1], dtype=self.dtype, device=self.device)
                if not all(self.idp_sk.mask_diag):
                    self.overlaponsite_param = torch.nn.Parameter(overlaponsite_param)
                else:
                    self.overlaponsite_param = overlaponsite_param

                
                # raise NotImplementedError("The overlap prediction is not implemented for e3tb method.")

        else:
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction_copy["method"]))

        
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
                    onsite=True,
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
                self.overlap = SKHamiltonian(
                    idp_sk=self.idp_sk, 
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    onsite=True,
                    strain=False,
                    soc=False,
                    dtype=self.dtype, 
                    device=self.device,
                    )


    def forward(self, data: AtomicDataDict.Type):
        if data.get(AtomicDataDict.EDGE_TYPE_KEY, None) is None:
            self.idp(data)

        data = self.embedding(data)
        if hasattr(self, "overlap") and self.method == "sktb":
            data[AtomicDataDict.EDGE_OVERLAP_KEY] = data[AtomicDataDict.EDGE_FEATURES_KEY]
        
        data = self.node_prediction_h(data)
        data = self.edge_prediction_h(data)
        if hasattr(self, "overlap"):
            data = self.edge_prediction_s(data)
            data[AtomicDataDict.NODE_OVERLAP_KEY] = self.overlaponsite_param[data[AtomicDataDict.ATOM_TYPE_KEY].flatten()]
            data[AtomicDataDict.NODE_OVERLAP_KEY][:,self.idp_sk.mask_diag] = 1.
        
        if self.transform:
            data = self.hamiltonian(data)
            if hasattr(self, "overlap"):
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
        if device == 'cuda':
            if not torch.cuda.is_available():
                device = 'cpu'
                log.warning("CUDA is not available. The model will be loaded on CPU.")

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
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
            nnsk: dict = None,
            dftbsk: dict = None,
            basis: Dict[str, Union[str, list]]=None,
            overlap: bool = False,
            idp_sk: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
            transform: bool = True,
            num_xgrid: int = -1,
            **kwargs,
    ):
        super(MIX, self).__init__()

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        self.dtype = dtype
        self.device = device
        self.transform = transform
        self.basis = basis
        
        self.nnenv = NNENV(
            embedding=embedding, 
            prediction=prediction, 
            basis=basis, 
            idp=idp_sk,
            overlap=overlap, 
            dtype=dtype, 
            device=device,
            transform=False,
            )
        
        if (dftbsk is None) == (nnsk is None):
            raise ValueError("The mixed model should have exactly one of the dftbsk model or nnsk model.")

        if nnsk is not None:
            self.nnsk = NNSK(
                basis=basis,
                idp_sk=idp_sk, 
                **nnsk,
                overlap=overlap,
                dtype=dtype, 
                device=device,
                transform=False,
                )
            self.idp = self.nnsk.idp
            assert not self.nnsk.push, "The push option is not supported in the mixed model. The push option is only supported in the nnsk model."
        
            self.model_options = self.nnsk.model_options
            self.hamiltonian = self.nnsk.hamiltonian
            if overlap:
                self.overlap = self.nnsk.overlap
        elif dftbsk is not None:
            self.dftbsk = DFTBSK(
                basis=basis,
                idp_sk=idp_sk,
                **dftbsk,
                overlap=overlap,
                dtype=dtype,
                device=device,
                transform=False,
                num_xgrid=num_xgrid,
                )
            self.idp = self.dftbsk.idp
            self.model_options = self.dftbsk.model_options
            self.hamiltonian = self.dftbsk.hamiltonian
            if overlap:
                self.overlap = self.dftbsk.overlap
        else:
            raise ValueError("The mixed model should have exactly one of the dftbsk model or nnsk model.")

        self.model_options.update(self.nnenv.model_options)
        

    def forward(self, data: AtomicDataDict.Type):

        if data.get(AtomicDataDict.EDGE_TYPE_KEY, None) is None:
            self.idp(data)
            
        data_nnenv = self.nnenv(data)
        if hasattr(self, "nnsk"):
            data_sk = self.nnsk(data)
        elif hasattr(self, "dftbsk"):
            data_sk = self.dftbsk(data)
        else:
            raise ValueError("The mixed model should have exactly one of the dftbsk model or nnsk model.")

        data_sk[AtomicDataDict.EDGE_FEATURES_KEY] = data_sk[AtomicDataDict.EDGE_FEATURES_KEY] * (1 + data_nnenv[AtomicDataDict.EDGE_FEATURES_KEY])
        data_sk[AtomicDataDict.NODE_FEATURES_KEY] = data_sk[AtomicDataDict.NODE_FEATURES_KEY] * (1 + data_nnenv[AtomicDataDict.NODE_FEATURES_KEY])

        if self.transform:
            data_sk = self.hamiltonian(data_sk)
            if hasattr(self, "overlap"):
                data_sk = self.overlap(data_sk)

        return data_sk
    
    @classmethod
    def from_reference(
        cls, 
        checkpoint, 
        embedding: dict=None,
        prediction: dict=None,
        nnsk: dict=None,
        dftbsk: dict = None,
        basis: Dict[str, Union[str, list]]=None,
        overlap: bool = None,
        dtype: Union[str, torch.dtype] = None,
        device: Union[str, torch.device] = None,
        transform: bool = True,
        **kwargs,
        ):
        # the mapping from the parameters of the ref_model and the current model can be found using
        # reference model's idp and current idp
        if device == 'cuda':
            if not torch.cuda.is_available():
                device = 'cpu'
                log.warning("CUDA is not available. The model will be loaded on CPU.")

        ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
        common_options = {
            "dtype": dtype,
            "device": device,
            "basis": basis,
            "overlap": overlap,
        }

        if ckpt["config"]["model_options"].get("nnsk",None) is not None:
            if nnsk is None or len(nnsk) == 0:
                nnsk = ckpt["config"]["model_options"]["nnsk"]
        if ckpt["config"]["model_options"].get("dftbsk",None) is not None:
            if dftbsk is None or len(dftbsk) == 0:
                dftbsk = ckpt["config"]["model_options"]["dftbsk"]

        if (dftbsk is None) == (nnsk is None):
            raise ValueError("The mixed model should have exactly one of the dftbsk model or nnsk model.")

        model_options = {
            "embedding": embedding,
            "prediction": prediction,
            "nnsk": nnsk,
            "dftbsk": dftbsk
        }

        if nnsk is not None:
            if model_options["nnsk"].get("push") is not None:
                model_options["nnsk"]["push"] = None
                log.warning("The push option is not supported in the mixed model. The push option is only supported in the nnsk model.")

        if len(embedding) == 0 or len(prediction) == 0:
            assert ckpt["config"]["model_options"].get("embedding") is not None and ckpt["config"]["model_options"].get("prediction") is not None, \
            "The reference model checkpoint should come from a mixed model if dptb info is not provided."

            model_options["embedding"] = ckpt["config"]["model_options"]["embedding"]
            model_options["prediction"] = ckpt["config"]["model_options"]["prediction"]

        for k,v in common_options.items():
            if v is None:
                common_options[k] = ckpt["config"]["common_options"][k]

        if nnsk is not None:
            assert ckpt["config"]["model_options"].get("nnsk") is not None, "The referenced checkpoint should provide at least the nnsk model info."
            
            if ckpt["config"]["model_options"].get("embedding") is not None and ckpt["config"]["model_options"].get("prediction") is not None:
                # read from mixed model
                model = cls(**model_options, **common_options,transform=transform)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                # read from nnsk model
                model = cls(**model_options, **common_options,transform=transform)
                model.nnsk.load_state_dict(ckpt["model_state_dict"])
        else:
            assert ckpt["config"]["model_options"].get("dftbsk") is not None, "The referenced checkpoint should provide at least the dftbsk model info."
            
            if ckpt["config"]["model_options"].get("embedding") is not None and ckpt["config"]["model_options"].get("prediction") is not None:    
                num_xgrid = ckpt["model_state_dict"]["dftbsk.distance_param"].shape[0]
                model = cls(**model_options, **common_options,transform=transform,num_xgrid=num_xgrid)
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                num_xgrid = ckpt["model_state_dict"]["distance_param"].shape[0]
                model = cls(**model_options, **common_options,transform=transform,num_xgrid=num_xgrid)
                model.dftbsk.load_state_dict(ckpt["model_state_dict"])

        del ckpt
        
        return model
