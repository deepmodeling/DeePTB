import torch.nn as nn
import torch
from typing import Union, Tuple, Optional, Callable, Dict
import torch.nn.functional as F
from dptb.nn.embedding import Embedding
from dptb.data.transforms import OrbitalMapper
from dptb.nn.base import AtomicFFN, AtomicResNet, AtomicLinear
from dptb.data import AtomicDataDict
from dptb.nn.hamiltonian import E3Hamiltonian, SKHamiltonian
from dptb.nn.nnsk import NNSK


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
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
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
        self.model_options = {"embedding": embedding, "prediction": prediction}
        
        self.method = prediction["hamiltonian"].get("method", "e3tb")
        self.overlap = prediction["hamiltonian"].get("overlap", False)
        self.soc = prediction["hamiltonian"].get("soc", False)
        self.prediction = prediction

        if basis is not None:
            self.idp = OrbitalMapper(basis, method=self.method, device=self.device)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_node_maps()
        self.idp.get_pair_maps()


        # initialize the embedding layer
        self.embedding = Embedding(**embedding, dtype=dtype, device=device, idp=self.idp, n_atom=len(self.basis.keys()))
        
        # initialize the prediction layer
        if prediction.get("method") == "linear":
            
            self.node_prediction_h = AtomicLinear(
                in_features=self.embedding.out_node_dim, 
                out_features=self.idp.node_reduced_matrix_element, 
                in_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                dtype=dtype,
                device=device
                )
            self.edge_prediction_h = AtomicLinear(
                in_features=self.embedding.out_edge_dim, 
                out_features=self.idp.edge_reduced_matrix_element, 
                in_field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                dtype=dtype,
                device=device
                )
            
            if self.overlap:
                self.node_prediction_s = AtomicLinear(
                    in_features=self.embedding.out_node_dim, 
                    out_features=self.idp.node_reduced_matrix_element, 
                    in_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    out_field=AtomicDataDict.NODE_OVERLAP_KEY,
                    dtype=dtype,
                    device=device
                    )
                self.edge_prediction_s = AtomicLinear(
                    in_features=self.embedding.out_edge_dim, 
                    out_features=self.idp.edge_reduced_matrix_element, 
                    in_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    out_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    dtype=dtype,
                    device=device
                    )
            
        elif prediction.get("method") == "nn":
            prediction["neurons"] = [self.embedding.out_node_dim] + prediction["neurons"] + [self.idp.node_reduced_matrix_element]
            prediction["config"] = get_neuron_config(prediction["neurons"])

            self.node_prediction_h = AtomicResNet(
                **prediction,
                in_field=AtomicDataDict.NODE_FEATURES_KEY,
                out_field=AtomicDataDict.NODE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            prediction["neurons"][0] = self.embedding.out_edge_dim
            prediction["neurons"][-1] = self.idp.edge_reduced_matrix_element
            prediction["config"] = get_neuron_config(prediction["neurons"])
            self.edge_prediction_h = AtomicResNet(
                **prediction,
                in_field=AtomicDataDict.EDGE_FEATURES_KEY,
                out_field=AtomicDataDict.EDGE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

            if self.overlap:
                self.edge_prediction_s = AtomicResNet(
                    **prediction,
                    in_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    out_field=AtomicDataDict.EDGE_OVERLAP_KEY,
                    device=device, 
                    dtype=dtype
                )
        elif prediction.get("method") == "none":
            pass
        else:
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction["method"]))

        
        # initialize the hamiltonian layer
        if self.method == "sktb":
            self.hamiltonian = SKHamiltonian(
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                idp=self.idp, 
                dtype=self.dtype, 
                device=self.device
                )
        elif self.method == "e3tb":
            self.hamiltonian = E3Hamiltonian(
                edge_field=AtomicDataDict.EDGE_FEATURES_KEY,
                node_field=AtomicDataDict.NODE_FEATURES_KEY,
                idp=self.idp, 
                dtype=self.dtype, 
                device=self.device
                )

        if self.overlap:
            if self.method == "sktb":
                self.overlap = SKHamiltonian(
                    idp=self.idp, 
                    edge_field=AtomicDataDict.EDGE_OVERLAP_KEY, 
                    node_field=AtomicDataDict.NODE_OVERLAP_KEY, 
                    dtype=self.dtype, 
                    device=self.device,
                    overlap=True,
                    )
            elif self.method == "e3tb":
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
        
        if not self.prediction.get("method") == "none":
            data = self.node_prediction_h(data)
            data = self.edge_prediction_h(data)
        
        # data = self.hamiltonian(data)

        if self.overlap:
            data = self.edge_prediction_s(data)
            data = self.overlap(data)

        return data
    
    @classmethod
    def from_reference(cls, checkpoint):
        
        ckpt = torch.load(checkpoint)
        model = cls(**ckpt["config"]["model_options"], **ckpt["config"]["mode_config"], **ckpt["idp"])
        model.load_state_dict(ckpt["model_state_dict"])

        return model
        

class MIX(nn.Module):
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            nnsk: dict,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self.dptb = DPTB(embedding, prediction, basis, idp, dtype, device)
        self.nnsk = NNSK(basis, idp, **nnsk, dtype=dtype, device=device)


    def forward(self, data: AtomicDataDict.Type):
        data_dptb = self.dptb(data)
        data_nnsk = self.nnsk(data)

        return data
    
    @classmethod
    def from_reference(cls, checkpoint, nnsk_options: Dict=None):
        # the mapping from the parameters of the ref_model and the current model can be found using
        # reference model's idp and current idp
        pass