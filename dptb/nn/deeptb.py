import torch.nn as nn
import torch
from typing import Union, Tuple, Optional, Callable, Dict
import torch.nn.functional as F
from dptb.nn.embedding import Embedding
from dptb.data.transforms import OrbitalMapper
from dptb.nn.base import AtomicFFN, AtomicResNet, AtomicLinear
from dptb.data import AtomicDataDict
from dptb.nn.hamiltonian import E3Hamiltonian, SKHamiltonian


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
    def __init__(
            self,
            embedding: dict,
            prediction: dict,
            basis: Dict[str, Union[str, list]]=None,
            idp: Union[OrbitalMapper, None]=None,
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
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
        
        # initialize the embedding layer
        self.embedding = Embedding(**embedding, dtype=dtype, device=device)


        # initialize the prediction layer
        method = prediction["hamiltonian"].get("method", "e3tb")

        if basis is not None:
            self.idp = OrbitalMapper(basis, method=method)
            if idp is not None:
                assert idp == self.idp, "The basis of idp and basis should be the same."
        else:
            assert idp is not None, "Either basis or idp should be provided."
            self.idp = idp
            
        self.basis = self.idp.basis
        self.idp.get_node_maps()
        self.idp.get_pair_maps()
        
        self.method = method
        if prediction["method"] == "linear":
            
            self.node_prediction = AtomicLinear(
                in_features=self.embedding.out_node_dim, 
                out_features=self.idp.node_reduced_matrix_element, 
                field=AtomicDataDict.NODE_FEATURES_KEY,
                dtype=dtype,
                device=device
                )
            self.edge_prediction = AtomicLinear(
                in_features=self.embedding.out_edge_dim, 
                out_features=self.idp.edge_reduced_matrix_element, 
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                dtype=dtype,
                device=device
                )
            
        elif prediction["method"] == "nn":
            prediction["neurons"] = [self.embedding.out_node_dim] + prediction["neurons"] + [self.idp.node_reduced_matrix_element]
            prediction["config"] = get_neuron_config(prediction["neurons"])

            self.node_prediction = AtomicResNet(
                **prediction,
                field=AtomicDataDict.NODE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )
            prediction["neurons"][0] = self.embedding.out_edge_dim
            prediction["neurons"][-1] = self.idp.edge_reduced_matrix_element
            prediction["config"] = get_neuron_config(prediction["neurons"])
            self.edge_prediction = AtomicResNet(
                **prediction,
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

        else:
            raise NotImplementedError("The prediction model {} is not implemented.".format(prediction["method"]))

        # initialize the hamiltonian layer
        if method == "sktb":
            self.hamiltonian = SKHamiltonian(idp=self.idp, dtype=self.dtype, device=self.device)
        elif method == "e3tb":
            self.hamiltonian = E3Hamiltonian(idp=self.idp, dtype=self.dtype, device=self.device)
        


    def forward(self, data: AtomicDataDict.Type):

        data = self.embedding(data)
        data = self.node_prediction(data)
        data = self.edge_prediction(data)
        data = self.hamiltonian(data)

        return data
        