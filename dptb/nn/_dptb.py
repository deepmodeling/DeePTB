import torch.nn as nn
import torch
from typing import Union, Tuple, Optional, Callable
import torch.nn.functional as F
from .embedding import Embedding
from dptb.utils.index_mapping import Index_Mapings_e3
from ._base import AtomicFFN, AtomicResNet, AtomicLinear
from dptb.data import AtomicDataDict
from torch import Tensor
from dptb.utils.tools import get_neuron_config


""" if this class is called, it suggest user choose a embedding method. If not, it should directly use _sktb.py
"""

def get_neuron_config(nl):
    n = len(nl)
    if n % 2 == 0:
        d_out = nl[-1]
        nl = nl[:-1]
    config = []
    for i in range(1,len(nl)-1, 2):
        config.append({'n_in': nl[i-1], 'n_hidden': nl[i], 'n_out': nl[i+1]})

    if n % 2 == 0:
        config.append({'n_in': nl[-1], 'n_out': d_out})

    return config


class dptb(nn.Module):
    def __init__(
            self,
            basis,
            embedding_config: dict,
            prediction_config: dict,
            method: str = "e3tb",
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(dptb, self).__init__()

        self.embedding = Embedding(**embedding_config)
        self.idp = Index_Mapings_e3(basis, method=method)
        self.idp.get_node_maps()
        self.idp.get_pair_maps()

        self.method = method

        # computing the in_feature and out_feature
        if prediction_config["mode"] == "linear":
            
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
        else:

            prediction_config["neurons"] = [self.embedding.out_node_dim] + prediction_config["neurons"] + [self.idp.node_reduced_matrix_element]
            prediction_config["config"] = get_neuron_config(prediction_config["neurons"])
            self.node_prediction = AtomicResNet(
                **prediction_config,
                field=AtomicDataDict.NODE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )
            prediction_config["neurons"][0] = [self.embedding.out_edge_dim]
            prediction_config["config"] = get_neuron_config(prediction_config["neurons"])
            self.edge_prediction = AtomicResNet(
                **prediction_config,
                field=AtomicDataDict.EDGE_FEATURES_KEY,
                device=device, 
                dtype=dtype
            )

    def forward(self, data: AtomicDataDict.Type):
        data = self.embedding(data)
        data = self.node_prediction(data)
        data = self.edge_prediction(data)
        return data
        