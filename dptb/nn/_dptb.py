import torch.nn as nn
import torch
from typing import Union, Tuple, Optional
import torch.nn.functional as F
from .embedding import Embedding
from dptb.utils.index_mapping import Index_Mapings_e3
from ._base import AtomicFFN, AtomicResNet, AtomicLinear
from dptb.data import AtomicDataDict


""" if this class is called, it suggest user choose a embedding method. If not, it should directly use _sktb.py
"""
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
            self.node_prediction = AtomicResNet(
            )
            self.edge_prediction = nn.Linear(self.embedding.out_edge_dim, self.idp.edge_reduced_matrix_element)
        