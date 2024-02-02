import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding

@Embedding.register("none")
class Identity(torch.nn.Module):
    def __init__(
        self,
        dtype: Union[str, torch.dtype] = torch.float32,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        super(Identity, self).__init__(Identity, dtype, device)

    def forward(self, data: AtomicDataDict) -> AtomicDataDict:
        return data
    
    @property
    def out_edge_dim(self):
        return 0

    @property
    def out_note_dim(self):
        return 0