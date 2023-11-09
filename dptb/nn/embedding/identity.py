# this is just a dumb class incase we don't want any embedding

import torch
from typing import Optional, Tuple, Union
from dptb.data import AtomicDataDict
from dptb.nn.embedding.emb import Embedding

@Embedding.register("identity")
class Identity(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Identity, self).__init__(**kwargs)

    def forward(data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        return data