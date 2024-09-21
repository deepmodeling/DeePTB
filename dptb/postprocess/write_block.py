import numpy as np
from dptb.utils.tools import j_must_have
from ase.io import read
import ase
from typing import Union
import matplotlib.pyplot as plt
import torch
from typing import Optional
import matplotlib
import logging
from dptb.data import AtomicData, AtomicDataDict
from dptb.data.interfaces.ham_to_feature import feature_to_block

log = logging.getLogger(__name__)

def write_block(
        data: Union[AtomicData, ase.Atoms, str], 
        model: torch.nn.Module,
        AtomicData_options: dict={},
        device: Union[str, torch.device]=None
        ):
    
    model.eval()
    if isinstance(device, str):
        device = torch.device(device)
    # get the AtomicData structure and the ase structure
    if isinstance(data, str):
        structase = read(data)
        data = AtomicData.from_ase(structase, **AtomicData_options)
    elif isinstance(data, ase.Atoms):
        structase = data
        data = AtomicData.from_ase(structase, **AtomicData_options)
    elif isinstance(data, AtomicData):
        structase = data.to("cpu").to_ase()
        data = data
    
    data = AtomicData.to_AtomicDataDict(data.to(device))
    with torch.no_grad():
        data = model.idp(data)

        # set the kpoint of the AtomicData
        data = model(data)
        block = feature_to_block(data=data, idp=model.idp)

    return block



    
