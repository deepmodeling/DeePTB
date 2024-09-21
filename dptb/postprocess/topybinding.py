from ase.io import read
import numpy as np
from matplotlib import pyplot as plt
import logging
import torch
import os
from typing import Optional, Union
from dptb.data import AtomicData, AtomicDataDict
import ase
from dptb.data.interfaces.ham_to_feature import feature_to_block

log = logging.getLogger(__name__)

Ang2nm = 0.1

try:
    import pybinding as pb
except ImportError:
    log.error("Pybinding is not installed. Please install it via `pip install pybinding`")

class ToPybinding(object):
    def __init__ (
            self, 
            model: torch.nn.Module,
            results_path: Optional[str]=None,
            use_gui=False,
            overlap=False,
            device: Union[str, torch.device]=torch.device('cpu')
            ):
        
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device    
        self.model = model
        self.model.eval()
        self.use_gui = use_gui
        self.results_path = results_path
        self.overlap = overlap


    def get_lattice(self, data: Union[AtomicData, ase.Atoms, str], AtomicData_options: dict={}, e_fermi: float=0.0):
        # get the AtomicData structure and the ase structure
        if isinstance(data, str):
            structase = read(data)
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, ase.Atoms):
            structase = data
            data = AtomicData.from_ase(structase, **AtomicData_options)
        elif isinstance(data, AtomicData):
            structase = data.to_ase()
            data = data
        
        data = AtomicData.to_AtomicDataDict(data.to(self.device))
        data = self.model.idp(data)
        
        # get the HR
        data = self.model(data)

        if self.overlap == True:
            log.error("Overlap is not supported in pybinding")
            raise NotImplementedError
        
        cell = data[AtomicDataDict.CELL_KEY].cpu().numpy()*Ang2nm
        positions = data[AtomicDataDict.POSITIONS_KEY]
        blocks =  feature_to_block(data, idp = self.model.idp) 
        lattice = pb.Lattice(a1=cell[0], a2=cell[1], a3=cell[2])

        onsite_bonds = []
        hop_bonds = []
        for ibond in blocks.keys():
            ijRs = [int(ii) for ii in ibond.split("_")]
            # onsites:
            if ijRs[2:] == [0,0,0] and ijRs[0] == ijRs[1]:
                onsite_bonds.append(ibond)
            else:
                hop_bonds.append(ibond)

        onsite_blocks= tuple(map(lambda x:
                        (x.split("_")[0], 
                        data['pos'].cpu().numpy()[int(x.split("_")[0])] * Ang2nm,
                        blocks[x].detach().cpu().numpy() - e_fermi * np.eye(blocks[x].shape[0])),
                        onsite_bonds))

        hop_blocks = tuple(map(lambda x: 
                               (x.split("_")[2:],
                               str(x.split("_")[0]),
                               str(x.split("_")[1]), 
                               blocks[x].detach().cpu().numpy()),
                               hop_bonds))

        lattice.add_sublattices(*onsite_blocks)
        lattice.add_hoppings(*hop_blocks)

        return lattice