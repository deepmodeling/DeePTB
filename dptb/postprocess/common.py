import os
import h5py
import ase
import torch
import logging
from typing import Union, Optional
from copy import deepcopy
from ase.io import read

from dptb.data import AtomicData, AtomicDataDict, block_to_feature
from dptb.utils.argcheck import get_cutoffs_from_model_options

log = logging.getLogger(__name__)

def get_orbitals_for_type(orb_dict):
    """Build expanded orbital list from basis dictionary entry."""
    orb_list = []
    for o in orb_dict:
        if "s" in o:
            orb_list.append(o)
        elif "p" in o:
            orb_list.extend([o+"_y", o+"_z", o+"_x"])
        elif "d" in o:
            orb_list.extend([o+"_xy", o+"_yz", o+"_z2", o+"_xz", o+"_x2-y2"])
    return orb_list


def load_data_for_model(
     data: Union[AtomicData, ase.Atoms, str],
     model: torch.nn.Module,
     device: Optional[Union[str, torch.device]] = None,
     pbc: Optional[Union[bool, list]] = None,
     AtomicData_options: Optional[dict] = None,
     override_overlap: Optional[str] = None
 ) -> AtomicData:
    """
    Standardized helper to load and process data for post-processing with a DeePTB model.
    Handles defaults from model options (r_max), user overrides, and device transfer.
    """
    
    # 1. Determine device
    if device is None:
        device = model.device
    if isinstance(device, str):
        device = torch.device(device)
        
    # 2. Get default cutoffs from model
    r_max, er_max, oer_max = get_cutoffs_from_model_options(model.model_options)
    atomic_options = {'r_max': r_max, 'er_max': er_max, 'oer_max': oer_max}
    
    # 3. Handle PBC overrides
    if pbc is not None:
        # If pbc provided, override default (which usually comes from atoms object)
        atomic_options.update({'pbc': pbc})
        
    # 4. Handle AtomicData_options overrides with warnings
    if AtomicData_options is not None:
        for key in ['r_max', 'er_max', 'oer_max']:
            if AtomicData_options.get(key) is not None:
                if atomic_options[key] != AtomicData_options.get(key):
                    atomic_options[key] = AtomicData_options.get(key)
                    log.warning(f'Overwrite the {key} setting in the model with the {key} setting in the AtomicData_options: {AtomicData_options.get(key)}')
                    log.warning(f'This is very dangerous, please make sure you know what you are doing.')
                    
    # 5. Validation
    if atomic_options['r_max'] is None:
        log.error('The r_max is not provided in model_options, please provide it in AtomicData_options.')
        raise RuntimeError('The r_max is not provided in model_options, please provide it in AtomicData_options.')
        
    # 6. Load Data
    if isinstance(data, str):
        structase = read(data)
        data_obj = AtomicData.from_ase(structase, **atomic_options)
    elif isinstance(data, ase.Atoms):
        structase = data
        data_obj = AtomicData.from_ase(structase, **atomic_options)
    elif isinstance(data, AtomicData):
        log.info('The data is already an instance of AtomicData. Then the data is used directly.')
        data_obj = data
    else:
        raise ValueError('data should be either a string, ase.Atoms, or AtomicData')
        
    # 7. Handle Overlap Override
    overlap_flag = hasattr(model, 'overlap')
    
    if isinstance(override_overlap, str):
        assert os.path.exists(override_overlap), "Overlap file not found."
        with h5py.File(override_overlap, "r") as overlap_blocks:
            if len(overlap_blocks) != 1:
                log.info('Overlap file contains more than one overlap matrix, only first will be used.')
            if overlap_flag:
                log.warning('override_overlap is enabled while model contains overlap, override_overlap will be used.')
                
            if "0" in overlap_blocks:
                overlaps = overlap_blocks["0"]
            else:
                overlaps = overlap_blocks["1"]
                
            block_to_feature(data_obj, model.idp, blocks=False, overlap_blocks=overlaps)
        
    data_obj = AtomicData.to_AtomicDataDict(data_obj.to(device))
    data_obj = model.idp(data_obj)
    # Note: Some functions like get_data in ElecStruCal usually return idp(data) NOT model(data).
    # model(data) is typically called later to get eigenvalues.
    # HOWEVER, interfaces.py needs model(data) to be called to get H blocks!
    # ElecStruCal.get_data returns `model.idp(data)`.
    # Let's align with ElecStruCal for now, but `interfaces.py` will need to call model(data) manually 
    # OR we add a flag 'run_model=False'.
    
    # Actually, ElecStruCal.get_data does NOT run self.model(data). It runs self.model.idp(data).
    # self.get_eigs runs self.model(data).
    return data_obj
