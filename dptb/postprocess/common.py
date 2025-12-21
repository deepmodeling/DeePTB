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

def load_data_for_model(
    data: Union[AtomicData, ase.Atoms, str],
    model: torch.nn.Module,
    device: Union[str, torch.device] = None,
    pbc: Union[bool, list] = None,
    AtomicData_options: dict = None,
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
        overlap_blocks = h5py.File(override_overlap, "r")
        if len(overlap_blocks) != 1:
            log.info('Overlap file contains more than one overlap matrix, only first will be used.')
        if overlap_flag:
            log.warning('override_overlap is enabled while model contains overlap, override_overlap will be used.')
            
        if "0" in overlap_blocks:
            overlaps = overlap_blocks["0"]
        else:
            overlaps = overlap_blocks["1"]
            
        block_to_feature(data_obj, model.idp, blocks=False, overlap_blocks=overlaps)
        overlap_blocks.close()
        
    # 8. Transfer to Device and Process
    # 8. Transfer to Device and Process
    # DEBUG: Check if cell exists before conversion
    print(f"DEBUG: load_data_for_model: data_obj type: {type(data_obj)}")
    if hasattr(data_obj, 'cell'):
        print(f"DEBUG: load_data_for_model: data_obj has cell. Shape: {data_obj.cell.shape}")
    else:
        print("DEBUG: load_data_for_model: data_obj MISSING cell attribute")
        
    try:
        keys_list = data_obj.keys
        # Check if keys is callable or property
        if callable(keys_list):
             keys_list = keys_list()
        print(f"DEBUG: load_data_for_model: data_obj keys: {keys_list}")
    except Exception as e:
        print(f"DEBUG: load_data_for_model: Failed to get keys: {e}")

    data_obj = AtomicData.to_AtomicDataDict(data_obj.to(device))
    print(f"DEBUG: load_data_for_model: resulting dict keys: {data_obj.keys()}")
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
