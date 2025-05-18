import ase
from ase.io import read,write
from ase.atoms import Atoms
try:
    import seekpath as skp
except ImportError:
    raise ImportError("seekpath is not installed. Please install it to use this feature.")
import numpy as np
from typing import Union
import logging
from dptb.utils.constants import atomic_num_dict_r


log = logging.getLogger(__name__)



def auto_band_config(structure: Union[str, ase.Atoms], kpathtype='vasp'):
    """
    Automatically generate band.json file for DPTB.
    Args:
        structure (str or ase.Atoms): The input structure, either a file path or an ase.Atoms object.
        kpathtype (str): The type of kpath to generate. Currently only 'vasp' is supported.
    Returns:
        dict: A dictionary containing the band.json data.
    """
    if isinstance(structure, str):
        asestruct = read(structure)
    elif isinstance(structure, ase.Atoms):
        asestruct = structure
    else:
        raise TypeError("structure must be a string (file path) or an ase.Atoms object.")
    skpstructure = (asestruct.cell.tolist(),
                    asestruct.get_scaled_positions().tolist(),
                    asestruct.get_atomic_numbers().tolist())
    # skpdata = skp.get_path(skpstructure)
    skpdata = skp.get_path_orig_cell(skpstructure)
    log.info("The structure space group is: %s (No. %d)"  %(skpdata['spacegroup_international'], skpdata['spacegroup_number']))
    #log.info("The structure space group number is: %d", skpdata['spacegroup_number'])

    #if not (skpdata['rotation_matrix'] == np.array([[1., 0., 0.], [0., 1., 0.],[0., 0., 1.]])).all() :
    #    log.warning("The input structure is not in standard primitive cell, will rotate it to standard primitive cell.")
    #    cell = skpdata['primitive_lattice']
    #    atomic_numbers = skpdata['primitive_types']
    #    atomic_positions = skpdata['primitive_positions']
    #    pri_structure = Atoms(cell=cell, numbers=atomic_numbers, scaled_positions=atomic_positions)
    #    write('standard_primitive_cell.vasp', pri_structure, format='vasp')
    
    high_sym_kpoints_dict = skpdata['point_coords']
    pathstr = ['-'.join(ipt) for ipt in skpdata['path']]

    assert kpathtype == 'vasp', "Only vasp kpath is supported now."
    bandjdata = {'task_options':{}}
    bandjdata['task_options']['kline_type'] = kpathtype
    bandjdata['task_options']['task'] = 'band'
    bandjdata['task_options']['kpath'] = pathstr
    bandjdata['task_options']['high_sym_kpoints'] = high_sym_kpoints_dict
    bandjdata['task_options']['number_in_line'] = 20

    atomsyb = [atomic_num_dict_r[i] for i in np.unique(asestruct.get_atomic_numbers())]
    basis = {}
    for isy in atomsyb:
        basis[isy] = ['s','p','d']
    common_options = {'basis': basis}

    return bandjdata, common_options

