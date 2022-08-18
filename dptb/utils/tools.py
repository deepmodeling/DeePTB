import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from dptb.utils.constants import atomic_num_dict, anglrMId
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)
import json
from pathlib import Path
import yaml
import torch.optim as optim
import logging

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    _DICT_VAL = TypeVar("_DICT_VAL")
    _OBJ = TypeVar("_OBJ")
    try:
        from typing import Literal  # python >3.6
    except ImportError:
        from typing_extensions import Literal  # type: ignore
    _ACTIVATION = Literal["relu", "relu6", "softplus", "sigmoid", "tanh", "gelu", "gelu_tf"]
    _PRECISION = Literal["default", "float16", "float32", "float64"]


def get_optimizer(opt_type: str, model_param, lr: float, **options: dict):
    if opt_type == 'Adam':
        optimizer = optim.Adam(params=model_param, lr=lr, **options)
    elif opt_type == 'SGD':
        optimizer = optim.SGD(params=model_param, lr=lr, **options)
    elif opt_type == 'RMSprop':
        optimizer = optim.RMSprop(params=model_param, lr=lr, **options)
    elif opt_type == 'LBFGS':
        optimizer = optim.LBFGS(params=model_param, lr=lr, **options)
    else:
        raise RuntimeError("Optimizer should be Adam/SGD/RMSprop, not {}".format(opt_type))
    return optimizer

def get_lr_scheduler(sch_type: str, optimizer: optim.Optimizer, **sch_options):
    if sch_type == 'Expo':
        schedular = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **sch_options)
    else:
        raise RuntimeError("Scheduler should be Expo/..., not {}".format(sch_type))

    return schedular

def j_must_have(
    jdata: Dict[str, "_DICT_VAL"], key: str, deprecated_key: List[str] = []
) -> "_DICT_VAL":
    """Assert that supplied dictionary conaines specified key.

    Returns
    -------
    _DICT_VAL
        value that was store unde supplied key

    Raises
    ------
    RuntimeError
        if the key is not present
    """
    if key not in jdata.keys():
        for ii in deprecated_key:
            if ii in jdata.keys():
                log.warning(f"the key {ii} is deprecated, please use {key} instead")
                return jdata[ii]
        else:
            raise RuntimeError(f"json database must provide key {key}")
    else:
        return jdata[key]

def _get_activation_fn(activation):
    if activation == "relu":
        return torch.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh

    raise RuntimeError("activation should be relu/gelu/tanh, not {}".format(activation))

def get_uniq_bond_type(proj_atom_type):
    uatomtype = get_uniq_symbol(proj_atom_type)
    out = []
    N = len(uatomtype)
    for aa in range(N):
        for bb in range(aa, N):
            bond_name = uatomtype[aa] + '-' + uatomtype[bb]
            out.append(bond_name)

    return out

def get_uniq_env_bond_type(proj_atom_type, atom_type):
    proj_atom_type = get_uniq_symbol(proj_atom_type)
    atom_type = get_uniq_symbol(atom_type)

    env_bond = []
    for ip in proj_atom_type:
        #id = atom_type.index(ip)
        #for idx in range(id, len(atom_type)):
        for jp in atom_type:
            bond_name = ip + '-' + jp
            env_bond.append(bond_name)

    return env_bond
    
def get_uniq_symbol(atomsymbols):
    '''>It takes a list of atomic symbols and returns a list of unique atomic symbols in the order of
    atomic number
    
    Parameters
    ----------
    atomsymbols
        a list of atomic symbols, e.g. ['C', 'C','H','H',...]
    
    Returns
    -------
        the unique atom types in the system, and the types are sorted descending order of atomic number.
    
    '''
    atomic_num_dict_r = dict(zip(atomic_num_dict.values(), atomic_num_dict.keys()))
    atom_num = []
    for it in atomsymbols:
        atom_num.append(atomic_num_dict[it])
    # uniq and sort.
    uniq_atom_num = sorted(np.unique(atom_num), reverse=True)
    # assert(len(uniq_atom_num) == len(atomsymbols))
    uniqatomtype = []
    for ia in uniq_atom_num:
        uniqatomtype.append(atomic_num_dict_r[ia])

    return uniqatomtype

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

def get_env_neuron_config(neuron_list):
    # neuron_list: a list of neuron number in each layer
    # return: a list of dict of neuron config
    neuron_list_all = [1] + list(neuron_list) 
    config = get_neuron_config(neuron_list_all)

    return config

def get_bond_neuron_config(neuron_list, bond_num_hops, bond_type, axis_neuron, env_out):
    # neuron_list: a list of neuron number in each layer
    # return: a list of dict of neuron config
    config = {}
    
    # env_i + env_j
    for ibondtype in bond_type:
        neuron_list_all = [axis_neuron * env_out + 1]  +  list(neuron_list) + [bond_num_hops[ibondtype]]
        config[ibondtype] = get_neuron_config(neuron_list_all)

    return config

def get_onsite_neuron_config(neuron_list,onsite_num, onsite_type, axis_neuron, env_out ):
    # neuron_list: a list of neuron number in each layer
    # return: a list of dict of neuron config
    config = {}
    
    for itype in onsite_type:
        neuron_list_all = [axis_neuron * env_out]  +  list(neuron_list) + [onsite_num[itype]]
        config[itype] = get_neuron_config(neuron_list_all)

    return config


def sortarr(refarr, tararr, dim1=1, dim2=1, axis=0,reverse=False):
    """ sort the target ndarray using the reference array

    inputs
    -----
    refarr: N * M1 reference array
    tararr: N * M2 target array
    dim1: 2-nd dimension  of reference array: ie : M1
    dim2: 2-nd dimension  of target array: ie : M2
    axis: the array is sorted according to the value of refarr[:,axis].

    return
    ------
    sortedarr: sorted array.
    """
    refarr = np.reshape(refarr, [-1, dim1])
    tararr = np.reshape(tararr, [-1, dim2])
    assert (len(refarr) == len(tararr))
    tmparr = np.concatenate([refarr, tararr], axis=1)
    sortedarr = np.asarray(sorted(tmparr, key=lambda s: s[axis]),reverse=reverse)
    return sortedarr[:, :]

def env_smoth(rr, rcut, rcut_smth):
    '''It takes a vector of distances, and returns a vector of the same length, where each element is the
    smoothed value of the corresponding element in the input vector
    
    Parameters
    ----------
    rr
        the distance between the two atoms
    rcut
        the cutoff radius for the environment
    rcut_smth
        the cutoff radius for the smoothing function
    
    Returns
    -------
        the smoothed inverse of the distance vector 
    
    '''
    srr = np.zeros_like(rr)
    eps = 1.0E-3
    assert ((rr - 0 > eps).all())
    rr_large = rr[rr >= rcut_smth]
    srr[rr < rcut_smth] = 1.0 / rr[rr < rcut_smth]
    srr[rr >= rcut_smth] = 1.0 / rr_large * (
                0.5 * np.cos(np.pi * (rr_large - rcut_smth) / (rcut - rcut_smth)) + 0.5)
    srr[rr > rcut] = 0.0
    return srr

def format_readline(line):
    '''In  SK files there are like 5*0 to represent the 5 0 zeros, 0 0 0 0 0. Here we replace the num * values 
    into the format of number of values.
    
    Parameters
    ----------
    line
        the line of text to be formatted
    
    Returns
    -------
        A list of strings.
    
    '''
    lsplit = re.split(',|;| +|\n|\t',line)
    lsplit = list(filter(None,lsplit))
    lstr = []
    for ii in range(len(lsplit)):
        strtmp = lsplit[ii]
        if re.search('\*',strtmp):
            strspt = re.split('\*|\n',strtmp)
            strspt = list(filter(None,strspt))
            strfull = int(strspt[0]) * [strspt[1]]
            lstr +=strfull
        else:
            lstr += [strtmp]
    return lstr

def j_loader(filename: Union[str, Path]) -> Dict[str, Any]:
    """Load yaml or json settings file.

    Parameters
    ----------
    filename : Union[str, Path]
        path to file

    Returns
    -------
    Dict[str, Any]
        loaded dictionary

    Raises
    ------
    TypeError
        if the supplied file is of unsupported type
    """
    filepath = Path(filename)
    if filepath.suffix.endswith("json"):
        with filepath.open() as fp:
            return json.load(fp)
    elif filepath.suffix.endswith(("yml", "yaml")):
        with filepath.open() as fp:
            return yaml.safe_load(fp)
    else:
        raise TypeError("config file must be json, or yaml/yml")

class Index_Mapings(object):
    def __init__(self, proj_atom_anglr_m=None):
        self.AnglrMID = anglrMId
        if  proj_atom_anglr_m is not None:
            self.update(proj_atom_anglr_m = proj_atom_anglr_m)

    def update(self, proj_atom_anglr_m):
        # bond and env type can get from stuct class.
        self.bondtype = get_uniq_symbol(list(proj_atom_anglr_m.keys()))
        # projected angular momentum. get from struct class.
        self.ProjAnglrM = proj_atom_anglr_m

    def Bond_Ind_Mapings(self):
        bond_index_map = {}
        bond_num_hops = {}
        for it in range(len(self.bondtype)):
            for jt in range(len(self.bondtype)):
                itype = self.bondtype[it]
                jtype = self.bondtype[jt]
                orbdict = {}
                ist = 0
                numhops = 0
                for ish in self.ProjAnglrM[itype]:
                    for jsh in self.ProjAnglrM[jtype]:
                        ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                        jshsymbol = ''.join(re.findall(r'[A-Za-z]',jsh))
                        ishid = self.AnglrMID[ishsymbol]
                        jshid = self.AnglrMID[jshsymbol]
                        if it == jt:
                            if  jsh + '-' + ish in orbdict.keys():
                                orbdict[ish + '-' + jsh] = orbdict[jsh + '-' + ish]
                                continue
                            else:
                                numhops += min(ishid, jshid) + 1
                                orbdict[ish + '-' + jsh] = np.arange(ist, ist + min(ishid, jshid) + 1).tolist()

                        elif it < jt:
                            numhops += min(ishid, jshid) + 1
                            orbdict[ish + '-' + jsh] = np.arange(ist, ist + min(ishid, jshid) + 1).tolist()
                        else:
                            numhops += min(ishid, jshid) + 1
                            orbdict[ish + '-' + jsh] = bond_index_map[jtype + '-' + itype][jsh +'-'+ ish]
                            continue

                        # orbdict[ish+jsh] = paralist
                        ist += min(ishid, jshid) + 1
                        # print (itype, jtype, ish+jsh, ishid, jshid,paralist)
                bond_index_map[itype + '-' + jtype] = orbdict
                bond_num_hops[itype + '-' + jtype] = numhops

        for key in bond_index_map.keys():
            print('# ' + key + ':', bond_num_hops[key], ' independent hoppings')
            print('## ', end='')
            ic = 1
            for key2 in bond_index_map[key]:

                print('' + key2 + ':', bond_index_map[key][key2], '   ', end='')
                if ic % 6 == 0:
                    print('\n## ', end='')
                ic += 1
            print()

        return bond_index_map, bond_num_hops
    
    def Onsite_Ind_Mapings(self):
        onsite_index_map = {}
        onsite_num = {}
        for it in range(len(self.bondtype)):
            itype = self.bondtype[it]
            orbdict = {}
            ist = 0
            numhops = 0
            for ish in self.ProjAnglrM[itype]:
                ishsymbol = ''.join(re.findall(r'[A-Za-z]',ish))
                ishid = self.AnglrMID[ishsymbol]
                orbdict[ish] = [ist]
                ist += 1
                numhops += 1
            onsite_index_map[itype] = orbdict
            onsite_num[itype] = numhops

        for key in onsite_index_map.keys():
            print('# ' + key + ':', onsite_index_map[key], ' independent onsite Es')
            print('## ', end='')
            ic = 1
            for key2 in onsite_index_map[key]:

                print('' + key2 + ':', onsite_index_map[key][key2], '   ', end='')
                if ic % 6 == 0:
                    print('\n## ', end='')
                ic += 1
            print()

        return onsite_index_map, onsite_num

def nnsk_correction(nn_onsiteEs, nn_hoppings, sk_onsiteEs, sk_hoppings, sk_onsiteSs=None, sk_overlaps=None):
    """Add the nn correction to SK parameters hoppings and onsite Es.
    Args:
        corr_strength (int, optional): correction strength for correction mode 2, Defaults to 1.
    Note: the overlaps are fixed on changed of SK parameters.
    """      
    assert len(nn_onsiteEs) == len(sk_onsiteEs)
    assert len(nn_hoppings) == len(sk_hoppings)  
    onsiteEs = []
    onsiteSs = []
    for ib in range(len(nn_onsiteEs)):
        sk_onsiteEs_ib = sk_onsiteEs[ib]
        sk_onsiteEs_ib.requires_grad = False

        onsiteEs.append(sk_onsiteEs_ib * (1 + nn_onsiteEs[ib]))

        if sk_onsiteSs:
            sk_onsiteSs_ib = sk_onsiteSs[ib]
            sk_onsiteSs_ib.requires_grad = False
            # no correction to overlap S, just transform to tensor.
            onsiteSs.append(sk_onsiteSs_ib)

    hoppings = []
    overlaps = []
    for ib in range(len(nn_hoppings)):
        if np.linalg.norm(sk_hoppings[ib]) < 1e-6:
            sk_hoppings_ib = sk_hoppings[ib] + 1e-6
        else:
            sk_hoppings_ib = sk_hoppings[ib]
        sk_hoppings_ib.requires_grad= False
        hoppings.append(sk_hoppings_ib * (1 + nn_hoppings[ib]))

        if sk_overlaps:
            sk_overlaps_ib = sk_overlaps[ib]
            sk_overlaps_ib.requires_grad = False
            # no correction to overlaps, just transform to tensor.
            overlaps.append(sk_overlaps_ib)

    if sk_overlaps:
        return onsiteEs, hoppings, onsiteSs, overlaps
    else:
        return onsiteEs, hoppings, None, None

if __name__ == '__main__':
    print(get_neuron_config(nl=[0,1,2,3,4,5,6,7]))
