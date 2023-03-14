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
import random


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


def flatten_dict(dictionary):
    queue = list(dictionary.items())
    f_dict = {}
    while(len(queue)>0):
        k, v = queue.pop()
        if isinstance(v, dict) and len(v.items()) != 0:
            for it in v.items():
                queue.append((k+"-"+it[0], it[1]))
        else:
            f_dict.update({k:v})
    
    return f_dict

def reconstruct_dict(dictionary):
    queue = list(dictionary.items())
    r_dict = {}
    while(len(queue)>0):
        k,v = queue.pop()
        ks = k.split("-")
        s_dict = r_dict
        if len(ks) > 1:
            for ik in ks[:-1]:
                if not s_dict.get(ik, None):
                    s_dict.update({ik:{}})
                s_dict = s_dict[ik]
        s_dict[ks[-1]] = v
    
    return r_dict


def checkdict(dict_prototype, dict_update, checklist):
    flatten_dict_prototype = flatten_dict(dict_prototype)
    flatten_dict_update = flatten_dict(dict_update)
    for cid in checklist:
        if flatten_dict_prototype.get(cid) != flatten_dict_update.get(cid):
            raise ValueError

    return True
    
def update_dict(temp_dict, update_dict, checklist):
    '''
        temp_dict: the dict that need to be update, and some of its value need to be checked in case of wrong update
        update_dict: the dict used to update the templete dict
    '''
    flatten_temp_dict = flatten_dict(temp_dict)
    flatten_update_dict = flatten_dict(update_dict)

    for cid in checklist:
        if flatten_update_dict.get(cid) != flatten_temp_dict.get(cid):
            raise ValueError
    
    flatten_temp_dict.update(flatten_update_dict)
    
    return reconstruct_dict(flatten_temp_dict)

def update_dict_with_warning(dict_input, update_list, update_value):
    flatten_input_dict = flatten_dict(dict_input)

    for cid in update_list:
        idx = update_list.index(cid)
        if flatten_input_dict[cid] != update_value[idx]:
            log.warning(msg="Warning! The value {0} of input config has been changed.".format(cid))
            flatten_input_dict[cid] = update_value[idx]
    
    return reconstruct_dict(flatten_input_dict)




def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def get_optimizer(type: str, model_param, lr: float, **options: dict):
    if type == 'Adam':
        optimizer = optim.Adam(params=model_param, lr=lr, **options)
    elif type == 'SGD':
        optimizer = optim.SGD(params=model_param, lr=lr, **options)
    elif type == 'RMSprop':
        optimizer = optim.RMSprop(params=model_param, lr=lr, **options)
    elif type == 'LBFGS':
        optimizer = optim.LBFGS(params=model_param, lr=lr, **options)
    else:
        raise RuntimeError("Optimizer should be Adam/SGD/RMSprop, not {}".format(type))
    return optimizer

def get_lr_scheduler(type: str, optimizer: optim.Optimizer, **sch_options):
    if type == 'Exp':
        schedular = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **sch_options)
    else:
        raise RuntimeError("Scheduler should be Expo/..., not {}".format(type))

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

def get_onsite_neuron_config(neuron_list, onsite_num, onsite_type, axis_neuron, env_out ):
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

def nnsk_correction(nn_onsiteEs, nn_hoppings, sk_onsiteEs, sk_hoppings, sk_onsiteSs=None, sk_overlaps=None, nn_soc_lambdas=None, sk_soc_lambdas=None):
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
        onsiteEs.append(sk_onsiteEs_ib * (1 + nn_onsiteEs[ib]))

        if sk_onsiteSs:
            sk_onsiteSs_ib = sk_onsiteSs[ib]
            onsiteSs.append(sk_onsiteSs_ib)

    if nn_soc_lambdas and sk_soc_lambdas:
        soc_lambdas = []
        for ib in range(len(nn_soc_lambdas)):
            sk_soc_ib = sk_soc_lambdas[ib]
            soc_lambdas.append(sk_soc_ib * (1 + nn_soc_lambdas[ib]))
    else:
        if nn_soc_lambdas:
            soc_lambdas = nn_soc_lambdas
        elif sk_soc_lambdas:
            soc_lambdas = sk_soc_lambdas
        else:
            soc_lambdas = None
    

    hoppings = []
    overlaps = []
    for ib in range(len(nn_hoppings)):
        sk_hoppings_ib = sk_hoppings[ib]
        hoppings.append(sk_hoppings_ib * (1 + nn_hoppings[ib]))

        if sk_overlaps:
            sk_overlaps_ib = sk_overlaps[ib]
            # no correction to overlaps, just transform to tensor.
            overlaps.append(sk_overlaps_ib)

    if sk_overlaps:
        return onsiteEs, hoppings, onsiteSs, overlaps, soc_lambdas
    else:
        return onsiteEs, hoppings, None, None, soc_lambdas


def read_wannier_hr(Filename='wannier90_hr.dat'):
    """Read wannier90_hr.dat."""
    print('reading wannier90_hr.dat ...')
    f=open(Filename,'r')
    data=f.readlines()
    #read hopping matrix
    num_wann = int(data[1])
    nrpts = int(data[2])
    r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    #hop=[]
    #skip n lines of degeneracy of each Wigner-Seitz grid point
    skiplines = int(np.ceil(nrpts / 15.0))
    istart = 3 + skiplines
    deg=[]
    for i in range(3,istart):
        deg.append(np.array([int(j) for j in data[i].split()]))
    deg=np.concatenate(deg,0)
    
    icount=0
    ii=-1
    Rlatt = []
    hopps = []
    for i in range(istart,len(data)):
        line=data[i].split()
        m = int(line[3]) - 1
        n = int(line[4]) - 1
        r_hop[m,n] = complex(round(float(line[5]),6),round(float(line[6]),6))
        icount+=1
        if(icount % (num_wann*num_wann) == 0):
            ii+=1
            R = np.array([float(x) for x in line[0:3]])
            #hop.append(np.asarray([R,r_hop]))
            #r_hop= np.zeros([num_wann,num_wann], dtype=complex)
            
            Rlatt.append(R)
            hopps.append(r_hop)
            #hop.append(np.asarray([R,r_hop]))
            r_hop= np.zeros([num_wann,num_wann], dtype=complex)
    Rlatt=np.asarray(Rlatt,dtype=int)
    hopps=np.asarray(hopps)
    deg = np.reshape(deg,[nrpts,1,1])
    hopps=hopps/deg

    for i in range(nrpts):
        if (Rlatt[i]==0).all():
            indR0 = i
    return Rlatt, hopps, indR0


def LorentzSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function

        \Delta(x) = \lim_{\sigma\to 0}  Lorentzian
    '''

    return 1. / np.pi * sigma**2 / ((x - x0)**2 + sigma**2)


def GaussianSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function

        \Delta(x) = \lim_{\sigma\to 0} Gaussian
    '''

    return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))



if __name__ == '__main__':
    print(get_neuron_config(nl=[0,1,2,3,4,5,6,7]))
