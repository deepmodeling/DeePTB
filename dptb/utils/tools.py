import os
import re
import numpy as np
import torch
import torch.nn.functional as F
from dptb.utils.constants import atomic_num_dict, anglrMId, SKBondType
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
from ase.neighborlist import neighbor_list
from ase.io.trajectory import Trajectory
import ase
import ssl
import os.path as osp
import urllib
import zipfile
import sys


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


def float2comlex(dtype):
    if isinstance(dtype, str):
        dtype =  getattr(torch, dtype)
    
    if dtype is torch.float32:
        cdtype = torch.complex64
    elif dtype is torch.float64:
        cdtype = torch.complex128
    else:
        raise ValueError("the dtype is not supported! now only float64, float32 is supported!")
    return cdtype


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
            log.error(msg="the {0} in input config is not align with it in checkpoint.".format(cid))
            raise ValueError("the {0} in input config is not align with it in checkpoint.".format(cid))

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
    if type == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, **sch_options)
    elif type == 'linear':
        scheduler = optim.lr_scheduler.LinearLR(optimizer=optimizer, **sch_options)
    elif type == "rop":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, **sch_options)
    elif type == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, **sch_options)
    elif type == "cyclic":
        scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, **sch_options)
    else:
        raise RuntimeError("Scheduler should be exp/linear/rop/cyclic..., not {}".format(type))

    return scheduler

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
    elif activation == "silu":
        return torch.nn.SiLU()

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

def get_hopping_neuron_config(neuron_list, bond_num_hops, bond_type, axis_neuron, env_out):
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
        if re.search('\\*',strtmp):
            strspt = re.split('\\*|\n',strtmp)
            strspt = list(filter(None,strspt))
            assert len(strspt) == 2, "The format of the line is not correct! n*value, the value is gone!"
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

        Delta(x) = lim_{sigma to 0}  Lorentzian
    '''

    return 1. / np.pi * sigma**2 / ((x - x0)**2 + sigma**2)


def GaussianSmearing(x, x0, sigma=0.02):
    '''
    Simulate the Delta function by a Lorentzian shape function

        Delta(x) = lim_{sigma to 0} Gaussian
    '''

    return 1. / (np.sqrt(2*np.pi) * sigma) * np.exp(-(x - x0)**2 / (2*sigma**2))

def write_skparam(
        onsite_coeff, 
        hopping_coeff, 
        onsite_index_dict, 
        rcut=None, 
        w=None,
        atom=None,
        onsite_cutoff=None, 
        bond_cutoff=None, 
        soc_coeff=None,
        thr=1e-3, 
        onsitemode="none", 
        functype="varTang96", 
        format="sktable",
        outPath="./"
        ):
    onsite = {}
    hopping = {}
    soc = {}

    if format == 'sktable':
        r_bonds = get_neighbours(atom, cutoff=bond_cutoff, thr=thr)
        r_onsites = get_neighbours(atom, cutoff=onsite_cutoff, thr=thr)
        skformula = SKFormula(functype=functype)

    jdata = {}

    # onsites
    if onsitemode ==  "strain":
        for i in onsite_coeff.keys():
            kk = "-".join(i.split("-")[:2])
            if format == "DeePTB":
                onsite[i] = onsite_coeff[i].tolist()
            elif format == "sktable":
                rijs = r_onsites.get(kk, None)
                if rijs is None:
                    kkr = "-".join(i.split("-")[:2][::-1])
                    rijs = r_onsites.get(kkr, None)
                # assert rijs is not None

                if rijs is not None:
                    paraArray = onsite_coeff[i]
                    iatomtype, jatomtype, iorb, jorb, lm = i.split("-")
                    ijbond_param = onsite.setdefault(iatomtype+"-"+jatomtype, {})
                    ijorb_param = ijbond_param.setdefault(iorb+"-"+jorb, {})
                    skparam = ijorb_param.setdefault(SKBondType[int(lm)], [])

                    for rij in rijs:
                        params = {
                            'paraArray':paraArray,'rij':torch.scalar_tensor(rij), 'iatomtype':iatomtype,
                            'jatomtype':jatomtype
                            }
                        with torch.no_grad():
                            skparam.append(skformula.skhij(**params).tolist()[0])
            else:
                log.error(msg="Wrong format, please choose from [DeePTB] for checkpoint format, or [sktable] for hopping and onsite table.")
                raise ValueError
            
    elif onsitemode in ['uniform','split']:
        if format == "DeePTB":
            for ia in onsite_coeff:
                for iikey in range(len(onsite_index_dict[ia])):
                    onsite[onsite_index_dict[ia][iikey]] = \
                                            [onsite_coeff[ia].tolist()[iikey]]
        elif format == "sktable":
            for ia in onsite_coeff:
                iatom_param = onsite.setdefault(ia, {})
                for iikey in range(len(onsite_index_dict[ia])):
                    iatomtype, iorb, index = onsite_index_dict[ia][iikey].split("-")
                    iorb_param = iatom_param.setdefault(iorb, {})
                    iorb_param[index] = onsite_coeff[iatomtype].tolist()[iikey]+onsite_energy_database[iatomtype][iorb]
                    # onsite[onsite_index_dict[ia][iikey]] = \
                    #                         [onsite_coeff[iatomtype].tolist()[iikey]+onsite_energy_database[iatomtype][iorb]]
                    
    elif onsitemode == "none":
        jdata["onsite"] = {}
    else:
        log.error(msg="The onsite mode is incorrect!")
        raise ValueError
    jdata["onsite"] = onsite

    for i in hopping_coeff.keys():
        if format == "DeePTB":
            hopping[i] = hopping_coeff[i].tolist()
        elif format == "sktable":
            kk = "-".join(i.split("-")[:2])

            rijs = r_bonds.get(kk, None)
            if rijs is None:
                kkr = "-".join(i.split("-")[:2][::-1])
                rijs = r_onsites.get(kkr, None)
            
            if rijs is not None:

                paraArray = hopping_coeff[i]
                iatomtype, jatomtype, iorb, jorb, lm = i.split("-")
                ijbond_param = hopping.setdefault(iatomtype+"-"+jatomtype, {})
                ijorb_param = ijbond_param.setdefault(iorb+"-"+jorb, {})
                skparam = ijorb_param.setdefault(SKBondType[int(lm)], [])

                for rij in rijs:
                    params = {
                        'paraArray':paraArray,'rij':torch.scalar_tensor(rij), 'iatomtype':iatomtype,
                        'jatomtype':jatomtype, 'rcut':rcut,'w':w
                        }
                    with torch.no_grad():
                        skparam.append(skformula.skhij(**params).tolist()[0])

    jdata["hopping"] = hopping
    
    if soc_coeff is not None:

        if format == "DeePTB":
            for ia in soc_coeff:
                for iikey in range(len(onsite_index_dict[ia])):
                    soc[onsite_index_dict[ia][iikey]] = \
                                            [soc_coeff[ia].tolist()[iikey]]
        elif format == "sktable":
            for ia in onsite_coeff:
                iatom_param = soc.setdefault(ia, {})
                for iikey in range(len(onsite_index_dict[ia])):
                    iatomtype, iorb, index = onsite_index_dict[ia][iikey].split("-")

                    iorb_param = iatom_param.setdefault(iorb, {})
                    iorb_param[index] = soc_coeff[iatomtype].tolist()[iikey]
                    # onsite[onsite_index_dict[ia][iikey]] = \
                    #                         [onsite_coeff[iatomtype].tolist()[iikey]+onsite_energy_database[iatomtype][iorb]]
        jdata["soc"] = soc

    with open(f'{outPath}/skparam.json', "w") as f:
        json.dump(jdata, f, indent=4)

    return True

def get_neighbours(atom: ase.Atom, cutoff: float =10., thr: float =1e-3):
    """
        Generating bond-wise distance dict, where key is the bond symbol such as "A-B", "A-A".
        and the value is a list containing the first, second, third ... bond distance.

    Args:
        atom (ase.Atom): ase atom type structure
        cutoff (float, optional): cutoff on bond distance, control how far the bonds need to be included. Defaults to 10..
        thr (float, optional): control the threshold of bond length difference, within which we assume two bond are the same. Defaults to 1e-3.

    Returns:
        dict: a bond-wise distance dict
    """

    neighbours = {}
    i,j,d = neighbor_list(quantities=["i","j","d"], a=atom, cutoff=cutoff)
    atom_symbols = np.array(atom.get_chemical_symbols(), dtype=str)
    for idx in range(len(i)):
        symbol = get_uniq_symbol([atom_symbols[i[idx]], atom_symbols[j[idx]]])
        if len(symbol) == 2:
            symbol = symbol[0]+"-"+symbol[1]
        else:
            symbol = symbol[0]+"-"+symbol[0]

        nns = neighbours.setdefault(symbol, [])
        if len(nns) >= 1:
            if not (np.abs(d[idx]-np.array(nns)) < thr).any():
                nns.append(d[idx])
        else:
            nns.append(d[idx])

    for kk in neighbours.keys():
        neighbours[kk] = sorted(neighbours[kk])
    
    return neighbours

def bn_stast(traj_path: str, cutoff: float =10., nns=[3.0, 4.5], first=False, remove_self=True):
    """
        Generating bond-wise distance dict, where key is the bond symbol such as "A-B", "A-A".
        and the value is a list containing the first, second, third ... bond distance.

    Args:
        atom (ase.Atom): ase atom type structure
        cutoff (float, optional): cutoff on bond distance, control how far the bonds need to be included. Defaults to 10..
        thr (float, optional): control the threshold of bond length difference, within which we assume two bond are the same. Defaults to 1e-3.

    Returns:
        dict: a bond-wise distance dict
    """
    from tqdm import tqdm

    d = []


    stast = []
    xdat = Trajectory(filename=traj_path, mode='r')
    for atom in tqdm(xdat):
        i,j,S,new_d = neighbor_list(quantities=["i","j","S","d"], a=atom, cutoff=cutoff)
        if remove_self:
            d = np.append(d, new_d[i!=j])
        else:
            d = np.append(d, new_d)

    if first:
        return d
    
    for i in range(len(nns)):
        if i > 0:
            stast.append(d[d.__gt__(nns[i-1]) * d.__lt__(nns[i])])
        else:
            stast.append(d[d.__lt__(nns[i])])
    
    return stast

def makedirs(dir):
    os.makedirs(dir, exist_ok=True)


def download_url(url, folder, log=True):
    r"""Downloads the content of an URL to a specific folder.

    Args:
        url (string): The url.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """

    filename = url.rpartition("/")[2].split("?")[0]
    path = osp.join(folder, filename)

    if osp.exists(path):  # pragma: no cover
        if log:
            print("Using existing file", filename, file=sys.stderr)
        return path

    if log:
        print("Downloading", url, file=sys.stderr)

    makedirs(folder)

    context = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=context)

    with open(path, "wb") as f:
        f.write(data.read())

    return path


def extract_zip(path, folder, log=True):
    r"""Extracts a zip archive to a specific folder.

    Args:
        path (string): The path to the tar archive.
        folder (string): The folder.
        log (bool, optional): If :obj:`False`, will not print anything to the
            console. (default: :obj:`True`)
    """
    with zipfile.ZipFile(path, "r") as f:
        f.extractall(folder)

if __name__ == '__main__':
    print(get_neuron_config(nl=[0,1,2,3,4,5,6,7]))
