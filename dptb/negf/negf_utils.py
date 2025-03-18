import torch
import numpy as np
from xitorch._core.pure_function import get_pure_function, make_sibling
from typing import Callable, Union, Mapping, Any, Sequence
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, \
    TensorPacker, get_method
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from abc import abstractmethod
import re
import os
import ase
from scipy.interpolate import interp1d
import logging



log = logging.getLogger(__name__)

anglrMId = {'s':0,'p':1,'d':2,'f':3}
MaxShells  = 3
NumHvals   = 10
atomic_num_dict = ase.atom.atomic_numbers

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
    lsplit = re.split(',|;| +|\n|\t', line)
    lsplit = list(filter(None, lsplit))
    lstr = []
    for ii in range(len(lsplit)):
        strtmp = lsplit[ii]
        if re.search(r'\*', strtmp):
            strspt = re.split(r'\*|\n', strtmp)
            strspt = list(filter(None, strspt))
            strfull = int(strspt[0]) * [strspt[1]]
            lstr += strfull
        else:
            lstr += [strtmp]
    return lstr

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

def find_isinf(x):
    return torch.any(torch.isinf(x))

class ADBaseInfTransform(object):
    @abstractmethod
    def forward(self, t):
        pass

    @abstractmethod
    def dxdt(self, t):
        pass

    @abstractmethod
    def x2t(self, x):
        pass

class ADTanInfTransform(ADBaseInfTransform):
    def forward(self, t):
        return torch.tan(t)

    def dxdt(self, t):
        sec = 1. / torch.cos(t)
        return sec * sec

    def x2t(self, x):
        return torch.atan(x)

def leggauss(fcn, xl, xu, params, n=100, **unused):
    """
    Performing 1D integration using Legendre-Gaussian quadrature

    Keyword arguments
    -----------------
    n: int
        The number of integration points.
    """
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    ndim = len(xu.shape)
    xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)

    res = wlg[0] * fcn(xs[0], *params)
    for i in range(1, n):
        res += wlg[i] * fcn(xs[i], *params)
    return res

def gauss_xw(xl, xu, n=100):
    '''calculates the Gauss-Legendre quadrature points and weights for numerical  integration.
    
    Parameters
    ----------
    xl
        The lower limit of integration.
    xu
        The upper limit of the integration range.
    n, optional
        the number of points to be used in the Gauss-Legendre quadrature.    
    Returns
    -------
        point xs and their weights wlg.
    
    '''
    ndim = len(xu.shape)
    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = torch.tensor(xlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg = torch.tensor(wlg, dtype=xu.dtype, device=xu.device)[(...,) + (None,) * ndim]  # (n, *nx)
    wlg *= 0.5 * (xu - xl)
    xs = xlg * (0.5 * (xu - xl)) + (0.5 * (xu + xl))  # (n, *nx)

    return xs, wlg


def quad(
        fcn: Union[Callable[..., torch.Tensor], Callable[..., Sequence[torch.Tensor]]],
        xl: Union[float, int, torch.Tensor],
        xu: Union[float, int, torch.Tensor],
        params: Sequence[Any] = [],
        bck_options: Mapping[str, Any] = {},
        method: Union[str, Callable, None] = None,
        **fwd_options) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
    r"""
    Calculate the quadrature:

    .. math::

        y = \int_{x_l}^{x_u} f(x, \theta)\ \mathrm{d}x

    Arguments
    ---------
    fcn: callable
        The function to be integrated. Its output must be a tensor with
        shape ``(*nout)`` or list of tensors.
    xl: float, int or 1-element torch.Tensor
        The lower bound of the integration.
    xu: float, int or 1-element torch.Tensor
        The upper bound of the integration.
    params: list
        Sequence of any other parameters for the function ``fcn``.
    bck_options: dict
        Options for the backward quadrature method.
    method: str or callable or None
        Quadrature method. If None, it will choose ``"leggauss"``.
    **fwd_options
        Method-specific options (see method section).

    Returns
    -------
    torch.tensor or a list of tensors
        The quadrature results with shape ``(*nout)`` or list of tensors.
    """
    # perform implementation check if debug mode is enabled
    if is_debug_enabled():
        assert_fcn_params(fcn, (xl, *params))
    if isinstance(xl, torch.Tensor):
        assert_runtime(torch.numel(xl) == 1, "xl must be a 1-element tensors")
    if isinstance(xu, torch.Tensor):
        assert_runtime(torch.numel(xu) == 1, "xu must be a 1-element tensors")
    if method is None:
        method = "leggauss"
    fwd_options["method"] = method

    out = fcn(xl, *params)
    if isinstance(out, torch.Tensor):
        dtype = out.dtype
        device = out.device
        is_tuple_out = False
    elif len(out) > 0:
        dtype = out[0].dtype
        device = out[0].device
        is_tuple_out = True
    else:
        raise RuntimeError("The output of the fcn must be non-empty")

    pfunc = get_pure_function(fcn)
    nparams = len(params)
    if is_tuple_out:
        packer = TensorPacker(out)

        @make_sibling(pfunc)
        def pfunc2(x, *params):
            y = fcn(x, *params)
            return packer.flatten(y)

        res = Quadrature.apply(pfunc2, xl, xu, fwd_options, bck_options, nparams,
                                dtype, device, *params, *pfunc.objparams())
        return packer.pack(res)
    else:
        return Quadrature.apply(pfunc, xl, xu, fwd_options, bck_options, nparams,
                                 dtype, device, *params, *pfunc.objparams())
    
def update_kmap(result_path, kpoint):
    '''updates a Kpoints file
    
    The function `update_kmap` updates a KMAP file by comparing a given kpoint with existing kpoints and
    either adding the new kpoint or returning the index of the closest matching kpoint.
    
    Parameters
    ----------
    result_path
        the path to the directory where the `KMAP.pth` file will be saved or loaded from.
    kpoint
        point in k-space. 
    
    Returns
    -------
        the index `ik` of the updated or newly added `kpoint` in the `kmap` list.
    
    '''
    if os.path.exists(os.path.join(result_path, "KMAP.pth")):
        kmap = torch.load(os.path.join(result_path, "KMAP.pth"), weights_only=False)
        err = np.abs(np.array(kmap) - np.array(kpoint).reshape(1,-1)).sum(axis=1)
        ik = np.argmin(err)
        
        if err[ik] > 1e-7:
            ik = len(kmap)
            kmap.append(kpoint)
    else:
        kmap = [kpoint]
        ik = 0

    torch.save(kmap, os.path.join(result_path, "KMAP.pth"))

    return ik

def update_temp_file(update_fn, file_path, ee, tags, info):
    """This function read the file for temperary results, computing the energy point need to be computed, and update the temperary file.s
    Args:
        file_path (str): _description_
        ee (list): list of energy point
        tags (str): the tag of files need to be updated

    Returns:
        _type_: _description_
    """

    # mismatch of the order of new, emap, e_mesh

    ## fix, giving a comparing accuracy of these ee
    ee = np.array(ee)
    if os.path.exists(file_path):
        file = torch.load(file_path, weights_only=False)
        dis = np.argmin(np.abs(np.array(file["e_mesh"]).reshape(1,-1) - ee.reshape(-1,1)), axis=1)
        # print(ee, file["e_mesh"], dis)
        err = np.abs(ee - np.array(file["e_mesh"])[dis])
        eecal = list(set(ee[err>1e-5]))
    else:
        eecal = list(set(ee))
        file = {"e_mesh":[], "emap":{}}
        err = [1] * len(ee)

        for tag in tags:
            file[tag] = []

    if len(eecal) != 0:
        log.info(msg=info)
        # update temp file
        new = update_fn(eecal)
        n = len(file["e_mesh"])
        # update e_mesh
        file["e_mesh"] += eecal
        # update emap
        for e in eecal:
            file["emap"][complex(e)] = n
            n += 1
        for i in tags:
            file[i] += new[i]

        torch.save(file, file_path)

    # print((torch.stack(new[tags[0]])-torch.stack([file[tags[0]][file["emap"][float(e)]] for e in eecal])).abs().max())
    
    out = {}
    for i in tags:
        # print([file["emap"][float(e)] for e in ee])
        # print(file["e_mesh"])
        out[i] = [file[i][file["emap"][complex(e)]] if err[j]>1e-5 else file[i][dis[j]] for j, e in enumerate(ee)]


    
    return out

class Quadrature(torch.autograd.Function):
    # NOTE: _Quadrature method do not involve changing the state (objparams) of
    # fcn, so there is no need in using `with fcn.useobjparams(objparams)`
    # statements.
    # The function `disable_state_change()` is used to disable state change of
    # the pure function during the execution of the forward and backward
    # calculations

    @staticmethod
    def forward(ctx, fcn, xl, xu, fwd_options, bck_options, nparams,
                dtype, device, *all_params):

        with fcn.disable_state_change():

            config = fwd_options
            ctx.bck_config = set_default_option(config, bck_options)

            params = all_params[:nparams]
            objparams = all_params[nparams:]

            # convert to tensor
            xl = torch.as_tensor(xl, dtype=dtype, device=device)
            xu = torch.as_tensor(xu, dtype=dtype, device=device)

            # apply transformation if the boundaries contain inf
            if find_isinf(xl) or find_isinf(xu):
                tfm = ADTanInfTransform()

                @make_sibling(fcn)
                def fcn2(t, *params):
                    ys = fcn(tfm.forward(t), *params)
                    dxdt = tfm.dxdt(t)
                    return ys * dxdt

                tl = tfm.x2t(xl)
                tu = tfm.x2t(xu)
            else:
                fcn2 = fcn
                tl = xl
                tu = xu

            method = config.pop("method")
            methods = {
                "leggauss": leggauss
            }
            method_fcn = get_method("quad", methods, method)
            y = method_fcn(fcn2, tl, tu, params, **config)

            # save the parameters for backward
            ctx.param_sep = TensorNonTensorSeparator(all_params, varonly=True)
            tensor_params = ctx.param_sep.get_tensor_params()
            ctx.xltensor = isinstance(xl, torch.Tensor)
            ctx.xutensor = isinstance(xu, torch.Tensor)
            xlxu_tensor = ([xl] if ctx.xltensor else []) + \
                          ([xu] if ctx.xutensor else [])
            ctx.xlxu_nontensor = ([xl] if not ctx.xltensor else []) + \
                                 ([xu] if not ctx.xutensor else [])

            ctx.save_for_backward(*xlxu_tensor, *tensor_params)
            ctx.fcn = fcn
            ctx.nparams = nparams
            return y

    @staticmethod
    def backward(ctx, grad_ys):
        # retrieve the params
        ntensor_params = ctx.param_sep.ntensors()
        if ntensor_params != 0:
            tensor_params = ctx.saved_tensors[-ntensor_params:]
        else:
            tensor_params = []
        allparams = ctx.param_sep.reconstruct_params(tensor_params)
        nparams = ctx.nparams
        params = allparams[:nparams]
        fcn = ctx.fcn

        with fcn.disable_state_change():

            # restore xl, and xu
            if ntensor_params != 0:
                xlxu_tensor = ctx.saved_tensors[:-ntensor_params]
            else:
                xlxu_tensor = ctx.saved_tensors
            if ctx.xltensor and ctx.xutensor:
                xl, xu = xlxu_tensor
            elif ctx.xltensor:
                xl = xlxu_tensor[0]
                xu = ctx.xlxu_nontensor[0]
            elif ctx.xutensor:
                xu = xlxu_tensor[0]
                xl = ctx.xlxu_nontensor[0]
            else:
                xl, xu = ctx.xlxu_nontensor

            # calculate the gradient for the boundaries

            grad_xl = -torch.dot(grad_ys.reshape(-1), fcn(xl, *params).reshape(-1)
                                 ).reshape(xl.shape) if ctx.xltensor else None
            grad_xu = torch.dot(grad_ys.reshape(-1), fcn(xu, *params).reshape(-1)
                                ).reshape(xu.shape) if ctx.xutensor else None
            print(grad_ys)

            def new_fcn(x, *grad_y_params):
                grad_ys = grad_y_params[0]
                # not setting objparams and params because the params and objparams
                # are still the same objects as the objects outside
                with torch.enable_grad():
                    f = fcn(x, *params)

                dfdts = torch.autograd.grad(f, tensor_params,
                                            grad_outputs=grad_ys,
                                            retain_graph=True,
                                            create_graph=torch.is_grad_enabled())

                return dfdts

            # reconstruct grad_params
            # listing tensor_params in the params of quad to make sure it gets
            # the gradient calculated
            if ntensor_params != 0:
                dydts = quad(new_fcn, xl, xu, params=(grad_ys, *tensor_params),
                         fwd_options=ctx.bck_config, bck_options=ctx.bck_config)
            else:
                dydts = []
            dydns = [None for _ in range(ctx.param_sep.nnontensors())]
            grad_params = ctx.param_sep.reconstruct_params(dydts, dydns)


            return (None, grad_xl, grad_xu, None, None, None, None, None, *grad_params)



class TensorNonTensorSeparator(object):
    """
    Class that provides function to separate/combine tensors and nontensors
    parameters.
    """

    def __init__(self, params, varonly=True):
        """
        Params is a list of tensor or non-tensor to be splitted into
        tensor/non-tensor
        """
        self.tensor_idxs = []
        self.tensor_params = []
        self.nontensor_idxs = []
        self.nontensor_params = []
        self.nparams = len(params)
        for (i, p) in enumerate(params):
            if isinstance(p, torch.Tensor) and ((varonly and p.requires_grad) or (not varonly)):
                self.tensor_idxs.append(i)
                self.tensor_params.append(p)
            else:
                self.nontensor_idxs.append(i)
                self.nontensor_params.append(p)
        self.alltensors = len(self.tensor_idxs) == self.nparams

    def get_tensor_params(self):
        return self.tensor_params

    def ntensors(self):
        return len(self.tensor_idxs)

    def nnontensors(self):
        return len(self.nontensor_idxs)

    def reconstruct_params(self, tensor_params, nontensor_params=None):
        if nontensor_params is None:
            nontensor_params = self.nontensor_params
        if len(tensor_params) + len(nontensor_params) != self.nparams:
            raise ValueError(
                "The total length of tensor and nontensor params "
                "do not match with the expected length: %d instead of %d" %
                (len(tensor_params) + len(nontensor_params), self.nparams))
        if self.alltensors:
            return tensor_params

        params = [None for _ in range(self.nparams)]
        for nidx, p in zip(self.nontensor_idxs, nontensor_params):
            params[nidx] = p
        for idx, p in zip(self.tensor_idxs, tensor_params):
            params[idx] = p
        return params

def sk_init(proj_atom_anglr_m, sk_file_path, **kwargs):
    '''It takes in a list of atom types, a dictionary of angular momentum for each atom type, and a path to
    the Slater-Koster files, and returns a list of Slater-Koster files, a list of Slater-Koster file
    types, and a dictionary of atom orbital information

    Parameters
    ----------
    proj_atom_anglr_m
        a dictionary of the angular momentum of each atom type. e.g. {'H': ['s'], 'O': ['s','p']}
    sk_file_path
        the path to the directory where the Slater-Koster files are stored.

    Returns
    -------
        skfiles: a list of lists of strings. Each string is a path to a file.
        skfile_types: a list of lists of atom type for the sk files.
            eg: [[...,['C','C'],['C','H'],...],
                 [...,['H','C'],['H','H'],...]]
        atom_orb_infor: a dictionary of atom and orbital information.
            atom_orb_infor ={
            "atom_types":proj_atom_type,
            "anglr_momentum_values":anglr_momentum_values,
            "num_orbs":num_orbs
            }
        proj_atom_type: a list of atom types. no duplicates.
        anglr_momentum_values: a list of angular momentum values, for each atom type.
        num_orbs: a list of the number of orbitals for each atom type.
    '''
    skparas = {
        "separator": "-",
        "suffix": ".skf"
    }
    skparas.update(kwargs)
    skfiles = {}

    proj_atom_type = get_uniq_symbol(list(proj_atom_anglr_m.keys()))

    for itype in proj_atom_type:
        for jtype in proj_atom_type:
            filename = sk_file_path + '/' + itype + skparas["separator"] + jtype + skparas["suffix"]
            if not os.path.exists(filename):
                print('Didn\'t find the skfile: ' + filename)
                exit()
            skfiles[itype + '-' + jtype] = filename

    # proj_type_norbs = {}
    # proj_type_momentum_ids={}
    # for itype in proj_atom_type:
    #     anglrm_i = []
    #     norbs_i = []
    #     for ishell  in proj_atom_anglr_m[itype]:  #[s','p', 'd', ...]
    #         ishell_value = anglrMId[ishell]
    #         norbs_i.append(2 * ishell_value + 1)
    #         anglrm_i.append(ishell_value)
    #     proj_type_norbs[itype] = (norbs_i)
    #     proj_type_momentum_ids[itype] = anglrm_i

    return skfiles


def read_skfiles(skfiles):
    '''It reads the Slater-Koster files names and returns the grid distance, number of grids,
    and the Slater-Koster integrals

    Parameters
    ----------
    skfiles
        a list of skfiles.

    Note: the Slater-Koster files name now consider that all the atoms are interacted with each other.
    Therefore, the number of Slater-Koster files is num_atom_types *2.

    Returns
    -------
    grid_distance
        list, intervals between the grid points from different sk files.
    num_grids,
        list, total number of grids from different sk files.
    HSintgrl
        list, the Slater-Koster integrals from different sk files.

    '''
    assert isinstance(skfiles, dict)
    skfiletypes = list(skfiles.keys())
    num_sk_files = len(skfiletypes)
    grid_distance = {}
    num_grids = {}
    HSintgrl = {}

    SiteE = {}
    HubdU = {}
    Occu = {}

    for isktype in skfiletypes:
        filename = skfiles[isktype]
        print('# Reading SlaterKoster File......')
        print('# ' + filename)
        fr = open(filename)
        data = fr.readlines()
        fr.close()
        # Line 1
        datline = format_readline(data[0])
        gridDist, ngrid = float(datline[0]), int(datline[1])
        ngrid = ngrid - 1
        grid_distance[isktype] = (gridDist)
        num_grids[isktype] = (ngrid)

        HSvals = np.zeros([ngrid, NumHvals * 2])
        atomtypes = isktype.split(sep='-')
        if atomtypes[0] == atomtypes[1]:
            print('# This file is a Homo-nuclear case!')
            # Line 2 for Homo-nuclear case
            datline = format_readline(data[1])
            # Ed Ep Es, spe, Ud Up Us, Od Op Os.
            # order from d p s -> s p d.
            SiteE[atomtypes[0]] = np.array([float(datline[2 - ish]) for ish in range(MaxShells)])
            HubdU[atomtypes[0]] = np.array([float(datline[6 - ish]) for ish in range(MaxShells)])
            Occu[atomtypes[0]] = np.array([float(datline[9 - ish]) for ish in range(MaxShells)])

            for il in range(3, 3 + ngrid):
                datline = format_readline(data[il])
                HSvals[il - 3] = np.array([float(val) for val in datline[0:2 * NumHvals]])
        else:
            print('# This is for Hetero-nuclear case!')
            for il in range(2, 2 + ngrid):
                datline = format_readline(data[il])
                HSvals[il - 2] = np.array([float(val) for val in datline[0:2 * NumHvals]])
        HSintgrl[isktype] = HSvals

    return grid_distance, num_grids, HSintgrl, SiteE, HubdU, Occu


def interp_sk_gridvalues(skfile_types, grid_distance, num_grids, HSintgrl):
    '''It reads the Slater-Koster files names and skfile_types
    to generate the  interpolates of the Slater-Koster integrals.

    Parameters
    ----------
    skfile_types
        a list of skfiles types like: 'C-C', 'C-H', 'H-C', ...
    grid_distance
       dict, the grid intervals for each skfile type. e.g. {'C-C':0.1, 'C-H':0.1, ...}
    num_grids
        dict, the number of grids for each skfile type. e.g. {'C-C':100, 'C-H':100, ...}
    HSintgrl
        dict, the H intergrals and ovelaps for each skfile type. e.g. {'C-C': H S values., 'C-H': ,H S values ...}

    Returns
    -------
    max_min_bond_length
        dict, the max and min bond length for each skfile type. e.g. {'C-C': [0.1, 10], 'C-H': [0.1, 10], ...}
    interp_skfunc
        dict, the interpolation functions for each skfile type. e.g. {'C-C': interpolation function, 'C-H': interpolation function ...}

    '''
    MaxDistail = 1.0
    eps = 1.0E-6
    interp_skfunc = {}
    max_min_bond_length = {}

    for isktype in skfile_types:
        xlist = np.arange(1, num_grids[isktype] + 1) * grid_distance[isktype]
        xlist = np.append(xlist, [xlist[-1] + MaxDistail], axis=0)
        max_tmp = num_grids[isktype] * grid_distance[isktype] + MaxDistail - eps
        min_tmp = grid_distance[isktype]

        target = HSintgrl[isktype]
        target = np.append(target, np.zeros([1, 2 * NumHvals]), axis=0)
        intpfunc = interp1d(xlist, target, axis=0)
        interp_skfunc[isktype] = intpfunc
        max_min_bond_length[isktype] = [max_tmp, min_tmp]

    return max_min_bond_length, interp_skfunc



def write_vesta_lcurrent(positions, vesta_file, lcurrent, current, outpath):
    ''' write local current in the form of VESTA file.
       
    Parameters
    ----------
    positions
        the positions of the atoms in the system. 
    vesta_file
        The name of the VESTA file.
    lcurrent
        local current flowing between atoms
    current
        total current flowing through the system.
    outpath
        the path where the modified VESTA file will be saved.
    
    '''
    with open(vesta_file, "r") as f:
        data = f.read()
        f.close()

    replace_start = data.find("VECTR")
    replace_end = data.find("SPLAN")

    N,M = lcurrent.shape
    L_VECTR = []
    L_VECTT = []

    count = 1
    for i in range(0,N):
        for j in range(i+1,M):
            net_current = lcurrent[i,j]-lcurrent[j,i]
            if abs(net_current) / abs(current) > 1e-6:
                if net_current > 0:
                    pos = positions[j]-positions[i]
                    line = [count] + list(pos / np.sqrt((pos**2).sum()) * (net_current/abs(current)))+[1]
                    line = [str(p) for p in line]
                    L_VECTR.append(" ".join(line))
                    L_VECTR.append(str(i+1)+" 0 0 0 0")
                    L_VECTR.append("0 0 0 0 0")
                    L_VECTT.append(str(count) + " 0.2 255 0 0 2")
                else:
                    pos = positions[i]-positions[j]
                    line = [count] + list(-pos / np.sqrt((pos**2).sum()) * (net_current/abs(current)))+[1]
                    line = [str(p) for p in line]
                    L_VECTR.append(" ".join(line))
                    L_VECTR.append(str(j+1)+" 0 0 0 0")
                    L_VECTR.append("0 0 0 0 0")
                    L_VECTT.append(str(count) + " 0.2 255 0 0 2")
                count += 1

    
    text = "VECTR\n"+"\n".join(L_VECTR)+"\n0 0 0 0 0\n"+"VECTT\n"+"\n".join(L_VECTT)+"\n0 0 0 0 0\n"

    data = data[:replace_start]+text+data[replace_end:]

    with open(outpath, "w") as f:
        f.write(data)