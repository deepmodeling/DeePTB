import torch
import numpy as np
from xitorch._core.pure_function import get_pure_function, make_sibling
from typing import Callable, Union, Mapping, Any, Sequence
from xitorch.debug.modes import is_debug_enabled
from xitorch._utils.misc import set_default_option, TensorNonTensorSeparator, \
    TensorPacker, get_method
from xitorch._utils.assertfuncs import assert_fcn_params, assert_runtime
from abc import abstractmethod
import sys
import time
from func_timeout import func_timeout, FunctionTimedOut
import matplotlib.pyplot as plt
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
        if re.search('\*', strtmp):
            strspt = re.split('\*|\n', strtmp)
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

def _isinf(x):
    return torch.any(torch.isinf(x))

class _BaseInfTransform(object):
    @abstractmethod
    def forward(self, t):
        pass

    @abstractmethod
    def dxdt(self, t):
        pass

    @abstractmethod
    def x2t(self, x):
        pass

class _TanInfTransform(_BaseInfTransform):
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

        res = _Quadrature.apply(pfunc2, xl, xu, fwd_options, bck_options, nparams,
                                dtype, device, *params, *pfunc.objparams())
        return packer.pack(res)
    else:
        return _Quadrature.apply(pfunc, xl, xu, fwd_options, bck_options, nparams,
                                 dtype, device, *params, *pfunc.objparams())
    
def update_kmap(result_path, kpoint):
    if os.path.exists(os.path.join(result_path, "KMAP.pth")):
        kmap = torch.load(os.path.join(result_path, "KMAP.pth"))
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
        file = torch.load(file_path)
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

class _Quadrature(torch.autograd.Function):
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
            if _isinf(xl) or _isinf(xu):
                tfm = _TanInfTransform()

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

def finite_difference(fn, x, h, dtype=torch.float64):
    t = torch.randn(1, dtype=dtype)
    h = torch.scalar_tensor(h, dtype=dtype)
    shape = x.shape
    x = x.flatten()
    dev = torch.zeros_like(x)
    for i, ix in enumerate(x):
        xp = x.clone()
        xm = x.clone()
        xp[i] += h
        xm[i] -= h
        dev[i] = fn(xp.reshape(shape)).type_as(t) - fn(xm.reshape(shape)).type_as(t)
        dev[i] = dev[i] / (2*h)
    
    dev = dev.reshape(shape)

    return dev

'''

Copyright 2020 Ryan (Mohammad) Solgi

Permission is hereby granted, free of charge, to any person obtaining a copy of 
this software and associated documentation files (the "Software"), to deal in 
the Software without restriction, including without limitation the rights to use, 
copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the 
Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
SOFTWARE.

'''

###############################################################################
###############################################################################
###############################################################################




###############################################################################
###############################################################################
###############################################################################

class geneticalgorithm():
    '''  Genetic Algorithm (Elitist version) for Python

    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.



    Implementation and output:

        methods:
                run(): implements the genetic algorithm

        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }

                report: a list including the record of the progress of the
                algorithm over iterations

    '''

    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None, \
                 variable_type_mixed=None, \
                 function_timeout=10, \
                 algorithm_parameters={'max_num_iteration': None, \
                                       'population_size': 100, \
                                       'mutation_probability': 0.1, \
                                       'elit_ratio': 0.01, \
                                       'crossover_probability': 0.5, \
                                       'parents_portion': 0.3, \
                                       'crossover_type': 'uniform', \
                                       'max_iteration_without_improv': None}, \
                 convergence_curve=True, \
                 progress_bar=True, id=0):

        '''
        @param function <Callable> - the given objective function to be minimized
        NOTE: This implementation minimizes the given objective function.
        (For maximization multiply function by a negative sign: the absolute
        value of the output would be the actual objective function)

        @param dimension <integer> - the number of decision variables

        @param variable_type <string> - 'bool' if all variables are Boolean;
        'int' if all variables are integer; and 'real' if all variables are
        real value or continuous (for mixed type see @param variable_type_mixed)

        @param variable_boundaries <numpy array/None> - Default None; leave it
        None if variable_type is 'bool'; otherwise provide an array of tuples
        of length two as boundaries for each variable;
        the length of the array must be equal dimension. For example,
        np.array([0,100],[0,200]) determines lower boundary 0 and upper boundary 100 for first
        and upper boundary 200 for second variable where dimension is 2.

        @param variable_type_mixed <numpy array/None> - Default None; leave it
        None if all variables have the same type; otherwise this can be used to
        specify the type of each variable separately. For example if the first
        variable is integer but the second one is real the input is:
        np.array(['int'],['real']). NOTE: it does not accept 'bool'. If variable
        type is Boolean use 'int' and provide a boundary as [0,1]
        in variable_boundaries. Also if variable_type_mixed is applied,
        variable_boundaries has to be defined.

        @param function_timeout <float> - if the given function does not provide
        output before function_timeout (unit is seconds) the algorithm raise error.
        For example, when there is an infinite loop in the given function.

        @param algorithm_parameters:
            @ max_num_iteration <int> - stoping criteria of the genetic algorithm (GA)
            @ population_size <int>
            @ mutation_probability <float in [0,1]>
            @ elit_ration <float in [0,1]>
            @ crossover_probability <float in [0,1]>
            @ parents_portion <float in [0,1]>
            @ crossover_type <string> - Default is 'uniform'; 'one_point' or
            'two_point' are other options
            @ max_iteration_without_improv <int> - maximum number of
            successive iterations without improvement. If None it is ineffective

        @param convergence_curve <True/False> - Plot the convergence curve or not
        Default is True.
        @progress_bar <True/False> - Show progress bar or not. Default is True.

        for more details and examples of implementation please visit:
            https://github.com/rmsolgi/geneticalgorithm

        '''
        self.__name__ = geneticalgorithm
        #############################################################
        # input function
        assert (callable(function)), "function must be callable"
        self.id = id
        self.f = function
        #############################################################
        # dimension

        self.dim = int(dimension)

        #############################################################
        # input variable type

        assert (variable_type == 'bool' or variable_type == 'int' or \
                variable_type == 'real'), \
            "\n variable_type must be 'bool', 'int', or 'real'"
        #############################################################
        # input variables' type (MIXED)

        if variable_type_mixed is None:

            if variable_type == 'real':
                self.var_type = np.array([['real']] * self.dim)
            else:
                self.var_type = np.array([['int']] * self.dim)


        else:
            assert (type(variable_type_mixed).__module__ == 'numpy'), \
                "\n variable_type must be numpy array"
            assert (len(variable_type_mixed) == self.dim), \
                "\n variable_type must have a length equal dimension."

            for i in variable_type_mixed:
                assert (i == 'real' or i == 'int'), \
                    "\n variable_type_mixed is either 'int' or 'real' " + \
                    "ex:['int','real','real']" + \
                    "\n for 'boolean' use 'int' and specify boundary as [0,1]"

            self.var_type = variable_type_mixed
        #############################################################
        # input variables' boundaries

        if variable_type != 'bool' or type(variable_type_mixed).__module__ == 'numpy':

            assert (type(variable_boundaries).__module__ == 'numpy'), \
                "\n variable_boundaries must be numpy array"

            assert (len(variable_boundaries) == self.dim), \
                "\n variable_boundaries must have a length equal dimension"

            for i in variable_boundaries:
                assert (len(i) == 2), \
                    "\n boundary for each variable must be a tuple of length two."
                assert (i[0] <= i[1]), \
                    "\n lower_boundaries must be smaller than upper_boundaries [lower,upper]"
            self.var_bound = variable_boundaries
        else:
            self.var_bound = np.array([[0, 1]] * self.dim)

        #############################################################
        # Timeout
        self.funtimeout = float(function_timeout)
        #############################################################
        # convergence_curve
        if convergence_curve == True:
            self.convergence_curve = True
        else:
            self.convergence_curve = False
        #############################################################
        # progress_bar
        if progress_bar == True:
            self.progress_bar = True
        else:
            self.progress_bar = False
        #############################################################
        #############################################################
        # input algorithm's parameters

        self.param = algorithm_parameters

        self.pop_s = int(self.param['population_size'])

        assert (self.param['parents_portion'] <= 1 \
                and self.param['parents_portion'] >= 0), \
            "parents_portion must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (self.prob_mut <= 1 and self.prob_mut >= 0), \
            "mutation_probability must be in range [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert (self.prob_cross <= 1 and self.prob_cross >= 0), \
            "mutation_probability must be in range [0,1]"

        assert (self.param['elit_ratio'] <= 1 and self.param['elit_ratio'] >= 0), \
            "elit_ratio must be in range [0,1]"

        trl = self.pop_s * self.param['elit_ratio']
        if trl < 1 and self.param['elit_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert (self.par_s >= self.num_elit), \
            "\n number of parents must be greater than number of elits"

        if self.param['max_num_iteration'] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * self.dim * (100 / self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] - self.var_bound[i][0]) * 50 * (100 / self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate * self.pop_s) > 10000000:
                self.iterate = 10000000 / self.pop_s
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or \
                self.c_type == 'two_point'), \
            "\n crossover_type must 'uniform', 'one_point', or 'two_point' Enter string"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] == None:
            self.mniwi = self.iterate + 1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

        #############################################################

    def run(self):

        #############################################################
        # Initial Population

        self.integers = np.where(self.var_type == 'int')
        self.reals = np.where(self.var_type == 'real')

        pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)
        solo = np.zeros(self.dim + 1)
        var = np.zeros(self.dim)

        for p in range(0, self.pop_s):

            for i in self.integers[0]:
                var[i] = np.random.randint(self.var_bound[i][0], \
                                           self.var_bound[i][1] + 1)
                solo[i] = var[i].copy()
            for i in self.reals[0]:
                var[i] = self.var_bound[i][0] + np.random.random() * \
                         (self.var_bound[i][1] - self.var_bound[i][0])
                solo[i] = var[i].copy()

            obj = self.sim(var)
            solo[self.dim] = obj
            pop[p] = solo.copy()

        #############################################################

        #############################################################
        # Report
        self.report = []
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj
        temp_report = []
        ##############################################################
        start = time.perf_counter()
        t = 1
        counter = 0
        while t <= self.iterate:
            temp_report.append((self.report,self.best_function,self.best_variable,time.perf_counter()-start))
            torch.save(obj=temp_report ,f="../dop_GA"+str(self.id)+".pth")



            if self.progress_bar == True:
                self.progress(t, self.iterate, status="GA is running...")
            #############################################################
            # Sort
            pop = pop[pop[:, self.dim].argsort()]

            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
            else:
                counter += 1
            #############################################################
            # Report

            self.report.append(pop[0, self.dim])


            ##############################################################
            # Normalizing objective function

            normobj = np.zeros(self.pop_s)

            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim] + abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm - normobj + 1

            #############################################################
            # Calculate probability

            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj / sum_normobj
            cumprob = np.cumsum(prob)

            #############################################################
            # Select parents
            par = np.array([np.zeros(self.dim + 1)] * self.par_s)

            for k in range(0, self.num_elit):
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()

            ef_par_list = np.array([False] * self.par_s)
            par_count = 0
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            ef_par = par[ef_par_list].copy()

            #############################################################
            # New generation
            pop = np.array([np.zeros(self.dim + 1)] * self.pop_s)

            for k in range(0, self.par_s):
                pop[k] = par[k].copy()

            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy()
                pvar2 = ef_par[r2, : self.dim].copy()

                ch = self.cross(pvar1, pvar2, self.c_type)
                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                ch1 = self.mut(ch1)
                ch2 = self.mutmidle(ch2, pvar1, pvar2)
                solo[: self.dim] = ch1.copy()
                obj = self.sim(ch1)
                solo[self.dim] = obj
                pop[k] = solo.copy()
                solo[: self.dim] = ch2.copy()
                obj = self.sim(ch2)
                solo[self.dim] = obj
                pop[k + 1] = solo.copy()
            #############################################################
            t += 1
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    if self.progress_bar == True:
                        self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(2)
                    t += 1
                    self.stop_mniwi = True

        #############################################################
        # Sort
        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:
            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()
        #############################################################
        # Report

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function': \
            self.best_function}
        if self.progress_bar == True:
            show = ' ' * 100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush()
        re = np.array(self.report)
        if self.convergence_curve == True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()

        if self.stop_mniwi == True:
            sys.stdout.write('\nWarning: GA is terminated due to the' + \
                             ' maximum number of iterations without improvement was met!')

    ##############################################################################
    ##############################################################################
    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == 'two_point':

            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == 'uniform':

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])

    ###############################################################################

    def mut(self, x):

        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = np.random.randint(self.var_bound[i][0], \
                                         self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                x[i] = self.var_bound[i][0] + np.random.random() * \
                       (self.var_bound[i][1] - self.var_bound[i][0])

        return x

    ###############################################################################
    def mutmidle(self, x, p1, p2):
        for i in self.integers[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = np.random.randint(p1[i], p2[i])
                elif p1[i] > p2[i]:
                    x[i] = np.random.randint(p2[i], p1[i])
                else:
                    x[i] = np.random.randint(self.var_bound[i][0], \
                                             self.var_bound[i][1] + 1)

        for i in self.reals[0]:
            ran = np.random.random()
            if ran < self.prob_mut:
                if p1[i] < p2[i]:
                    x[i] = p1[i] + np.random.random() * (p2[i] - p1[i])
                elif p1[i] > p2[i]:
                    x[i] = p2[i] + np.random.random() * (p1[i] - p2[i])
                else:
                    x[i] = self.var_bound[i][0] + np.random.random() * \
                           (self.var_bound[i][1] - self.var_bound[i][0])
        return x

    ###############################################################################
    def evaluate(self):
        return self.f(self.temp)

    ###############################################################################
    def sim(self, X):
        self.temp = X.copy()
        obj = None
        try:
            obj = func_timeout(self.funtimeout, self.evaluate)
        except FunctionTimedOut:
            print("given function is not applicable")
        assert (obj != None), "After " + str(self.funtimeout) + " seconds delay " + \
                              "func_timeout: the given function does not provide any output"
        return obj

    ###############################################################################
    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()
    ###############################################################################
###############################################################################

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



def write_vesta_lcurrent(positions, vesta_file, lcurrent, current):
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
            if abs(net_current) > 1e-6:
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

    with open(vesta_file, "w") as f:
        f.write(data)