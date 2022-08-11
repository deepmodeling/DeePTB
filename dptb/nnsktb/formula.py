# define the integrals formula.
from atexit import register
import numpy as np
import torch as th


def skformula(**kwargs):
    '''This is a wrap function for a self-defined formula of sk integrals. one can easily modify it into whatever form they want.
    
    Returns
    -------
        The function varTang is being returned.
    
    '''
    return varTang(**kwargs)

    
def varTang(alpha1 ,alpha2, alpha3, alpha4, rij, **kwargs):
    '''> This function calculates the value of the variational form of Tang et al 1996. without the
    environment dependent

            $$ h(rij) = \alpha_1 * (rij)^(-\alpha_2) * exp(-\alpha_3 * (rij)^(\alpha_4))$$
    
    Parameters
    ----------
    alpha1
        fitting parameter unit eV. the sk integral at  r = 0. (just a value doesnot meaning anything, since the bond length =0 is non-physical)
    alpha2
        fitting parameter unitless.
    alpha3
        fitting parameter unitless.
    alpha4
        fitting parameter unitless.
    rij
        distance between two atoms
    
    Returns
    -------
        hij, the sk intgral.
    
    '''
    
    hij = alpha1 * (rij**(-alpha2)) * th.exp(-alpha3 * (rij**alpha4))
    
    return hij


