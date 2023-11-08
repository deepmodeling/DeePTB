# define the integrals formula.
import torch as th
from abc import ABC, abstractmethod
from dptb.nnsktb.bondlengthDB import bond_length


class BaseOnsite(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def skEs(self, **kwargs):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by type is called to cal onsite energies and returned.
        
        '''
        pass


class onsiteFormula(BaseOnsite):

    def __init__(self, functype='none') -> None:
        super().__init__() 
        if functype == 'none':
            self.functype = functype
            self.num_paras = 0

        elif functype == 'uniform':
            self.functype = functype
            self.num_paras = 1
            assert hasattr(self, 'uniform')
        elif functype == 'NRL':
            self.functype = functype
            self.num_paras = 4
            assert hasattr(self, 'NRL')

        elif functype == 'custom':
            self.functype = functype
            self.num_paras = None # defined by custom.
            assert hasattr(self, 'custom')
        else:
            raise ValueError('No such formula')
        
    def skEs(self, **kwargs):
        if self.functype == 'uniform':
            return self.uniform(**kwargs)
        if self.functype == 'NRL':
            return self.NRL(**kwargs)
    
    def uniform(self, xtype, onsite_db, nn_onsite_paras):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal onsite energies and returned.
        
        '''
        assert xtype in onsite_db.keys(), f'{xtype} is not in the onsite_db.'
        assert xtype in nn_onsite_paras.keys(), f'{xtype} is not in the nn_onsite_paras.'
        assert onsite_db[xtype].shape == nn_onsite_paras[xtype].shape, f'{xtype} onsite_db and nn_onsite_paras have different shape.'
        return onsite_db[xtype] + nn_onsite_paras[xtype]
        

    def NRL(self, x_onsite_envs, nn_onsite_paras, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, lda=1.0):
        """ This is NRL-TB formula for onsite energies.

            rho_i = \sum_j exp(- lda**2 r_ij) f(r_ij)
        
            E_il = a_l + b_l rho_i^(2/3) + c_l rho_i^(4/3) + d_l rho_i^2

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)
        Parameters
        ----------
        x_onsite_envs: list
            the rij list for i atom. j is the neighbor atoms of i.
        nn_onsite_paras: dict
            the parameters coefficient for onsite energies.
            ['N-2s-0':[...]
            ...]
        rcut: float
            the cutoff radius for onsite energies.
        w: float
            the decay for the  cutoff smoth function.
        lda: float
            the decay for the  calculateing rho.
        """ 
        r_ijs = x_onsite_envs
        exp_rij = th.exp(-lda**2 * r_ijs)
        f_rij = 1/(1+th.exp((r_ijs-rcut+5*w)/w))
        f_rij[r_ijs>=rcut] = 0.0
        rho_i = th.sum(exp_rij * f_rij)
        a_l, b_l, c_l, d_l = nn_onsite_paras[:,0], nn_onsite_paras[:,1], nn_onsite_paras[:,2], nn_onsite_paras[:,3]
        E_il = a_l + b_l * rho_i**(2/3) + c_l * rho_i**(4/3) + d_l * rho_i**2
        return E_il