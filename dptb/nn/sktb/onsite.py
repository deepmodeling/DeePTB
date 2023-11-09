# define the integrals formula.
import torch as th
from abc import ABC, abstractmethod
from dptb.nnsktb.bondlengthDB import bond_length
from torch_scatter import scatter
from onsiteDB import onsite_energy_database


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


class OnsiteFormula(BaseOnsite):

    def __init__(self, atomtype=None, functype='none') -> None:
        super().__init__() 
        if functype in ['none', 'strain']:
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
        
        if isinstance(atomtype, list):
            self.E_base = {k:onsite_energy_database[k] for k in atomtype}
        
    def get_skEs(self, **kwargs):
        if self.functype == 'uniform':
            return self.uniform(**kwargs)
        if self.functype == 'NRL':
            return self.NRL(**kwargs)
        if self.functype in ['none', 'strain']:
            return self.none(**kwargs)
        
    def none(self, atype_list, otype_list, **kwargs):
        out = th.zeros([len(atype_list), len(otype_list)], dtype=th.float32)

        for i, at in enumerate(atype_list):
            for j, ot in enumerate(otype_list):
                out[i,j] = self.E_base[at].get([ot], 0.)
        

        return out
    
    def uniform(self, atype_list, otype_list, nn_onsite_paras, **kwargs):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal onsite energies and returned.
        
        '''
        return nn_onsite_paras + self.none(atype_list, otype_list)
        

    def NRL(self, onsitenv_index, onsitenv_length, nn_onsite_paras, rcut:th.float32 = th.tensor(6), w:th.float32 = 0.1, lda=1.0):
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
        r_ijs = onsitenv_length.view(-1) # [N]
        exp_rij = th.exp(-lda**2 * r_ijs)
        f_rij = 1/(1+th.exp((r_ijs-rcut+5*w)/w))
        f_rij[r_ijs>=rcut] = 0.0
        rho_i = scatter(exp_rij * f_rij, onsitenv_index, 0, None, "sum").unsqueeze(1) # [N_atom, 1]
        a_l, b_l, c_l, d_l = nn_onsite_paras[:,:,0], nn_onsite_paras[:,:,1], nn_onsite_paras[:,:,2], nn_onsite_paras[:,:,3]
        E_il = a_l + b_l * rho_i**(2/3) + c_l * rho_i**(4/3) + d_l * rho_i**2 # [N_atom, n_orb]
        return E_il # [N_atom_n_orb]