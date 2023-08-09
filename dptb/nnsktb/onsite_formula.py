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

        elif functype == 'custom':
            self.functype = functype
            self.num_paras = None # defined by custom.
            assert hasattr(self, 'custom')
        else:
            raise ValueError('No such formula')
        
    def skEs(self, **kwargs):
        if self.functype == 'uniform':
            return self.uniform(**kwargs)
    
    def uniform(self,xtype, onsite_db, nn_onsite_paras, **kwargs):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by functype is called to cal onsite energies and returned.
        
        '''
        assert xtype in onsite_db.keys(), f'{xtype} is not in the onsite_db.'
        assert xtype in nn_onsite_paras.keys(), f'{xtype} is not in the nn_onsite_paras.'
        assert onsite_db[xtype].shape == nn_onsite_paras[xtype].shape, f'{xtype} onsite_db and nn_onsite_paras have different shape.'
        return onsite_db[xtype] + nn_onsite_paras[xtype]
        





