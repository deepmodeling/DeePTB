# define the integrals formula.
import torch as th
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from torch_runstats.scatter import scatter
from dptb.nn.sktb.socDB import soc_strength_database
from dptb.data.transforms import OrbitalMapper


class BaseSOC(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_socLs(self, **kwargs):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by type is called to cal onsite energies and returned.
        
        '''
        pass


class SOCFormula(BaseSOC):
    num_paras_dict = {
        'uniform': 1,
        "none": 0,
        "custom": None,
    }

    def __init__(
            self, 
            idp: Union[OrbitalMapper, None]=None,
            functype='none', 
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__() 
        if functype=='none':
            pass
        elif functype == 'uniform':
            assert hasattr(self, 'uniform')
        elif functype == 'custom':
            assert hasattr(self, 'custom')
        else:
            raise ValueError('No such SOC formula')
        
        self.functype = functype
        self.num_paras = self.num_paras_dict[functype]
        
        self.idp = idp
        if self.functype in ["none","uniform"]:
            self.socL_base = torch.zeros(self.idp.num_types, self.idp.n_onsite_socLs, dtype=dtype, device=device)
            for asym, idx in self.idp.chemical_symbol_to_type.items():
                self.socL_base[idx] = torch.zeros(self.idp.n_onsite_socLs)
                for ot in self.idp.basis[asym]:
                    fot = self.idp.basis_to_full_basis[asym][ot]
                    self.socL_base[idx][self.idp.skonsite_maps[fot+"-"+fot]] = soc_strength_database[asym][ot]
        
    def get_socLs(self, **kwargs):
        if self.functype == 'none':
            return self.none(**kwargs)
        elif self.functype == 'uniform':
            return self.uniform(**kwargs)
        elif self.functype == 'custom':
            return self.custom(**kwargs)
        else:
            raise ValueError('No such SOC formula')
        
    def none(self, atomic_numbers: torch.Tensor, **kwargs):
        """The none onsite soc function, the soc strength output is directly loaded from the onsite soc Database.
        Parameters
        ----------
        atomic_numbers : torch.Tensor(N)
            The atomic number list.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite soc strength by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)

        idx = self.idp.transform_atom(atomic_numbers)
        
        return self.socL_base[idx]
    
    def uniform(self, atomic_numbers: torch.Tensor, nn_soc_paras: torch.Tensor, **kwargs):
        """The uniform onsite soc function, that have the same onsite soc strength for one specific orbital of a atom type.

        Parameters
        ----------
        atomic_numbers : torch.Tensor(N) or torch.Tensor(N,1)
            The atomic number list.
        nn_onsite_paras : torch.Tensor(N_atom_type, n_orb)
            The nn fitted parameters for onsite soc strength.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite soc strength by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)
        if nn_soc_paras.shape[-1] == 1:
            nn_soc_paras = nn_soc_paras.squeeze(-1)
        
        # soc strength should be positive
        nn_soc_paras = torch.abs(nn_soc_paras)
        
        assert len(nn_soc_paras) == self.socL_base.shape[0]

        idx = self.idp.transform_atom(atomic_numbers)

        return nn_soc_paras[idx] + self.none(atomic_numbers=atomic_numbers)
        