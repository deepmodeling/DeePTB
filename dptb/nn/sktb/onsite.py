# define the integrals formula.
import torch as th
import torch
from typing import List, Union
from abc import ABC, abstractmethod
from torch_runstats.scatter import scatter
from dptb.nn.sktb.onsiteDB import onsite_energy_database
from dptb.data.transforms import OrbitalMapper


class BaseOnsite(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_skEs(self, **kwargs):
        '''This is a wrap function for a self-defined formula of onsite energies. one can easily modify it into whatever form they want.
        
        Returns
        -------
            The function defined by type is called to cal onsite energies and returned.
        
        '''
        pass


class OnsiteFormula(BaseOnsite):
    num_paras_dict = {
        'uniform': 1,
        'uniform_noref': 1,
        'none': 0,
        'strain': 0,
        "NRL": 4,
        "dftb":1,
        "custom": None,
    }

    def __init__(
            self, 
            idp: Union[OrbitalMapper, None]=None,
            functype='none', 
            dtype: Union[str, torch.dtype] = torch.float32,
            device: Union[str, torch.device] = torch.device("cpu")) -> None:
        super().__init__() 
        if functype == 'strain':
            pass
        elif functype == 'none':
            assert hasattr(self, 'none')
        elif functype == 'uniform':
            assert hasattr(self, 'uniform')
        elif functype == 'uniform_noref':
            assert hasattr(self, 'uniform_noref')
        elif functype == 'NRL':
            assert hasattr(self, 'NRL')

        elif functype == 'custom':
            assert hasattr(self, 'custom')
        elif functype == 'dftb':
            assert hasattr(self, 'dftb')
        else:
            raise ValueError('No such formula')
        
        self.functype = functype
        self.num_paras = self.num_paras_dict[functype]
        
        self.idp = idp
        if self.functype in ["uniform", "none", "strain"]:
            self.E_base = torch.zeros(self.idp.num_types, self.idp.n_onsite_Es, dtype=dtype, device=device)
            for asym, idx in self.idp.chemical_symbol_to_type.items():
                self.E_base[idx] = torch.zeros(self.idp.n_onsite_Es)
                for ot in self.idp.basis[asym]:
                    fot = self.idp.basis_to_full_basis[asym][ot]
                    self.E_base[idx][self.idp.skonsite_maps[fot+"-"+fot]] = onsite_energy_database[asym][ot]

    def get_skEs(self, **kwargs):
        if self.functype == 'uniform':
            return self.uniform(**kwargs)
        if self.functype == 'uniform_noref':
            return self.uniform_noref(**kwargs)
        if self.functype == 'NRL':
            return self.NRL(**kwargs)
        if self.functype in ['none', 'strain']:
            return self.none(**kwargs)
        if self.functype == 'dftb':
            return self.dftb(**kwargs)
    
    def dftb(self, atomic_numbers: torch.Tensor, nn_onsite_paras: torch.Tensor, **kwargs):
        """The dftb onsite function, the energy output is directly loaded from the onsite Database.
        Parameters
        ----------
        atomic_numbers : torch.Tensor(N)
            The atomic number list.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite energies by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)

        if nn_onsite_paras.shape[-1] == 1:
            nn_onsite_paras = nn_onsite_paras.squeeze(-1)
        idx = self.idp.transform_atom(atomic_numbers)
        return nn_onsite_paras[idx]
    

    def none(self, atomic_numbers: torch.Tensor, **kwargs):
        """The none onsite function, the energy output is directly loaded from the onsite Database.
        Parameters
        ----------
        atomic_numbers : torch.Tensor(N)
            The atomic number list.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite energies by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)

        idx = self.idp.transform_atom(atomic_numbers)
        
        return self.E_base[idx]
    
    def uniform(self, atomic_numbers: torch.Tensor, nn_onsite_paras: torch.Tensor, **kwargs):
        """The uniform onsite function, that have the same onsite energies for one specific orbital of a atom type.

        Parameters
        ----------
        atomic_numbers : torch.Tensor(N) or torch.Tensor(N,1)
            The atomic number list.
        nn_onsite_paras : torch.Tensor(N_atom_type, n_orb)
            The nn fitted parameters for onsite energies.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite energies by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)
        if nn_onsite_paras.shape[-1] == 1:
            nn_onsite_paras = nn_onsite_paras.squeeze(-1)
        
        assert len(nn_onsite_paras) == self.E_base.shape[0]

        idx = self.idp.transform_atom(atomic_numbers)

        return nn_onsite_paras[idx] + self.none(atomic_numbers=atomic_numbers)

    def uniform_noref(self, atomic_numbers: torch.Tensor, nn_onsite_paras: torch.Tensor, **kwargs):
        """The uniform onsite function, that have the same onsite energies for one specific orbital of a atom type.

        Parameters
        ----------
        atomic_numbers : torch.Tensor(N) or torch.Tensor(N,1)
            The atomic number list.
        nn_onsite_paras : torch.Tensor(N_atom_type, n_orb)
            The nn fitted parameters for onsite energies.

        Returns
        -------
        torch.Tensor(N, n_orb)
            the onsite energies by composing results from nn and ones from database.
        """
        atomic_numbers = atomic_numbers.reshape(-1)
        if nn_onsite_paras.shape[-1] == 1:
            nn_onsite_paras = nn_onsite_paras.squeeze(-1)

        idx = self.idp.transform_atom(atomic_numbers)

        return nn_onsite_paras[idx]
             

    def NRL(self, atomic_numbers, onsitenv_index, onsitenv_length, nn_onsite_paras, rs:th.float32 = th.tensor(6), w:th.float32 = 0.1, lda=1.0, **kwargs):
        """ This is NRL-TB formula for onsite energies.

            rho_i = sum_j exp(- lda**2 r_ij) f(r_ij)
        
            E_il = a_l + b_l rho_i^(2/3) + c_l rho_i^(4/3) + d_l rho_i^2

            f(r_ij) = [1+exp((r_ij-rcut+5w)/w)]^-1;    (r_ij <  rcut)
                    = 0;                               (r_ij >= rcut)
        Parameters
        ----------
        onsitenv_index: torch.LongTensor
            env index shaped as [2, N]
        onsitenv_length: torch.Tensor
            env index shaped as [N] or [N,1]
        nn_onsite_paras: torch.Tensor
            [N, n_orb, 4]
        rcut: float
            the cutoff radius for onsite energies.
        w: float
            the decay for the  cutoff smoth function.
        lda: float
            the decay for the  calculateing rho.
        """ 
        atomic_numbers = atomic_numbers.reshape(-1)
        idx = self.idp.transform_atom(atomic_numbers)
        nn_onsite_paras = nn_onsite_paras[idx]
        r_ijs = onsitenv_length.view(-1) # [N]
        exp_rij = th.exp(-lda**2 * r_ijs)
        f_rij = 1/(1+th.exp((r_ijs-rs+5*w)/w))
        f_rij[r_ijs>=rs] = 0.0
        rho_i = scatter(src=exp_rij * f_rij, index=onsitenv_index[0], dim=0, reduce="sum").unsqueeze(1) # [N_atom, 1]
        a_l, b_l, c_l, d_l = nn_onsite_paras[:,:,0], nn_onsite_paras[:,:,1], nn_onsite_paras[:,:,2], nn_onsite_paras[:,:,3]
        E_il = a_l + b_l * rho_i**(2/3) + c_l * rho_i**(4/3) + d_l * rho_i**2 # [N_atom, n_orb]
        return E_il # [N_atom, n_orb]