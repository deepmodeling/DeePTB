import torch
import numpy as np
from scipy import constants
from torch_scatter import scatter_sum
from dptb.utils.constants import Bohr2Ang, Harte2eV
from dptb.data import AtomicDataDict
from dptb.data.transforms import OrbitalMapper
from typing import Union
from dptb.utils._xitorch.interpolate import Interp1D
from ase.data import atomic_numbers
import os
from scipy import interpolate



def calculate_atomic_rep(data: AtomicDataDict, 
                         idp_sk: OrbitalMapper, 
                         sigma_rep: dict):
    """
    Calculate repulsive term and return node energy and total repulsive energy.

    For node `i`, the energy is calculated as:
    E_i = Z_i * sum_j Z_j * [ 1/r_ij * (1 - erf(gamma_rep * r_ij)) ]
    where gamma_rep = 1/sqrt(sigma_i^2 + sigma_j^2), and Z_i, Z_j are the atomic numbers of atoms i and j respectively.
    The analytical form is short-ranged, so there is no need to include a cutoff function or Ewald summation.
    
    The total energy is then E_total = 0.5 * sum_i E_i.

    Parameters:
    -----------
    data : dict
        Dictionary containing atomic and edge information.
    idp_sk : OrbitalMapper
        Orbital mapper containing chemical_symbol_to_type mapping.
    sigma_rep : dict
        Dictionary of atomic sigma representations, e.g. {'B': 0.5, 'N': 0.5} in Angstrom units.
    
    Returns:
    --------
    node_rep_energy : torch.Tensor
        Energy of each node (atom), shape [n_atoms, 1].
    total_rep_energy : torch.Tensor
        Total repulsive energy (scalar).
    """
    # Convert atom dictionary to tensor
    chemical_symbol_to_type = idp_sk.chemical_symbol_to_type
    sigma_tensor = atom_dict_to_atomic_tensor(chemical_symbol_to_type, sigma_rep)
    
    # Get edge indices and atom types
    edge_index = data[AtomicDataDict.EDGE_TYPE_KEY].flatten()
    edge_number = idp_sk.untransform_bond(edge_index).T
    edge_atom_types = idp_sk.transform_atom(edge_number.flatten()).reshape(2, -1)

    # Calculate sigma_bond and gamma_rep
    sigma_bond = sigma_tensor[edge_atom_types[0]] ** 2 + sigma_tensor[edge_atom_types[1]] ** 2
    gamma_rep = Bohr2Ang * 1.0 / torch.sqrt(sigma_bond)  # Unit: 1 / Bohr = Ang / Bohr / Ang.

    # Calculate potential
    pot_edge = Repcurve._repulsive_core(
        r_ang = data[AtomicDataDict.EDGE_LENGTH_KEY],        # r in angstrom
        Zi = edge_number[0],                                 # Z_i: atomic number of atom i
        Zj = edge_number[1],                                 # Z_j: atomic number of atom j
        gamma_pair_bohr = gamma_rep                          # gamma in 1/Bohr
    ) # unit: eV
    
    # Aggregate to nodes using scatter_sum
    node_rep_energy = scatter_sum(pot_edge, data[AtomicDataDict.EDGE_INDEX_KEY][0], dim=0)
    # Total repulsive energy with a factor of 0.5 to avoid double counting
    total_rep_energy = 0.5 * torch.sum(node_rep_energy)
    
    return pot_edge, node_rep_energy.reshape([-1, 1]), total_rep_energy


def atom_dict_to_atomic_tensor(chemical_symbol_to_type, sigma_dict):
    """
    Convert atom dictionary to tensor representation.
    
    Parameters:
    -----------
    chemical_symbol_to_type : dict
        Mapping from chemical symbols to type indices.
    sigma_dict : dict
        Dictionary of atomic sigma values.
    
    Returns:
    --------
    sigma_tensors : torch.Tensor
        Tensor representation of atomic sigma values.
    """
    len_types_in_model = len(chemical_symbol_to_type)
    sigma_tensors = torch.zeros(len_types_in_model, dtype=torch.float32)
    for isym, ind in chemical_symbol_to_type.items():
        sigma_tensors[ind] = sigma_dict[isym]
    return sigma_tensors


def atom_tensor_to_atomic_dict(chemical_symbol_to_type, sigma_tensors):
    """
    Convert tensor representation back to atom dictionary.
    
    Parameters:
    -----------
    chemical_symbol_to_type : dict
        Mapping from chemical symbols to type indices.
    sigma_tensors : torch.Tensor
        Tensor representation of atomic sigma values.
    
    Returns:
    --------
    sigma_dict : dict
        Dictionary of atomic sigma values.
    """
    sigma_dict = {}
    for isym, ind in chemical_symbol_to_type.items():
        sigma_dict[isym] = sigma_tensors[ind]
    return sigma_dict


class Repcurve(object):
    def __init__(self,
                 element_symbols: list[str],
                 sigma_rep: dict,
                 r_min: float = 0.25,
                 r_max: float = 25,
                 num_points: int = 800,
                 device: Union[torch.device,str] = None,
                 dtype: torch.dtype = torch.float32):
        
        self.element_symbols = element_symbols
        self.sigma_rep = sigma_rep
        assert all([sym in sigma_rep for sym in element_symbols]), "All element symbols must be in sigma_rep."
        self.atomic_numbers = atomic_numbers

        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.num_points = int(num_points)
        self.device = device if device is not None else torch.device("cpu")
        if isinstance(self.device, str):
            self.device = torch.device(self.device)
        self.dtype = dtype

    def _get_sigma_pair(self, sym_i: str, sym_j: str) -> torch.Tensor:
        try:
            si = float(self.sigma_rep[sym_i])
            sj = float(self.sigma_rep[sym_j])
        except KeyError as e:
            raise KeyError(f" Element not found in sigma_rep.") from e
        # sqrt(si^2 + sj^2)
        return torch.sqrt(torch.tensor(si * si + sj * sj, dtype=self.dtype, device=self.device))

    def _get_Z_pair(self, sym_i: str, sym_j: str) -> tuple[int, int]:
        try:
            Zi = int(self.atomic_numbers[sym_i])
            Zj = int(self.atomic_numbers[sym_j])
        except Exception as e:
            raise KeyError(f" Element not found in symbol_to_Z.") from e
        return Zi, Zj

    @staticmethod
    def _repulsive_core(r_ang: torch.Tensor, 
                        Zi: int, 
                        Zj: int, 
                        gamma_pair_bohr: torch.Tensor) -> torch.Tensor:
        # r_ang: angstrom; gamma_pair_bohr: 1/Bohr
        # the repulsive core potential function is written in atomic units
        # minimum r to avoid NaN
        r_ang = torch.clamp(r_ang, min=1e-6)
        Vij = (Zi * Zj) * (Bohr2Ang / r_ang) * (1.0 - torch.erf(gamma_pair_bohr * (r_ang / Bohr2Ang))) # unit: 1/Bohr
        Vij = Vij * Harte2eV  # unit: eV
        return Vij

    def gamma_for_pair(self, sym_i: str, sym_j: str) -> torch.Tensor:
        # gamma = Bohr2Ang / sqrt(sigma_i^2 + sigma_j^2), sigma unit: angstrom
        sigma_pair = self._get_sigma_pair(sym_i, sym_j)  # angstrom
        return torch.tensor(Bohr2Ang / sigma_pair, dtype=self.dtype, device=self.device)   # 1/Bohr

    def sample_r(self, r_min: float | None = None, r_max: float | None = None, num_points: int | None = None) -> torch.Tensor:
        rmin = float(self.r_min if r_min is None else r_min)
        rmax = float(self.r_max if r_max is None else r_max)
        npts = int(self.num_points if num_points is None else num_points)
        return torch.linspace(rmin, rmax, npts, dtype=self.dtype, device=self.device)

    def pair_curve(self,
                   sym_i: str,
                   sym_j: str,
                   r: Union[torch.Tensor,np.ndarray,None] = None
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the repulsive core potential curve for a given pair of atomic symbols over a range of distances.

        Parameters:
        -----------
            sym_i (str): Symbol of the first atom (e.g., 'H', 'C').
            sym_j (str): Symbol of the second atom.
            r (torch.Tensor | np.ndarray | None, optional): Interatomic distances at which to evaluate the potential.
                If None, uses a default sampled range. Units are in Angstroms.

        Returns:
        -----------
            tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - r_t (torch.Tensor): The tensor of interatomic distances.
                - Vij (torch.Tensor): The tensor of repulsive core potential values corresponding to each distance.

        """

        assert sym_i in self.element_symbols, f"Element {sym_i} not in element_symbols."
        assert sym_j in self.element_symbols, f"Element {sym_j} not in element_symbols."

        if r is None:
            r_t = self.sample_r() # unit: Angstrom
        else:
            r_t = torch.as_tensor(r, dtype=self.dtype, device=self.device) # unit: Angstrom
        Zi, Zj = self._get_Z_pair(sym_i, sym_j)
        gamma_pair = self.gamma_for_pair(sym_i, sym_j)  # 1/Bohr
        Vij = self._repulsive_core(r_t, Zi, Zj, gamma_pair)
        # output units: r in Angstrom, V in eV
        return r_t, Vij

    def interp_cubic_spline(self,
                         sym_i: str,
                         sym_j: str,
                         r: torch.Tensor | np.ndarray | None = None,
                         rcut_min: float | None = None,
                         rcut_max: float | None = None,
                         bc_type: str = 'natural',
                         unit_E: str = 'eV',
                         unit_r: str = 'Ang'
                         ) -> interpolate.CubicSpline:   
        """
        Create a cubic-spline interpolant for the pair potential between two species.

        This routine queries the stored pair curve for the atom pair (sym_i, sym_j),
        optionally converts units, applies an optional radial cutoff window, and
        returns a scipy.interpolate.CubicSpline constructed from the remaining points.

        The default units are Angstrom for radius and eV for energy. When writing into 
        SK files, it's necessary to convert to Bohr for radius and Hartree for energy.

        Parameters
        ----------
        sym_i : str
            Chemical symbol or identifier for the first species.
        sym_j : str
            Chemical symbol or identifier for the second species.
        r : torch.Tensor | numpy.ndarray | None, optional
            Optional array of radii to pass through to self.pair_curve. If None,
            the full internal sampling for this pair is used. The array type returned
            by pair_curve (torch tensor or numpy array) is forwarded to the spline
            constructor.
        rcut_min : float | None, optional
            Minimum radius (inclusive). Points with r < rcut_min are removed.
            If None, no lower cutoff is applied.
        rcut_max : float | None, optional
            Maximum radius (inclusive). Points with r > rcut_max are removed.
            If None, no upper cutoff is applied.
        bc_type : str or tuple, optional
            Boundary condition passed directly to scipy.interpolate.CubicSpline.
            The default is 'natural'. See scipy.interpolate.CubicSpline documentation
            for supported values (e.g. 'natural', 'clamped', or explicit derivative
            tuples).
        unit_E : str, optional
            Unit of the energy values returned by pair_curve. Default 'eV'.
            If set to 'Hartree' the code divides the energies by the module
            constant Harte2eV (i.e. converts from eV to Hartree).
        unit_r : str, optional
            Unit of the radii returned by pair_curve. Default 'Ang' (Angstrom).
            If set to 'Bohr' the code divides radii by the module constant Bohr2Ang
            (i.e. converts from Angstrom to Bohr).

        Returns
        -------
        scipy.interpolate.CubicSpline
            A cubic-spline interpolator defined on the (possibly filtered and
            converted) radius and potential arrays. The returned object can be used
            to evaluate values and derivatives at arbitrary radii.
        """   
        r_t, v_t = self.pair_curve(sym_i, sym_j, r=r)
        if unit_E == 'Hartree':
            v_t = v_t / Harte2eV  # convert to Hartree unit
        if unit_r == 'Bohr':
            r_t = r_t / Bohr2Ang  # convert to Bohr unit

        if rcut_min is not None or rcut_max is not None:
            mask = torch.ones_like(r_t, dtype=torch.bool)
            if rcut_min is not None:
                mask &= (r_t >= float(rcut_min))
            if rcut_max is not None:
                mask &= (r_t <= float(rcut_max))
            r_t, v_t = r_t[mask], v_t[mask]

        # return Interp1D(x = r_t, y = v_t, method='cspline')
        return interpolate.CubicSpline(r_t, v_t, bc_type=bc_type)

    def write_rep_to_skfile(self,out_dir: str, element_symbols: list[str] | None = None):
        """
        Write repulsive potential data for element pairs to .skf spline files.
        For every pair of element symbols (sym_i, sym_j) a file named
        "{sym_i}-{sym_j}.skf" is created (or appended to) and a spline block is
        written. The spline data are produced by calling self.interp_cubic_spline(...)
        with energy units set to Hartree and distance units to Bohr.

        The detailed 'Spline' block structure can be found in :
        https://www.dftb.org/_downloads/85b02a0893bd3402438aec77de5bc1df/slakoformat.pdf

        Parameters
        ----------
        out_dir : str
            Path to the directory where .skf files will be written. The directory is
            created if it does not already exist.
        element_symbols : list[str] | None, optional
            Iterable of element symbols to export. If None, self.element_symbols is
            used. All provided symbols must be present in self.element_symbols.
        """
        
        if os.path.exists(out_dir) is False:
            os.makedirs(out_dir)
        elem_syms = self.element_symbols if element_symbols is None else element_symbols
        assert len(elem_syms) > 0, "element_symbols must contain at least one element."
        assert all([sym in self.element_symbols for sym in elem_syms]), "All element symbols must be in Repcurve.element_symbols."

        for sym_i in elem_syms:
            for sym_j in elem_syms:
                filename = f"{sym_i}-{sym_j}.skf"
                filepath = os.path.join(out_dir, filename)
                cspline_interp = self.interp_cubic_spline(sym_i, sym_j, 
                                                          unit_E='Hartree',
                                                          unit_r='Bohr')
                
                # add repulsive data to skf file
                with open(filepath, 'a') as f:
                    f.write("\nSpline\n")
                    nint = cspline_interp.x.shape[0] - 1 # number of intervals
                    coef = cspline_interp.c # coefficients shape (4, nint-1)
                    r_sec = cspline_interp.x # grid points
                    # number of intervals and last grid point
                    f.write(f"{nint}  {r_sec[-1]} \n") 
                    # when r < rmin, use the following exponential function ensuring V(rep->rmin) = exp_func(rmin)
                    f.write(f"1.0  0.0  {coef[3,0]-self.exp_func(r_sec[0],1.0,0.0,0.0)} \n")
                    # write spline sections
                    for i in range(nint-1):
                        f.write(f"{r_sec[i]} {r_sec[i+1]}  {coef[3,i]}  {coef[2,i]}  {coef[1,i]}  {coef[0,i]} \n")
                    f.write(f"{r_sec[nint-1]} {r_sec[nint]}  {coef[3,nint-1]}  {coef[2,nint-1]}  {coef[1,nint-1]}  {coef[0,nint-1]} 0.0 0.0 \n")


    @staticmethod
    def exp_func(r: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
        return np.exp(-A * r + B) + C